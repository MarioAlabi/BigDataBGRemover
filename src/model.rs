use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicBool, Ordering};

use image::{DynamicImage, GrayImage, ImageBuffer, imageops::FilterType};
use ndarray::Array4;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::TensorRef;

pub const MODEL_FILENAME: &str = "birefnet-general.onnx";
pub const MODEL_URL: &str =
    "https://huggingface.co/withoutbg/withoutbg-openweights-onnx/resolve/main/birefnet-general.onnx";

pub fn model_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("bigdata-bg-remover")
        .join("models")
}

pub fn default_model_path() -> PathBuf {
    model_dir().join(MODEL_FILENAME)
}

/// Downloads a file from `url` to `dest`, reporting `(bytes_downloaded, total_bytes_if_known)`.
pub fn download_file(
    url: &str,
    dest: &Path,
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("No se pudo crear carpeta: {e}"))?;
    }

    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("Error de red al descargar el modelo: {e}"))?;

    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let tmp_path = dest.with_extension("onnx.part");
    let mut file =
        File::create(&tmp_path).map_err(|e| format!("No se pudo crear archivo temporal: {e}"))?;

    let mut reader = response.into_body().into_reader();
    let mut buf = [0u8; 64 * 1024];
    let mut downloaded: u64 = 0;

    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = fs::remove_file(&tmp_path);
            return Err("Descarga cancelada".to_string());
        }
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("Error leyendo descarga: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("Error escribiendo modelo: {e}"))?;
        downloaded += n as u64;
        on_progress(downloaded, total);
    }

    file.flush()
        .map_err(|e| format!("Error guardando modelo: {e}"))?;
    drop(file);

    fs::rename(&tmp_path, dest).map_err(|e| format!("Error al finalizar archivo: {e}"))?;
    Ok(())
}

/// Downloads the BiRefNet ONNX model to `dest`.
pub fn download_model(
    dest: &Path,
    cancel: &AtomicBool,
    on_progress: impl FnMut(u64, Option<u64>),
) -> Result<(), String> {
    download_file(MODEL_URL, dest, cancel, on_progress)
}

static ORT_ENV: OnceLock<Result<Arc<Environment>, String>> = OnceLock::new();

pub fn shared_ort_env() -> Result<Arc<Environment>, String> {
    ORT_ENV
        .get_or_init(|| {
            ort::init()
                .with_name("bigdata-bg-remover")
                .build()
                .map(Arc::new)
                .map_err(|e| format!("Error iniciando ONNX Runtime: {e}"))
        })
        .clone()
}

pub struct BgRemoverSession {
    _env: Arc<Environment>,
    session: Session,
    input_w: u32,
    input_h: u32,
}

impl BgRemoverSession {
    pub fn load(model_path: &Path) -> Result<Self, String> {
        let env = shared_ort_env()?;

        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(4);

        let session = Session::builder(&env)
            .map_err(|e| format!("Error creando sesión ONNX: {e}"))?
            .with_intra_threads(threads)
            .map_err(|e| format!("Error configurando hilos: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("No se pudo cargar el modelo {}: {e}", model_path.display()))?;

        // BiRefNet general typically expects 1024x1024; detect if fixed dimensions are present.
        let (mut input_h, mut input_w) = (1024u32, 1024u32);
        if let Some(input) = session.inputs().first()
            && let Some(shape) = input.dtype().tensor_shape()
            && shape.len() == 4
        {
            if shape[2] > 0 {
                input_h = shape[2] as u32;
            }
            if shape[3] > 0 {
                input_w = shape[3] as u32;
            }
        }

        Ok(Self {
            _env: env,
            session,
            input_w,
            input_h,
        })
    }

    /// Runs BiRefNet inference and returns an alpha mask at the original image resolution.
    pub fn remove_background(&mut self, img: &DynamicImage) -> Result<GrayImage, String> {
        let orig_w = img.width();
        let orig_h = img.height();

        let resized = img
            .resize_exact(self.input_w, self.input_h, FilterType::Triangle)
            .to_rgb8();

        let h = self.input_h as usize;
        let w = self.input_w as usize;
        let mut input_array = Array4::<f32>::zeros((1, 3, h, w));

        // ImageNet normalization
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];

        for y in 0..h {
            for x in 0..w {
                let p = resized.get_pixel(x as u32, y as u32);
                input_array[[0, 0, y, x]] = (p[0] as f32 / 255.0 - MEAN[0]) / STD[0];
                input_array[[0, 1, y, x]] = (p[1] as f32 / 255.0 - MEAN[1]) / STD[1];
                input_array[[0, 2, y, x]] = (p[2] as f32 / 255.0 - MEAN[2]) / STD[2];
            }
        }

        let input_tensor = TensorRef::from_array_view(input_array.view())
            .map_err(|e| format!("Error preparando tensor de entrada: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![input_tensor])
            .map_err(|e| format!("Error ejecutando inferencia: {e}"))?;

        let (out_shape, out_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Error leyendo salida del modelo: {e}"))?;

        let (out_h, out_w) = match out_shape.len() {
            4 => (out_shape[2] as usize, out_shape[3] as usize),
            3 => (out_shape[1] as usize, out_shape[2] as usize),
            2 => (out_shape[0] as usize, out_shape[1] as usize),
            _ => (h, w),
        };

        if out_data.len() < out_h * out_w {
            return Err("La salida del modelo tiene un tamaño inesperado".to_string());
        }

        // Check if values are logits (outside [0, 1]) and apply sigmoid if needed
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        for &v in &out_data[..out_h * out_w] {
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        let apply_sigmoid = min_v < -0.05 || max_v > 1.05;

        let mut mask_buf = vec![0u8; out_h * out_w];
        for (dst, &raw) in mask_buf.iter_mut().zip(&out_data[..out_h * out_w]) {
            let prob = if apply_sigmoid {
                1.0 / (1.0 + (-raw).exp())
            } else {
                raw
            };
            *dst = (prob.clamp(0.0, 1.0) * 255.0).round() as u8;
        }

        let small_mask: GrayImage = ImageBuffer::from_raw(out_w as u32, out_h as u32, mask_buf)
            .ok_or_else(|| "No se pudo construir la máscara de salida".to_string())?;

        let full_mask =
            image::imageops::resize(&small_mask, orig_w, orig_h, FilterType::Triangle);

        Ok(full_mask)
    }
}
