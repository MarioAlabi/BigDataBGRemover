use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use image::{DynamicImage, GrayImage, ImageBuffer, imageops::FilterType};
use ndarray::{Array1, Array2, Array3, Array4};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::TensorRef;

use crate::model::{download_file, model_dir, shared_ort_env};

pub const SAM_ENCODER_URL: &str = "https://huggingface.co/aletrn/sam-quantized/resolve/b526b71ee5310cf6e45ac3a6de08f39ace060b30/machine_learning_models/mobile_sam.encoder.onnx";
pub const SAM_DECODER_URL: &str = "https://huggingface.co/aletrn/sam-quantized/resolve/b526b71ee5310cf6e45ac3a6de08f39ace060b30/machine_learning_models/mobile_sam.decoder.onnx";

pub fn sam_encoder_path() -> PathBuf {
    model_dir().join("mobile_sam.encoder.onnx")
}

pub fn sam_decoder_path() -> PathBuf {
    model_dir().join("mobile_sam.decoder.onnx")
}

pub fn sam_models_exist() -> bool {
    sam_encoder_path().exists() && sam_decoder_path().exists()
}

pub fn download_sam_models(
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(&str, u64, Option<u64>),
) -> Result<(), String> {
    download_file(SAM_ENCODER_URL, &sam_encoder_path(), cancel, |done, tot| {
        on_progress("MobileSAM Encoder", done, tot);
    })?;
    download_file(SAM_DECODER_URL, &sam_decoder_path(), cancel, |done, tot| {
        on_progress("MobileSAM Decoder", done, tot);
    })?;
    Ok(())
}

#[derive(Clone)]
pub struct SamEmbedding {
    pub data: Array4<f32>,
    pub scale: f32,
    pub orig_w: u32,
    pub orig_h: u32,
}

pub struct SamSession {
    _env: Arc<Environment>,
    encoder: Session,
    decoder: Session,
}

impl SamSession {
    pub fn load() -> Result<Self, String> {
        let env = shared_ort_env()?;

        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(4);

        let encoder = Session::builder(&env)
            .map_err(|e| format!("Error creando encoder SAM: {e}"))?
            .with_intra_threads(threads)
            .map_err(|e| format!("Error hilos encoder SAM: {e}"))?
            .commit_from_file(sam_encoder_path())
            .map_err(|e| format!("No se pudo cargar mobile_sam.encoder.onnx: {e}"))?;

        let decoder = Session::builder(&env)
            .map_err(|e| format!("Error creando decoder SAM: {e}"))?
            .with_intra_threads(threads)
            .map_err(|e| format!("Error hilos decoder SAM: {e}"))?
            .commit_from_file(sam_decoder_path())
            .map_err(|e| format!("No se pudo cargar mobile_sam.decoder.onnx: {e}"))?;

        Ok(Self {
            _env: env,
            encoder,
            decoder,
        })
    }

    pub fn encode_image(&mut self, img: &DynamicImage) -> Result<SamEmbedding, String> {
        let orig_w = img.width();
        let orig_h = img.height();
        let target = 1024u32;

        let scale = target as f32 / orig_w.max(orig_h) as f32;
        let new_w = ((orig_w as f32 * scale).round() as u32).clamp(1, target);
        let new_h = ((orig_h as f32 * scale).round() as u32).clamp(1, target);

        let resized = img.resize_exact(new_w, new_h, FilterType::Triangle).to_rgb8();

        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];

        let mut input = Array4::<f32>::zeros((1, 3, target as usize, target as usize));
        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let p = resized.get_pixel(x as u32, y as u32);
                input[[0, 0, y, x]] = (p[0] as f32 / 255.0 - MEAN[0]) / STD[0];
                input[[0, 1, y, x]] = (p[1] as f32 / 255.0 - MEAN[1]) / STD[1];
                input[[0, 2, y, x]] = (p[2] as f32 / 255.0 - MEAN[2]) / STD[2];
            }
        }

        let input_tensor = TensorRef::from_array_view(input.view())
            .map_err(|e| format!("Error tensor encoder SAM: {e}"))?;

        let outputs = self
            .encoder
            .run(ort::inputs![input_tensor])
            .map_err(|e| format!("Error ejecutando encoder SAM: {e}"))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Error leyendo embedding SAM: {e}"))?;

        if shape.len() != 4 {
            return Err(format!("Forma de embedding SAM inesperada: {shape:?}"));
        }

        let emb = Array4::from_shape_vec(
            (
                shape[0] as usize,
                shape[1] as usize,
                shape[2] as usize,
                shape[3] as usize,
            ),
            data.to_vec(),
        )
        .map_err(|e| format!("Error reconstruyendo embedding SAM: {e}"))?;

        Ok(SamEmbedding {
            data: emb,
            scale,
            orig_w,
            orig_h,
        })
    }

    /// Runs MobileSAM decoder for a single click `(x, y)` in original image coordinates.
    pub fn predict_point(&mut self, emb: &SamEmbedding, x: f32, y: f32) -> Result<GrayImage, String> {
        // Standard SAM ONNX format appends a padding point (0, 0) with label -1.
        let mut coords = Array3::<f32>::zeros((1, 2, 2));
        coords[[0, 0, 0]] = x * emb.scale;
        coords[[0, 0, 1]] = y * emb.scale;

        let mut labels = Array2::<f32>::zeros((1, 2));
        labels[[0, 0]] = 1.0;
        labels[[0, 1]] = -1.0;

        let mask_input = Array4::<f32>::zeros((1, 1, 256, 256));
        let has_mask_input = Array1::<f32>::zeros(1);
        let orig_im_size = Array1::from_vec(vec![emb.orig_h as f32, emb.orig_w as f32]);

        let emb_t = TensorRef::from_array_view(emb.data.view())
            .map_err(|e| format!("Error embedding SAM: {e}"))?;
        let coords_t = TensorRef::from_array_view(coords.view())
            .map_err(|e| format!("Error point_coords SAM: {e}"))?;
        let labels_t = TensorRef::from_array_view(labels.view())
            .map_err(|e| format!("Error point_labels SAM: {e}"))?;
        let mask_t = TensorRef::from_array_view(mask_input.view())
            .map_err(|e| format!("Error mask_input SAM: {e}"))?;
        let has_mask_t = TensorRef::from_array_view(has_mask_input.view())
            .map_err(|e| format!("Error has_mask_input SAM: {e}"))?;
        let size_t = TensorRef::from_array_view(orig_im_size.view())
            .map_err(|e| format!("Error orig_im_size SAM: {e}"))?;

        let outputs = self
            .decoder
            .run(ort::inputs![
                "image_embeddings" => emb_t,
                "point_coords" => coords_t,
                "point_labels" => labels_t,
                "mask_input" => mask_t,
                "has_mask_input" => has_mask_t,
                "orig_im_size" => size_t,
            ])
            .map_err(|e| format!("Error ejecutando decoder SAM: {e}"))?;

        let (mask_shape, mask_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Error leyendo máscara SAM: {e}"))?;

        if mask_shape.len() != 4 {
            return Err(format!("Forma de máscara SAM inesperada: {mask_shape:?}"));
        }

        let num_masks = mask_shape[1] as usize;
        let mh = mask_shape[2] as usize;
        let mw = mask_shape[3] as usize;

        // Pick the mask with the highest IoU score if multiple masks are returned.
        let mut best_idx = 0usize;
        if num_masks > 1 && outputs.len() > 1 {
            if let Ok((_, iou_data)) = outputs[1].try_extract_tensor::<f32>() {
                let mut best_iou = f32::NEG_INFINITY;
                for (i, &score) in iou_data.iter().take(num_masks).enumerate() {
                    if score > best_iou {
                        best_iou = score;
                        best_idx = i;
                    }
                }
            }
        }

        let offset = best_idx * mh * mw;
        let slice = &mask_data[offset..offset + mh * mw];

        let mut buf = vec![0u8; mh * mw];
        for (dst, &logit) in buf.iter_mut().zip(slice) {
            let prob = 1.0 / (1.0 + (-logit).exp());
            *dst = if prob >= 0.5 { 255 } else { 0 };
        }

        let raw_mask: GrayImage = ImageBuffer::from_raw(mw as u32, mh as u32, buf)
            .ok_or_else(|| "No se pudo construir máscara SAM".to_string())?;

        if mw as u32 != emb.orig_w || mh as u32 != emb.orig_h {
            Ok(image::imageops::resize(
                &raw_mask,
                emb.orig_w,
                emb.orig_h,
                FilterType::Triangle,
            ))
        } else {
            Ok(raw_mask)
        }
    }
}
