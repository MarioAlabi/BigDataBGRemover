mod editor;
mod model;
mod sam;

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread;

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions};
use image::{DynamicImage, GrayImage, ImageBuffer, RgbaImage};

use editor::ImageEditor;
use model::{BgRemoverSession, default_model_path, download_model};
use sam::{SamEmbedding, SamSession, download_sam_models, sam_models_exist};

#[derive(Clone, Copy, PartialEq, Eq)]
enum ToolMode {
    View,
    AiKeep,
    AiErase,
    BrushKeep,
    BrushErase,
}

enum BgMessage {
    DownloadProgress(String, u64, Option<u64>),
    DownloadDone(Result<PathBuf, String>),
    SamDownloadDone(Result<(), String>),
    SingleDone(Result<(String, RgbaImage, GrayImage), String>),
    SamClickDone(Result<(GrayImage, bool, SamEmbedding, SamSession), String>),
    BatchProgress(usize, usize, String),
    BatchDone(usize, usize, Vec<String>),
}

struct BigDataBgApp {
    model_path: PathBuf,
    model_exists: bool,
    sam_exists: bool,
    session: Option<BgRemoverSession>,
    sam_session: Option<SamSession>,
    sam_embedding: Option<SamEmbedding>,

    editor: Option<ImageEditor>,
    current_filename: Option<String>,
    preview_tex: Option<TextureHandle>,
    show_original: bool,

    tool: ToolMode,
    brush_radius: f32,
    is_painting: bool,
    last_paint_pos: Option<(f32, f32)>,

    busy_label: Option<String>,
    download_progress: Option<(String, u64, Option<u64>)>,
    download_cancel: Arc<AtomicBool>,
    status_msg: Option<(String, bool)>,

    rx: Option<Receiver<BgMessage>>,
}

impl BigDataBgApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_modern_theme(&cc.egui_ctx);
        let model_path = default_model_path();
        let model_exists = model_path.exists();
        let sam_exists = sam_models_exist();

        Self {
            model_path,
            model_exists,
            sam_exists,
            session: None,
            sam_session: None,
            sam_embedding: None,
            editor: None,
            current_filename: None,
            preview_tex: None,
            show_original: false,
            tool: ToolMode::View,
            brush_radius: 30.0,
            is_painting: false,
            last_paint_pos: None,
            busy_label: None,
            download_progress: None,
            download_cancel: Arc::new(AtomicBool::new(false)),
            status_msg: None,
            rx: None,
        }
    }

    fn is_busy(&self) -> bool {
        self.busy_label.is_some() || self.download_progress.is_some()
    }

    fn start_download(&mut self, ctx: &egui::Context) {
        let dest = self.model_path.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        self.download_cancel = Arc::clone(&cancel);
        self.download_progress = Some(("Descargando BiRefNet...".to_string(), 0, None));
        self.status_msg = None;

        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        let ctx = ctx.clone();

        thread::spawn(move || {
            let res = download_model(&dest, &cancel, |done, total| {
                let _ = tx.send(BgMessage::DownloadProgress(
                    "Descargando BiRefNet...".to_string(),
                    done,
                    total,
                ));
                ctx.request_repaint();
            });
            let _ = tx.send(match res {
                Ok(()) => BgMessage::DownloadDone(Ok(dest)),
                Err(e) => BgMessage::DownloadDone(Err(e)),
            });
            ctx.request_repaint();
        });
    }

    fn start_sam_download(&mut self, ctx: &egui::Context) {
        let cancel = Arc::new(AtomicBool::new(false));
        self.download_cancel = Arc::clone(&cancel);
        self.download_progress = Some(("Descargando MobileSAM (~40 MB)...".to_string(), 0, None));
        self.status_msg = None;

        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        let ctx = ctx.clone();

        thread::spawn(move || {
            let res = download_sam_models(&cancel, |part, done, total| {
                let _ = tx.send(BgMessage::DownloadProgress(
                    format!("Descargando {part}..."),
                    done,
                    total,
                ));
                ctx.request_repaint();
            });
            let _ = tx.send(BgMessage::SamDownloadDone(res));
            ctx.request_repaint();
        });
    }

    fn process_dynamic_image(&mut self, name: String, img: DynamicImage, ctx: &egui::Context) {
        if !self.model_exists {
            self.status_msg = Some((
                "Primero descarga o selecciona el modelo BiRefNet (.onnx).".to_string(),
                true,
            ));
            return;
        }

        let model_path = self.model_path.clone();
        let existing_session = self.session.take();
        self.sam_embedding = None;

        self.busy_label = Some(format!("Quitando fondo de {name}..."));
        self.status_msg = None;

        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        let ctx = ctx.clone();

        thread::spawn(move || {
            let work = || -> Result<(String, RgbaImage, GrayImage), String> {
                let mut sess = match existing_session {
                    Some(s) => s,
                    None => BgRemoverSession::load(&model_path)?,
                };
                let mask = sess.remove_background(&img)?;
                Ok((name, img.to_rgba8(), mask))
            };

            let _ = tx.send(BgMessage::SingleDone(work()));
            ctx.request_repaint();
        });
    }

    fn process_single_image(&mut self, path: PathBuf, ctx: &egui::Context) {
        let filename = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imagen")
            .to_string();

        match image::open(&path) {
            Ok(img) => self.process_dynamic_image(filename, img, ctx),
            Err(e) => {
                self.status_msg = Some((format!("No se pudo abrir la imagen: {e}"), true));
            }
        }
    }

    fn paste_from_clipboard(&mut self, ctx: &egui::Context) {
        if !self.model_exists || self.is_busy() {
            return;
        }
        let Ok(mut cb) = arboard::Clipboard::new() else {
            self.status_msg = Some(("No se pudo acceder al portapapeles.".to_string(), true));
            return;
        };
        match cb.get_image() {
            Ok(img_data) => {
                let w = img_data.width as u32;
                let h = img_data.height as u32;
                let bytes = img_data.bytes.into_owned();
                if let Some(rgba) = ImageBuffer::from_raw(w, h, bytes) {
                    self.process_dynamic_image(
                        "portapapeles".to_string(),
                        DynamicImage::ImageRgba8(rgba),
                        ctx,
                    );
                } else {
                    self.status_msg = Some((
                        "Formato de imagen del portapapeles no compatible.".to_string(),
                        true,
                    ));
                }
            }
            Err(_) => {
                self.status_msg = Some((
                    "No hay ninguna imagen copiada en el portapapeles.".to_string(),
                    true,
                ));
            }
        }
    }

    fn run_sam_click(&mut self, ix: f32, iy: f32, keep: bool, ctx: &egui::Context) {
        let Some(ed) = &self.editor else { return };
        let orig = ed.original.clone();
        let existing_emb = self.sam_embedding.clone();
        let existing_sam = self.sam_session.take();

        self.busy_label = Some(if keep {
            "Analizando zona a incluir con IA...".to_string()
        } else {
            "Analizando zona a quitar con IA...".to_string()
        });
        self.status_msg = None;

        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        let ctx = ctx.clone();

        thread::spawn(move || {
            let work = || -> Result<(GrayImage, bool, SamEmbedding, SamSession), String> {
                let mut sam = match existing_sam {
                    Some(s) => s,
                    None => SamSession::load()?,
                };
                let emb = match existing_emb {
                    Some(e) => e,
                    None => sam.encode_image(&DynamicImage::ImageRgba8(orig))?,
                };
                let mask = sam.predict_point(&emb, ix, iy)?;
                Ok((mask, keep, emb, sam))
            };

            let _ = tx.send(BgMessage::SamClickDone(work()));
            ctx.request_repaint();
        });
    }

    fn process_batch(&mut self, paths: Vec<PathBuf>, out_dir: PathBuf, ctx: &egui::Context) {
        if paths.is_empty() {
            return;
        }
        let model_path = self.model_path.clone();
        let existing_session = self.session.take();
        let total = paths.len();

        self.busy_label = Some(format!("Procesando lote (0 / {total})..."));
        self.status_msg = None;

        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        let ctx = ctx.clone();

        thread::spawn(move || {
            let mut sess = match existing_session {
                Some(s) => s,
                None => match BgRemoverSession::load(&model_path) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx.send(BgMessage::BatchDone(0, total, vec![e]));
                        ctx.request_repaint();
                        return;
                    }
                },
            };

            let mut ok_count = 0usize;
            let mut errors = Vec::new();

            for (idx, path) in paths.iter().enumerate() {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("imagen")
                    .to_string();
                let _ = tx.send(BgMessage::BatchProgress(idx + 1, total, stem.clone()));
                ctx.request_repaint();

                match image::open(path) {
                    Ok(img) => match sess.remove_background(&img) {
                        Ok(mask) => {
                            let ed = ImageEditor::new(img.to_rgba8(), mask);
                            let out = ed.export_rgba();
                            let out_path = out_dir.join(format!("{stem}_sin_fondo.png"));
                            if let Err(e) = out.save(&out_path) {
                                errors.push(format!("{stem}: error al guardar ({e})"));
                            } else {
                                ok_count += 1;
                            }
                        }
                        Err(e) => errors.push(format!("{stem}: {e}")),
                    },
                    Err(e) => errors.push(format!("{stem}: no se pudo abrir ({e})")),
                }
            }

            let _ = tx.send(BgMessage::BatchDone(ok_count, total, errors));
            ctx.request_repaint();
        });
    }

    fn refresh_texture(&mut self, ctx: &egui::Context) {
        if let Some(ed) = &self.editor {
            let color_img = ed.build_preview(self.show_original);
            self.preview_tex =
                Some(ctx.load_texture("preview", color_img, TextureOptions::LINEAR));
        }
    }

    fn poll_messages(&mut self, ctx: &egui::Context) {
        let mut clear_rx = false;
        let mut need_refresh = false;

        if let Some(rx) = &self.rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    BgMessage::DownloadProgress(label, done, total) => {
                        self.download_progress = Some((label, done, total));
                    }
                    BgMessage::DownloadDone(res) => {
                        self.download_progress = None;
                        clear_rx = true;
                        match res {
                            Ok(path) => {
                                self.model_path = path;
                                self.model_exists = true;
                                self.status_msg = Some((
                                    "Modelo BiRefNet descargado y listo.".to_string(),
                                    false,
                                ));
                            }
                            Err(e) => {
                                self.status_msg = Some((e, true));
                            }
                        }
                    }
                    BgMessage::SamDownloadDone(res) => {
                        self.download_progress = None;
                        clear_rx = true;
                        match res {
                            Ok(()) => {
                                self.sam_exists = sam_models_exist();
                                self.status_msg = Some((
                                    "MobileSAM descargado. Ya puedes hacer clic en la imagen para incluir o quitar objetos.".to_string(),
                                    false,
                                ));
                            }
                            Err(e) => {
                                self.status_msg = Some((e, true));
                            }
                        }
                    }
                    BgMessage::SingleDone(res) => {
                        self.busy_label = None;
                        clear_rx = true;
                        match res {
                            Ok((name, orig, mask)) => {
                                self.current_filename = Some(name);
                                self.editor = Some(ImageEditor::new(orig, mask));
                                self.show_original = false;
                                self.tool = ToolMode::View;
                                need_refresh = true;
                                self.status_msg = Some((
                                    "Fondo eliminado. Usa 'Incluir/Quitar con IA' (clics), el pincel manual o guarda el PNG.".to_string(),
                                    false,
                                ));
                            }
                            Err(e) => {
                                self.status_msg = Some((e, true));
                            }
                        }
                    }
                    BgMessage::SamClickDone(res) => {
                        self.busy_label = None;
                        clear_rx = true;
                        match res {
                            Ok((sam_mask, keep, emb, sam)) => {
                                self.sam_embedding = Some(emb);
                                self.sam_session = Some(sam);
                                if let Some(ed) = &mut self.editor {
                                    ed.push_undo();
                                    ed.merge_sam_mask(&sam_mask, keep);
                                    need_refresh = true;
                                }
                                self.status_msg = Some((
                                    if keep {
                                        "Zona incluida con IA (Ctrl+Z para deshacer).".to_string()
                                    } else {
                                        "Zona quitada con IA (Ctrl+Z para deshacer).".to_string()
                                    },
                                    false,
                                ));
                            }
                            Err(e) => {
                                self.status_msg = Some((e, true));
                            }
                        }
                    }
                    BgMessage::BatchProgress(cur, total, name) => {
                        self.busy_label =
                            Some(format!("Procesando lote ({cur} / {total}): {name}..."));
                    }
                    BgMessage::BatchDone(ok, total, errors) => {
                        self.busy_label = None;
                        clear_rx = true;
                        if errors.is_empty() {
                            self.status_msg = Some((
                                format!("Lote completado: {ok} de {total} imágenes guardadas."),
                                false,
                            ));
                        } else {
                            self.status_msg = Some((
                                format!(
                                    "Lote terminado ({ok}/{total} exitosas). Primer error: {}",
                                    errors[0]
                                ),
                                true,
                            ));
                        }
                    }
                }
            }
        }

        if clear_rx {
            self.rx = None;
        }
        if need_refresh {
            self.refresh_texture(ctx);
        }
    }

    fn save_current(&mut self) {
        let Some(ed) = &self.editor else { return };
        let default_name = format!(
            "{}_sin_fondo.png",
            self.current_filename.as_deref().unwrap_or("imagen")
        );

        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Imagen PNG", &["png"])
            .set_file_name(&default_name)
            .save_file()
        {
            let rgba = ed.export_rgba();
            match rgba.save(&path) {
                Ok(()) => {
                    self.status_msg =
                        Some((format!("Guardado en {}", path.display()), false));
                }
                Err(e) => {
                    self.status_msg = Some((format!("Error al guardar: {e}"), true));
                }
            }
        }
    }
}

fn is_supported_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp")
    )
}

impl eframe::App for BigDataBgApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_messages(ctx);

        // Ctrl+Z -> Undo
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Z))
            && let Some(ed) = &mut self.editor
            && ed.undo()
        {
            self.refresh_texture(ctx);
        }

        // Ctrl+V -> Paste image from clipboard
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::V)) {
            self.paste_from_clipboard(ctx);
        }

        // Drag-and-drop files
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .filter(|p| is_supported_image(p))
                .collect()
        });
        if !dropped.is_empty() && !self.is_busy() {
            if dropped.len() == 1 {
                self.process_single_image(dropped[0].clone(), ctx);
            } else if let Some(out_dir) = rfd::FileDialog::new()
                .set_title("Elige la carpeta donde guardar las imágenes sin fondo")
                .pick_folder()
            {
                self.process_batch(dropped, out_dir, ctx);
            }
        }

        // Top toolbar
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.heading("BigDataBGRemover");
                ui.separator();

                let can_act = self.model_exists && !self.is_busy();

                if ui
                    .add_enabled(can_act, egui::Button::new("📂 Abrir imagen"))
                    .clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("Imágenes", &["png", "jpg", "jpeg", "webp"])
                        .pick_file()
                {
                    self.process_single_image(path, ctx);
                }

                if ui
                    .add_enabled(can_act, egui::Button::new("📋 Pegar (Ctrl+V)"))
                    .clicked()
                {
                    self.paste_from_clipboard(ctx);
                }

                if ui
                    .add_enabled(can_act, egui::Button::new("📁 Procesar lote"))
                    .clicked()
                    && let Some(files) = rfd::FileDialog::new()
                        .add_filter("Imágenes", &["png", "jpg", "jpeg", "webp"])
                        .pick_files()
                    && !files.is_empty()
                {
                    if files.len() == 1 {
                        self.process_single_image(files[0].clone(), ctx);
                    } else if let Some(out_dir) = rfd::FileDialog::new()
                        .set_title("Elige la carpeta de salida")
                        .pick_folder()
                    {
                        self.process_batch(files, out_dir, ctx);
                    }
                }

                let has_editor = self.editor.is_some() && !self.is_busy();
                if ui
                    .add_enabled(has_editor, egui::Button::new("💾 Guardar PNG"))
                    .clicked()
                {
                    self.save_current();
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.model_exists {
                        ui.label(
                            egui::RichText::new("✅ BiRefNet listo")
                                .color(Color32::from_rgb(100, 210, 120)),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new("⚠ Falta modelo .onnx")
                                .color(Color32::from_rgb(240, 180, 70)),
                        );
                    }
                });
            });
            ui.add_space(4.0);
        });

        // Secondary toolbar for AI 2nd-pass + manual brush
        if self.editor.is_some() {
            egui::TopBottomPanel::top("edit_bar").show(ctx, |ui| {
                ui.add_space(3.0);
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(&mut self.tool, ToolMode::View, "👁 Ver");
                    ui.separator();

                    ui.label("2ª Pasada IA (Clic):");
                    ui.selectable_value(
                        &mut self.tool,
                        ToolMode::AiKeep,
                        egui::RichText::new("✨➕ Incluir con IA")
                            .color(Color32::from_rgb(120, 220, 255)),
                    );
                    ui.selectable_value(
                        &mut self.tool,
                        ToolMode::AiErase,
                        egui::RichText::new("✨➖ Quitar con IA")
                            .color(Color32::from_rgb(255, 165, 90)),
                    );

                    if !self.sam_exists
                        && matches!(self.tool, ToolMode::AiKeep | ToolMode::AiErase)
                        && !self.is_busy()
                        && ui.button("⬇ Descargar MobileSAM (~40 MB)").clicked()
                    {
                        self.start_sam_download(ctx);
                    }

                    ui.separator();
                    ui.label("Pincel:");
                    ui.selectable_value(
                        &mut self.tool,
                        ToolMode::BrushKeep,
                        egui::RichText::new("➕ Conservar")
                            .color(Color32::from_rgb(90, 220, 120)),
                    );
                    ui.selectable_value(
                        &mut self.tool,
                        ToolMode::BrushErase,
                        egui::RichText::new("➖ Borrar").color(Color32::from_rgb(240, 100, 100)),
                    );

                    if matches!(self.tool, ToolMode::BrushKeep | ToolMode::BrushErase) {
                        ui.add(
                            egui::Slider::new(&mut self.brush_radius, 5.0..=150.0).suffix(" px"),
                        );
                    }

                    ui.separator();
                    let can_undo = self.editor.as_ref().is_some_and(|e| e.can_undo());
                    if ui
                        .add_enabled(can_undo, egui::Button::new("↩ Deshacer"))
                        .clicked()
                        && let Some(ed) = &mut self.editor
                        && ed.undo()
                    {
                        self.refresh_texture(ctx);
                    }

                    if ui
                        .checkbox(&mut self.show_original, "Ver original")
                        .changed()
                    {
                        self.refresh_texture(ctx);
                    }
                });
                ui.add_space(3.0);
            });
        }

        // Bottom status bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some((msg, is_err)) = &self.status_msg {
                    let color = if *is_err {
                        Color32::from_rgb(240, 100, 100)
                    } else {
                        Color32::from_rgb(170, 210, 240)
                    };
                    ui.label(egui::RichText::new(msg).color(color));
                } else {
                    ui.label("Listo. Arrastra una imagen, pégala (Ctrl+V) o haz clic en Abrir imagen.");
                }
            });
        });

        // Main workspace
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some((label, done, total)) = &self.download_progress {
                let label = label.clone();
                let done = *done;
                let total = *total;
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(80.0);
                        ui.heading(label);
                        ui.add_space(10.0);
                        let mb_done = done as f64 / (1024.0 * 1024.0);
                        if let Some(tot) = total {
                            let mb_tot = tot as f64 / (1024.0 * 1024.0);
                            let frac = (done as f32 / tot as f32).clamp(0.0, 1.0);
                            ui.add(
                                egui::ProgressBar::new(frac)
                                    .desired_width(360.0)
                                    .text(format!("{mb_done:.1} MB / {mb_tot:.1} MB")),
                            );
                        } else {
                            ui.spinner();
                            ui.label(format!("{mb_done:.1} MB descargados..."));
                        }
                        ui.add_space(12.0);
                        if ui.button("Cancelar").clicked() {
                            self.download_cancel.store(true, Ordering::Relaxed);
                        }
                    });
                });
                return;
            }

            if let Some(busy) = &self.busy_label {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(100.0);
                        ui.spinner();
                        ui.add_space(10.0);
                        ui.heading(busy);
                    });
                });
                return;
            }

            if !self.model_exists {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(60.0);
                        ui.heading("Bienvenido a BigDataBGRemover");
                        ui.add_space(8.0);
                        ui.label("Para funcionar 100% sin internet, necesitamos descargar el modelo de IA BiRefNet una sola vez.");
                        ui.label(format!("Se guardará en: {}", self.model_path.display()));
                        ui.add_space(16.0);
                        if ui.button("⬇ Descargar modelo BiRefNet").clicked() {
                            self.start_download(ctx);
                        }
                        ui.add_space(8.0);
                        if ui.button("📂 Ya tengo un archivo .onnx local").clicked()
                            && let Some(path) = rfd::FileDialog::new()
                                .add_filter("Modelo ONNX", &["onnx"])
                                .pick_file()
                        {
                            self.model_path = path;
                            self.model_exists = true;
                            self.status_msg =
                                Some(("Modelo local cargado.".to_string(), false));
                        }
                    });
                });
                return;
            }

            let Some(tex) = &self.preview_tex else {
                let is_hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(70.0);
                        let card_stroke = if is_hovering {
                            Stroke::new(2.0_f32, Color32::from_rgb(129, 140, 248))
                        } else {
                            Stroke::new(1.5_f32, Color32::from_rgb(55, 62, 82))
                        };
                        let card_fill = if is_hovering {
                            Color32::from_rgb(30, 35, 52)
                        } else {
                            Color32::from_rgb(24, 27, 36)
                        };
                        egui::Frame::new()
                            .fill(card_fill)
                            .stroke(card_stroke)
                            .corner_radius(14.0)
                            .inner_margin(egui::Margin::symmetric(48, 40))
                            .show(ui, |ui| {
                                ui.set_max_width(480.0);
                                ui.vertical_centered(|ui| {
                                    ui.label(
                                        egui::RichText::new("🖼")
                                            .size(42.0),
                                    );
                                    ui.add_space(8.0);
                                    ui.heading(
                                        egui::RichText::new(if is_hovering {
                                            "¡Suelta la imagen aquí!"
                                        } else {
                                            "Arrastra una imagen aquí"
                                        })
                                        .size(22.0)
                                        .color(Color32::WHITE),
                                    );
                                    ui.add_space(6.0);
                                    ui.label(
                                        egui::RichText::new("Soporta PNG, JPG y WebP • Suelta varias para procesar por lote")
                                            .color(Color32::from_rgb(155, 163, 184)),
                                    );
                                    ui.add_space(20.0);
                                    ui.horizontal(|ui| {
                                        let primary = egui::Button::new(
                                            egui::RichText::new("📂 Abrir imagen").color(Color32::WHITE),
                                        )
                                        .fill(Color32::from_rgb(99, 102, 241))
                                        .corner_radius(8.0)
                                        .min_size(egui::vec2(145.0, 36.0));
                                        if ui.add(primary).clicked()
                                            && let Some(path) = rfd::FileDialog::new()
                                                .add_filter("Imágenes", &["png", "jpg", "jpeg", "webp"])
                                                .pick_file()
                                        {
                                            self.process_single_image(path, ctx);
                                        }

                                        ui.add_space(8.0);

                                        let secondary = egui::Button::new("📋 Pegar del portapapeles")
                                            .corner_radius(8.0)
                                            .min_size(egui::vec2(185.0, 36.0));
                                        if ui.add(secondary).clicked() {
                                            self.paste_from_clipboard(ctx);
                                        }
                                    });
                                });
                            });
                    });
                });
                return;
            };

            let avail = ui.available_size();
            let img_size = tex.size_vec2();
            let scale = (avail.x / img_size.x).min(avail.y / img_size.y).min(1.0);
            let draw_size = img_size * scale;

            let (rect, response) = ui.allocate_exact_size(avail, Sense::click_and_drag());
            let img_rect = Rect::from_center_size(rect.center(), draw_size);

            ui.painter().image(
                tex.id(),
                img_rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );

            // Handle 2nd-pass AI click (MobileSAM)
            if matches!(self.tool, ToolMode::AiKeep | ToolMode::AiErase) && !self.show_original {
                if response.clicked()
                    && let Some(pos) = response.interact_pointer_pos()
                    && img_rect.contains(pos)
                {
                    if !self.sam_exists {
                        self.start_sam_download(ctx);
                    } else {
                        let ix = (pos.x - img_rect.min.x) / scale;
                        let iy = (pos.y - img_rect.min.y) / scale;
                        let keep = self.tool == ToolMode::AiKeep;
                        self.run_sam_click(ix, iy, keep, ctx);
                    }
                }
                return;
            }

            // Handle manual brush painting
            if matches!(self.tool, ToolMode::BrushKeep | ToolMode::BrushErase)
                && !self.show_original
            {
                let keep = self.tool == ToolMode::BrushKeep;
                let brush_screen_r = self.brush_radius;
                let brush_img_r = (self.brush_radius / scale).max(1.0);

                if let Some(hover_pos) = response.hover_pos()
                    && img_rect.contains(hover_pos)
                {
                    let color = if keep {
                        Color32::from_rgb(80, 230, 110)
                    } else {
                        Color32::from_rgb(240, 80, 80)
                    };
                    ui.painter()
                        .circle_stroke(hover_pos, brush_screen_r, Stroke::new(1.5_f32, color));
                }

                if response.drag_started()
                    || (response.is_pointer_button_down_on() && !self.is_painting)
                {
                    if let Some(ed) = &mut self.editor {
                        ed.push_undo();
                    }
                    self.is_painting = true;
                    self.last_paint_pos = None;
                }

                if self.is_painting
                    && response.is_pointer_button_down_on()
                    && let Some(pos) = response.interact_pointer_pos()
                {
                    let ix = (pos.x - img_rect.min.x) / scale;
                    let iy = (pos.y - img_rect.min.y) / scale;

                    if let Some(ed) = &mut self.editor {
                        if let Some((lx, ly)) = self.last_paint_pos {
                            let dist = ((ix - lx).powi(2) + (iy - ly).powi(2)).sqrt();
                            let step = (brush_img_r * 0.25).max(1.0);
                            let steps = (dist / step).ceil().max(1.0) as usize;
                            for s in 1..=steps {
                                let t = s as f32 / steps as f32;
                                ed.paint_circle(
                                    lx + (ix - lx) * t,
                                    ly + (iy - ly) * t,
                                    brush_img_r,
                                    keep,
                                );
                            }
                        } else {
                            ed.paint_circle(ix, iy, brush_img_r, keep);
                        }
                        self.last_paint_pos = Some((ix, iy));
                    }
                    self.refresh_texture(ctx);
                }

                if response.drag_stopped() || !response.is_pointer_button_down_on() {
                    self.is_painting = false;
                    self.last_paint_pos = None;
                }
            }
        });
    }
}

fn apply_modern_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = Color32::from_rgb(18, 20, 26);
    visuals.panel_fill = Color32::from_rgb(22, 25, 33);
    visuals.faint_bg_color = Color32::from_rgb(28, 32, 42);
    visuals.extreme_bg_color = Color32::from_rgb(14, 16, 21);

    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(42, 47, 61));
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(34, 38, 50);
    visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(34, 38, 50);
    visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(7);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(48, 54, 72);
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(48, 54, 72);
    visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(7);
    visuals.widgets.active.bg_fill = Color32::from_rgb(99, 102, 241);
    visuals.widgets.active.corner_radius = egui::CornerRadius::same(7);
    visuals.selection.bg_fill = Color32::from_rgb(99, 102, 241);
    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 5.0);
    ctx.set_style(style);
}

fn log_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("bigdata-bg-remover")
        .join("app.log")
}

fn write_log(line: &str) {
    let path = log_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{line}");
    }
}

fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let bt = std::backtrace::Backtrace::force_capture();
        let msg = format!("[PANIC] {info}\nBacktrace:\n{bt}");
        eprintln!("{msg}");
        write_log(&msg);
        default_hook(info);
    }));
}

fn main() -> eframe::Result<()> {
    install_panic_hook();
    write_log("[INFO] Iniciando BigDataBGRemover...");

    // `winit` 0.30 only implements file drag-and-drop on X11, not Wayland.
    // Removing `WAYLAND_DISPLAY` when `DISPLAY` (XWayland) is present forces X11 backend.
    if std::env::var_os("WAYLAND_DISPLAY").is_some() && std::env::var_os("DISPLAY").is_some() {
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("No se pudo iniciar el runtime de Tokio");
    let _rt_guard = rt.enter();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 720.0])
            .with_min_inner_size([700.0, 500.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    let res = eframe::run_native(
        "BigDataBGRemover",
        options,
        Box::new(|cc| Ok(Box::new(BigDataBgApp::new(cc)))),
    );
    if let Err(ref e) = res {
        write_log(&format!("[ERROR] eframe::run_native falló: {e}"));
    }
    res
}
