use eframe::egui::ColorImage;
use image::{GrayImage, Luma, RgbaImage};

const MAX_UNDO: usize = 20;

pub struct ImageEditor {
    pub original: RgbaImage,
    pub mask: GrayImage,
    undo_stack: Vec<GrayImage>,
}

impl ImageEditor {
    pub fn new(original: RgbaImage, mask: GrayImage) -> Self {
        Self {
            original,
            mask,
            undo_stack: Vec::new(),
        }
    }

    pub fn width(&self) -> u32 {
        self.original.width()
    }

    pub fn height(&self) -> u32 {
        self.original.height()
    }

    pub fn push_undo(&mut self) {
        if self.undo_stack.len() >= MAX_UNDO {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(self.mask.clone());
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn undo(&mut self) -> bool {
        if let Some(prev) = self.undo_stack.pop() {
            self.mask = prev;
            true
        } else {
            false
        }
    }

    /// Paints a soft circular stroke onto the alpha mask at image coordinates `(cx, cy)`.
    /// `keep = true` restores the original image (alpha -> 255); `keep = false` erases (alpha -> 0).
    pub fn paint_circle(&mut self, cx: f32, cy: f32, radius: f32, keep: bool) {
        let w = self.mask.width() as i32;
        let h = self.mask.height() as i32;
        let r_ceil = radius.ceil() as i32;

        let x0 = (cx.floor() as i32 - r_ceil).max(0);
        let x1 = (cx.ceil() as i32 + r_ceil).min(w - 1);
        let y0 = (cy.floor() as i32 - r_ceil).max(0);
        let y1 = (cy.ceil() as i32 + r_ceil).min(h - 1);

        let target = if keep { 255.0_f32 } else { 0.0_f32 };
        let r_sq = radius * radius;

        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cy;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq <= r_sq {
                    let dist = dist_sq.sqrt();
                    // Inner 65% of brush is solid, outer 35% feathers smoothly
                    let t = if dist <= radius * 0.65 {
                        1.0
                    } else {
                        ((radius - dist) / (radius * 0.35)).clamp(0.0, 1.0)
                    };
                    let cur = self.mask.get_pixel(x as u32, y as u32)[0] as f32;
                    let next = if keep {
                        cur.max(cur + (target - cur) * t)
                    } else {
                        cur.min(cur + (target - cur) * t)
                    };
                    self.mask
                        .put_pixel(x as u32, y as u32, Luma([next.round() as u8]));
                }
            }
        }
    }

    /// Merges a MobileSAM object mask into the current alpha mask (`keep = true` adds, `keep = false` subtracts).
    pub fn merge_sam_mask(&mut self, sam_mask: &GrayImage, keep: bool) {
        for (dst, src) in self.mask.pixels_mut().zip(sam_mask.pixels()) {
            if keep {
                dst[0] = dst[0].max(src[0]);
            } else {
                dst[0] = dst[0].min(255 - src[0]);
            }
        }
    }

    /// Exports the final transparent RGBA image at full original resolution.
    pub fn export_rgba(&self) -> RgbaImage {
        let mut out = self.original.clone();
        for (px, m) in out.pixels_mut().zip(self.mask.pixels()) {
            // Preserve original semi-transparency if any, scaled by mask
            px[3] = ((px[3] as u16 * m[0] as u16) / 255) as u8;
        }
        out
    }

    /// Builds an `egui::ColorImage` for preview (either original or cutout over checkerboard).
    pub fn build_preview(&self, show_original: bool) -> ColorImage {
        let w = self.width() as usize;
        let h = self.height() as usize;

        if show_original {
            let rgba = self.original.as_raw();
            return ColorImage::from_rgba_unmultiplied([w, h], rgba);
        }

        let mut buf = Vec::with_capacity(w * h * 4);
        let checker_size = (w.max(h) / 48).clamp(8, 24);

        for y in 0..h {
            for x in 0..w {
                let px = self.original.get_pixel(x as u32, y as u32);
                let mask_a = self.mask.get_pixel(x as u32, y as u32)[0];
                let alpha = ((px[3] as u16 * mask_a as u16) / 255) as f32 / 255.0;

                let is_light = ((x / checker_size) + (y / checker_size)) % 2 == 0;
                let bg = if is_light { 215.0 } else { 165.0 };

                let r = (px[0] as f32 * alpha + bg * (1.0 - alpha)).round() as u8;
                let g = (px[1] as f32 * alpha + bg * (1.0 - alpha)).round() as u8;
                let b = (px[2] as f32 * alpha + bg * (1.0 - alpha)).round() as u8;

                buf.push(r);
                buf.push(g);
                buf.push(b);
                buf.push(255);
            }
        }

        ColorImage::from_rgba_unmultiplied([w, h], &buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn brush_keep_erase_and_undo() {
        let orig = RgbaImage::from_pixel(20, 20, Rgba([100, 150, 200, 255]));
        let mask = GrayImage::from_pixel(20, 20, Luma([0]));
        let mut ed = ImageEditor::new(orig, mask);

        ed.push_undo();
        ed.paint_circle(10.0, 10.0, 4.0, true);
        assert_eq!(ed.mask.get_pixel(10, 10)[0], 255);
        assert_eq!(ed.export_rgba().get_pixel(10, 10)[3], 255);
        assert_eq!(ed.export_rgba().get_pixel(0, 0)[3], 0);

        ed.push_undo();
        ed.paint_circle(10.0, 10.0, 2.0, false);
        assert_eq!(ed.mask.get_pixel(10, 10)[0], 0);

        assert!(ed.undo());
        assert_eq!(ed.mask.get_pixel(10, 10)[0], 255);
    }
}
