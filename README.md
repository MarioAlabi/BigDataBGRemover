# BigDataBGRemover

Removedor de fondos de escritorio 100% local, gratuito y multi-distro para Linux, escrito en **Rust** con **ONNX Runtime (`ort`)**.

## Características
- **100% offline y privado:** ejecuta **BiRefNet** (`birefnet-general.onnx`) localmente sin subir imágenes a la nube.
- **Segunda pasada inteligente con IA (MobileSAM):** haz clic en *"Incluir con IA"* o *"Quitar con IA"* para sumar o restar objetos completos.
- **Pincel de retoque rápido:** ajusta bordes manualmente con *Conservar / Borrar* y *Deshacer* (`Ctrl+Z`).
- **Arrastrar y soltar, pegar (`Ctrl+V`) y modo por lotes.**

## Compilación
```bash
cargo build --release
```
