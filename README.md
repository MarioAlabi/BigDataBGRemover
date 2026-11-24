# BigDataBGRemover

Removedor de fondos para Linux **100% local, gratuito, sin límites de resolución y multi-distro**, escrito en **Rust** puro usando **ONNX Runtime (`ort`)** y **egui (`eframe`)**.

> No sube tus imágenes a ningún servidor: toda la inferencia ocurre en tu equipo.

---

## ✨ Características

- **Recorte automático de alta precisión:** usa el modelo **BiRefNet** (`birefnet-general.onnx`), excelente con cabello, pelaje y bordes complejos.
- **2ª pasada inteligente con IA (MobileSAM):** si faltó un objeto o quedó un resto de fondo, haz **clic** con **"✨➕ Incluir con IA"** o **"✨➖ Quitar con IA"** para sumar o restar regiones completas al instante.
- **Pincel de retoque rápido:** ajusta bordes manualmente con **"➕ Conservar"** / **"➖ Borrar"**, tamaño regulable y **Deshacer** (`Ctrl+Z`).
- **Flujo rápido sin conocimientos de edición:**
  - Arrastrar y soltar (*Drag & Drop*).
  - Pegar directo desde el portapapeles (**`Ctrl+V`** o botón **Pegar**).
  - **Procesamiento por lotes:** suelta o selecciona varias imágenes para exportarlas todas en PNG transparente.
- **Portable y agnóstico a la distro:** se empaqueta en un **AppImage** de ~17 MB compatible con **Gear Lever**.

---

## 🚀 Instalación y uso

### Opción 1: AppImage (recomendado)
1. Descarga `BigDataBGRemover-x86_64.AppImage` (o compílalo con el script incluido).
2. Dale permisos de ejecución o intégralo directamente con **Gear Lever**:
   ```bash
   chmod +x BigDataBGRemover-x86_64.AppImage
   ./BigDataBGRemover-x86_64.AppImage
   ```
3. **Primera ejecución:** la app te pedirá descargar el modelo `birefnet-general.onnx` (con barra de progreso) a `~/.local/share/bigdata-bg-remover/models/`. También puedes elegir un archivo `.onnx` que ya tengas guardado.
4. Si usas la **2ª pasada con IA** (*Incluir/Quitar con IA*), la app descargará automáticamente **MobileSAM (~40 MB)** la primera vez.

### Opción 2: Compilar desde el código fuente
Requisitos: Rust (`cargo`) y herramientas estándar de compilación (`cmake`, `g++`, `pkg-config`).

```bash
git clone https://github.com/MarioAlabi/BigDataBGRemover.git
cd BigDataBGRemover
cargo build --release
./target/release/bigdata-bg-remover
```

### Crear el AppImage
```bash
./packaging/build_appimage.sh
```
*(Requiere `appimagetool-x86_64.AppImage` en `/tmp/appimagetool-x86_64.AppImage` o definido en la variable `APPIMAGETOOL`).*

---

## 📂 Rutas importantes
- **Modelos descargados:** `~/.local/share/bigdata-bg-remover/models/`
- **Logs de errores:** `~/.local/share/bigdata-bg-remover/app.log`

---

## 📸 Capturas de pantalla

*(Agrega aquí tus capturas en `docs/screenshots/` para mostrar la ventana principal y un ejemplo de recorte).*
