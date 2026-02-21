# Video Crop Annotator — Manual video crop tool for dataset building (ES/EN)

## ES — Descripción
**Video Crop Annotator** es una herramienta para abrir un video, avanzar frame por frame y recortar manualmente regiones (rectángulos) para construir un dataset de imágenes recortadas (*crops*).

Permite organizar automáticamente los recortes por:
- **Local** (carpeta del lugar / cámara)
- **Tipo**: Persona, Trabajador, Mechero
- **Identidad incremental**: Persona1, Persona2, etc.

Los crops se guardan como **PNG** y se ordenan en carpetas de forma automática.

---

## EN — Description
**Video Crop Annotator** is a tool to open a video, move frame-by-frame, and manually crop rectangular regions to build a dataset of image crops.

Crops are automatically organized by:
- **Location** (place/camera folder)
- **Type**: Person, Worker, Thief
- **Incremental identity**: Person1, Person2, etc.

Crops are saved as **PNG** and automatically structured into folders.

---

## ES — Requisitos
- Python 3.9+ (recomendado)
- OpenCV: `opencv-python`
- Tkinter (normalmente viene con Python en Windows; en Linux puede requerir instalación por sistema)

Instalación:
```bash
pip install opencv-python