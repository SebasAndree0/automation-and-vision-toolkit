# Camera Prelabel Pipeline — Pre-etiquetado por cámara con máscara YAML + empaquetado (ES/EN)

## ES — Descripción
**Camera Prelabel Pipeline** es una herramienta semi-automática para preparar datasets desde **cámaras** (por ejemplo, cámaras cercanas o entornos controlados).

Incluye:
- Renombrado de imágenes (normaliza timestamp `YYYY-MM-DD-HH...` → `YYYY-MM-DD_HH...`)
- (Opcional) Aplicación de **máscara/crop** usando un archivo **YAML** por cámara
- Pre-etiquetado usando un modelo **RT-DETR** (Ultralytics)
- Generación de etiquetas en formato **YOLO** (`.txt`)
- Verificación de pares **imagen + label**
- Agrupación en carpetas de entrega (100 pares por carpeta)
- Compresión automática a `.zip`

> ⚠️ El modelo `.pt` y los YAML **no se incluyen** en este repositorio. Debes usar tus propios archivos.

---

## EN — Description
**Camera Prelabel Pipeline** is a semi-automated tool to prepare datasets from **camera feeds** (e.g., close-range or controlled environments).

It includes:
- Image renaming (timestamp normalization `YYYY-MM-DD-HH...` → `YYYY-MM-DD_HH...`)
- (Optional) Mask/crop step using a per-camera **YAML** file
- Pre-labeling using **RT-DETR** (Ultralytics)
- YOLO label generation (`.txt`)
- Image/label pair validation
- Grouping into delivery folders (100 pairs each)
- Automatic `.zip` compression

> ⚠️ The `.pt` model and YAML files are **not included** in this repository. You must provide your own.

---

## 📁 Estructura esperada / Expected Structure
