# Prelabeling Pipeline — Automated Dataset Preparation Tool (ES/EN)

## ES — Descripción
**Prelabeling Pipeline** es una herramienta para preparar datasets de visión computacional de forma semi-automatizada.

El flujo incluye:
- Renombrado automático de imágenes (formato fecha/hora)
- Pre-etiquetado usando un modelo RT-DETR
- Generación de etiquetas en formato YOLO
- Verificación de pares imagen + label
- Agrupación en carpetas de entrega (100 imágenes)
- Compresión automática en archivos .zip

Está pensado para trabajar con imágenes provenientes de cámaras (por ejemplo: entornos controlados o cámaras cercanas).

---

## EN — Description
**Prelabeling Pipeline** is a semi-automated tool for preparing computer vision datasets.

It includes:
- Automatic image renaming (timestamp normalization)
- Pre-labeling using an RT-DETR model
- YOLO format label generation
- Image-label pair validation
- Dataset grouping into delivery folders (100 images each)
- Automatic compression into .zip files

Designed for datasets captured from camera systems (e.g., close-range environments).

---

## 📁 Estructura esperada / Expected Structure

## DATASET_ROOT = r"C:\path\to\your_dataset"
## WEIGHTS_PATH = r"C:\path\to\your_model.pt"