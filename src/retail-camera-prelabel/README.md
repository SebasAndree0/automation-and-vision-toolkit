# Retail Camera Prelabel — Dataset pipeline for camera images (ES/EN)

## ES — Descripción
**Retail Camera Prelabel** es un pipeline semi-automático para preparar datasets desde cámaras (entornos controlados / cámaras cercanas).

Incluye:
- Renombrado de imágenes (normaliza timestamps: `YYYY-MM-DD-HH...` → `YYYY-MM-DD_HH...`)
- (Opcional) Aplicación de máscara/crop mediante un archivo YAML por cámara
- Pre-etiquetado usando RT-DETR (Ultralytics)
- Generación de labels en formato YOLO (`.txt`)
- Verificación de pares imagen + etiqueta
- Agrupación en entregas de 100 pares por carpeta
- Compresión automática a `.zip`

> ⚠️ Este repositorio NO incluye modelos `.pt` ni YAML (debes aportar los tuyos).

---

## EN — Description
**Retail Camera Prelabel** is a semi-automated pipeline to prepare datasets from camera images (close-range / controlled environments).

It includes:
- Image renaming (timestamp normalization: `YYYY-MM-DD-HH...` → `YYYY-MM-DD_HH...`)
- (Optional) Mask/crop step using a per-camera YAML file
- Pre-labeling using RT-DETR (Ultralytics)
- YOLO label generation (`.txt`)
- Image/label pair validation
- Grouping into 100-pair delivery folders
- Automatic `.zip` compression

> ⚠️ This repository does NOT include `.pt` models or YAML files (you must provide your own).

---

## 📁 Estructura / Structure

Expected dataset layout: