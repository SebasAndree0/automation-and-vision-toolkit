# Bag-Person Cropper — Crop people that have bags (YOLO) (ES/EN)

## ES — Descripción
Este script genera **crops de personas que tienen bolsas**, usando:
- **Bolsas** desde labels YOLO existentes (por defecto `class_id = 0`)
- **Personas** detectadas por un modelo (RT-DETR o YOLO vía `ultralytics`)

Flujo:
1) Lee cada imagen en `images/`
2) Lee su label YOLO correspondiente en `labels/` (bolsas)
3) Detecta personas con el modelo
4) Asigna bolsas a personas (por centro dentro de bbox o por IoU)
5) Recorta 1 vez por persona (unión persona ∪ bolsas) + padding asimétrico
6) Guarda:
   - `crops_with_bags/images/*.jpg`
   - `crops_with_bags/labels/*.txt` (solo bolsas dentro del crop; clase 0 = bag)
7) Genera un CSV resumen

---

## EN — Description
This script generates **person crops that contain bags**, using:
- **Bags** from existing YOLO labels (default `class_id = 0`)
- **People** detected by a model (RT-DETR or YOLO via `ultralytics`)

Pipeline:
1) Reads each image from `images/`
2) Reads its YOLO label from `labels/` (bags)
3) Detects people with the model
4) Assigns bags to people (bbox-center or IoU)
5) Crops once per person (union person ∪ bags) + asymmetric padding
6) Saves:
   - `crops_with_bags/images/*.jpg`
   - `crops_with_bags/labels/*.txt` (bags-only inside crop; output class 0 = bag)
7) Writes a summary CSV

---

## 📦 Requirements
Create `requirements.txt`: