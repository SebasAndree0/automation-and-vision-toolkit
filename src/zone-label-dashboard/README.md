# Zone Label Dashboard — Zone-based YOLO label analytics (ES/EN)

## ES — Descripción
**Zone Label Dashboard** es una app (Tkinter) para analizar datasets con etiquetas YOLO y zonas definidas por YAML.

Permite:
- Cargar zonas desde YAML (rectángulos o polígonos)
- Cargar clases desde `classes.txt`
- Recorrer labels YOLO y contar objetos por:
  - Zona + clase
  - Total por zona
  - Total global por clase
- Generar dos imágenes:
  1) `*_zones.png`: imagen base con zonas dibujadas y numeradas
  2) `*_dashboard.png`: dashboard con conteos por zona y resumen global

Los checkboxes permiten “marcar” zonas para dibujarlas en un color distinto.

---

## EN — Description
**Zone Label Dashboard** is a Tkinter app that analyzes YOLO-labeled datasets using zones defined in YAML.

It can:
- Load zones from YAML (rectangles or polygons)
- Load class names from `classes.txt`
- Parse YOLO label files and count detections by:
  - Zone + class
  - Zone totals
  - Global totals per class
- Generate two images:
  1) `*_zones.png`: base image with zones drawn and numbered
  2) `*_dashboard.png`: dashboard with per-zone counts and a global summary

Checkboxes let you highlight zones using a different color.

---

## ✅ Input formats

### YOLO labels
Each label line: