# Image Rect Mask Batch — Keep only a rectangle visible (ES/EN)

## ES — Descripción
Este script aplica una **máscara rectangular** a todas las imágenes dentro de una carpeta:
- Todo queda en negro
- Solo el área dentro del rectángulo (x1, y1, x2, y2) queda visible

Por seguridad, por defecto **NO sobrescribe** las imágenes originales: crea una carpeta de salida.
Si quieres sobrescribir, existe la opción `--in-place` (peligrosa).

---

## EN — Description
This script applies a **rectangular mask** to every image in a folder:
- Everything becomes black
- Only the rectangle area (x1, y1, x2, y2) stays visible

For safety, by default it **does NOT overwrite** originals: it writes to an output folder.
If you really want to overwrite, use `--in-place` (dangerous).

---

## 📦 Requirements
Create `requirements.txt`: