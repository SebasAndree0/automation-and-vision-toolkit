# Background Object Selector — Remove background & toggle detected objects (ES/EN)

## ES — Descripción
**Background Object Selector** es una app de escritorio (Tkinter) para:
1) Cargar una imagen (`.jpg/.png`)
2) Remover el fondo (usa `rembg`)
3) Detectar “objetos” principales usando componentes conectados (alpha mask)
4) Mostrar miniaturas de cada objeto detectado
5) Activar/desactivar objetos con doble clic
6) Exportar el resultado como PNG con transparencia

> Este repositorio no incluye modelos personalizados; `rembg` se encarga de la remoción de fondo usando un modelo estándar (por defecto `u2net`).

---

## EN — Description
**Background Object Selector** is a desktop app (Tkinter) that:
1) Loads an image (`.jpg/.png`)
2) Removes the background (via `rembg`)
3) Detects main objects using connected components on the alpha mask
4) Shows a thumbnail list for each detected object
5) Toggles objects on/off with double click
6) Exports the result as a transparent PNG

> This repository does not include custom models; `rembg` handles background removal using a standard model (default: `u2net`).

---

## 📦 Requirements

Create `requirements.txt`: