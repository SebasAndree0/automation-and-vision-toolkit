README (Español / English)

# Masked Video Detection (RT-DETR) — Detección de Video con Máscara (RT-DETR)

## Qué es / What is it
Procesa videos con rotación+crop+máscara ROI desde YAML y detecta objetos con RT-DETR. — Processes videos with rotation+crop+ROI mask from YAML and runs RT-DETR object detection.

## Requisitos / Requirements
Python 3.9+ — Python 3.9+  
opencv-python, pyyaml, numpy, torch, ultralytics — opencv-python, pyyaml, numpy, torch, ultralytics  
(Opt) ffmpeg para reparar MP4 — (Opt) ffmpeg to fix MP4 files

## Instalación / Install
```bash
pip install opencv-python pyyaml numpy torch ultralytics