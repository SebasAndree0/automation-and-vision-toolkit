# -*- coding: utf-8 -*-
"""
Masked Video Detection (RT-DETR)
- Applies rotation + crop + polygon mask (ROI) from YAML
- Crops output to the mask bounding box (computed once) for constant output size
- Runs Ultralytics RT-DETR detection
- Writes annotated MP4
- Optionally repairs MP4 (moov atom) via ffmpeg faststart
"""

import os
import json
import cv2
import yaml
import shutil
import argparse
import numpy as np
from datetime import datetime
from ultralytics import RTDETR

DEFAULT_CONF = 0.6
PALETA_BGR = [
    (255, 0, 0), (0, 255, 0), (0, 140, 255), (255, 255, 0), (0, 165, 255),
    (192, 192, 192), (255, 0, 255), (0, 255, 255), (255, 0, 127), (127, 0, 255),
    (0, 127, 255), (255, 127, 0)
]


# =========================
# UTILIDADES
# =========================
def limpiar_ruta(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if s.startswith(r'\"') and s.endswith(r'\"'):
        s = s[2:-2]
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s


def asegurar_carpeta(path_dir: str):
    if path_dir and not os.path.exists(path_dir):
        os.makedirs(path_dir, exist_ok=True)


def leer_yaml(path_yaml: str) -> dict:
    with open(path_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preparar_estilo(clases, thresholds_map=None, colors_map=None, default_conf=DEFAULT_CONF):
    clases = list(clases or [])
    if not clases:
        raise RuntimeError("No classes provided. Provide --classes or a profile with classes.")
    thresholds_map = dict(thresholds_map or {})
    colors_map = dict(colors_map or {})

    thresholds = {c: float(thresholds_map.get(c, default_conf)) for c in clases}
    colors = {}
    for i, c in enumerate(clases):
        colors[c] = colors_map.get(c, PALETA_BGR[i % len(PALETA_BGR)])
    return clases, thresholds, colors


# =========================
# ROTACIÓN, CROP, MÁSCARA y RECORTE A LA MÁSCARA
# =========================
def aplicar_rotacion_crop_mascara(frame, config, roi_bbox=None, compute_roi=False):
    """
    - compute_roi=True: calcula el bounding box de la máscara (pixeles != 0) en el primer frame.
    - roi_bbox=(x,y,w,h): si llega, recorta a ese bbox después de aplicar la máscara.
    Devuelve: (img_recortada, roi_bbox_usado_o_calculado)
    """
    angle = float(config["FishEye"]["Angle"])
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h))

    left = int(config["FishEye"]["CropSize"]["Left"])
    right = w - int(config["FishEye"]["CropSize"]["Right"])
    top = int(config["FishEye"]["CropSize"]["Top"])
    bottom = h - int(config["FishEye"]["CropSize"]["Bottom"])
    cropped = rotated[top:bottom, left:right]
    ch, cw = cropped.shape[:2]

    mask = np.zeros((ch, cw), dtype=np.uint8)
    recinto = np.array(config["Track"]["Recinto"], dtype=np.int32)

    # Si el polígono está en coords del frame rotado completo, llevar al recorte:
    recinto_local = recinto.copy()
    if recinto_local.max() > max(cw, ch) or recinto_local.min() < 0:
        recinto_local[:, 0] -= left
        recinto_local[:, 1] -= top

    recinto_local[:, 0] = np.clip(recinto_local[:, 0], 0, cw - 1)
    recinto_local[:, 1] = np.clip(recinto_local[:, 1], 0, ch - 1)
    cv2.fillPoly(mask, [recinto_local], 255)

    img_masked = cv2.bitwise_and(cropped, cropped, mask=mask)

    if compute_roi:
        nz = cv2.findNonZero(mask)
        if nz is not None:
            x, y, wbb, hbb = cv2.boundingRect(nz)
            roi_bbox = (int(x), int(y), int(wbb), int(hbb))
        else:
            roi_bbox = None

    if roi_bbox is not None:
        x, y, wbb, hbb = roi_bbox
        x = max(0, x); y = max(0, y)
        x2 = min(x + wbb, cw)
        y2 = min(y + hbb, ch)
        if x < x2 and y < y2:
            img_masked = img_masked[y:y2, x:x2]
            roi_bbox = (x, y, x2 - x, y2 - y)

    return img_masked, roi_bbox


# =========================
# DIBUJO DE DETECCIONES
# =========================
def dibujar_detecciones(frame, preds, clases, thresholds, colors):
    h, w = frame.shape[:2]
    grosor = max(2, int(0.002 * (h + w)))

    for det in preds:
        if len(det) < 6:
            continue
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)
        if cls_id < 0 or cls_id >= len(clases):
            continue

        nombre = clases[cls_id]
        if float(conf) < float(thresholds.get(nombre, DEFAULT_CONF)):
            continue

        color = colors.get(nombre, (0, 255, 0))
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor)
        label = f"{nombre} {float(conf):.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th_text), _ = cv2.getTextSize(label, font, font_scale, thickness)
        y_text_top = max(y1 - th_text - 6, 0)
        cv2.rectangle(frame, (x1, y_text_top), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame


# =========================
# REPARACIÓN MP4 (moov atom)
# =========================
def ffmpeg_bin(ffmpeg_path: str):
    ffmpeg_path = limpiar_ruta(ffmpeg_path or "")
    if ffmpeg_path and os.path.exists(ffmpeg_path):
        return ffmpeg_path
    which = shutil.which("ffmpeg")
    return which if which else None


def reparar_mp4_faststart(input_path: str, ffmpeg_path: str) -> str:
    ffmpeg = ffmpeg_bin(ffmpeg_path)
    if not ffmpeg:
        print("⚠️  ffmpeg no encontrado. No se puede reparar el MP4.")
        return ""

    out_path = os.path.splitext(input_path)[0] + "_faststart.mp4"
    cmd = f'"{ffmpeg}" -y -v error -i "{input_path}" -c copy -movflags +faststart "{out_path}"'
    print(f"🛠  Reparando contenedor con ffmpeg: {out_path}")
    rc = os.system(cmd)

    if rc == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    print("⚠️  Reparación falló o produjo archivo vacío.")
    return ""


# =========================
# SALIDA
# =========================
def construir_salida(video_path: str, out_dir: str, tag: str, fps: float, size):
    base = os.path.splitext(os.path.basename(video_path))[0]
    slug = tag or "masked"
    out_name = f"{base}_detections_{datetime.now().strftime('%Y-%m-%d')}_{slug}.mp4"

    out_dir = out_dir or os.path.dirname(video_path) or "."
    asegurar_carpeta(out_dir)

    out_path = os.path.join(out_dir, out_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, size)
    print(f"💾 Output → {out_path} ({size[0]}x{size[1]}, {fps:.1f} fps)")
    return out_path, out


# =========================
# PROCESADO
# =========================
def procesar_video(video_path: str, yaml_path: str, model_path: str, out_dir: str,
                  clases, thresholds_map, colors_map, tag: str,
                  try_repair: bool, ffmpeg_path: str):

    video_path = limpiar_ruta(video_path)
    yaml_path = limpiar_ruta(yaml_path)
    model_path = limpiar_ruta(model_path)
    out_dir = limpiar_ruta(out_dir or "")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    config = leer_yaml(yaml_path)
    clases, thresholds, colors = preparar_estilo(clases, thresholds_map, colors_map)

    print("🔄 Loading RT-DETR model...")
    model = RTDETR(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️  Could not open video: {video_path}")
        if try_repair:
            repaired = reparar_mp4_faststart(video_path, ffmpeg_path)
            if repaired:
                print(f"🔁 Retrying repaired file: {repaired}")
                cap = cv2.VideoCapture(repaired)
                if not cap.isOpened():
                    raise RuntimeError(f"Still cannot open repaired file: {repaired}")
                video_path = repaired
            else:
                raise RuntimeError("Could not open video and repair failed.")
        else:
            raise RuntimeError("Could not open video (repair disabled).")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame.")

    frame_proc, roi_bbox = aplicar_rotacion_crop_mascara(frame, config, compute_roi=True)
    height, width = frame_proc.shape[:2]

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 25.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = total if total > 0 else None

    out_path, out = construir_salida(video_path, out_dir, tag, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"🎯 Active classes ({len(clases)}): {clases}")
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed, _ = aplicar_rotacion_crop_mascara(frame, config, roi_bbox=roi_bbox, compute_roi=False)

            try:
                results = model.predict(processed, verbose=False)
                preds = results[0].boxes.data.detach().cpu().numpy() if len(results) else np.empty((0, 6))
            except Exception as e:
                print(f"⚠️  Inference error at frame {frame_idx}: {e}")
                preds = np.empty((0, 6))

            annotated = dibujar_detecciones(processed, preds, clases, thresholds, colors)
            out.write(annotated)

            frame_idx += 1
            if frame_idx % 50 == 0:
                if total:
                    print(f"Progress: {frame_idx}/{total} frames ({100.0 * frame_idx / total:.1f}%)")
                else:
                    print(f"Progress: {frame_idx} frames")
    finally:
        cap.release()
        out.release()

    print(f"✅ Done. Video generated: {out_path}")


# =========================
# PROFILES (optional)
# =========================
def load_profiles(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "profiles" not in data:
        raise ValueError("profiles.json must be an object with a 'profiles' array.")
    profiles = data["profiles"]
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("'profiles' must be a non-empty array.")
    return profiles


def pick_profile_menu(profiles):
    print("=========== Masked Video Detection (RT-DETR) ===========")
    print("Profiles:")
    for i, p in enumerate(profiles, start=1):
        name = p.get("name", f"profile_{i}")
        print(f" {i}) {name}")
    print(" e) Exit")

    while True:
        sel = input("Choose profile: ").strip().lower()
        if sel in ("e", "q", "exit"):
            return None
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= len(profiles):
                return profiles[idx - 1]
        print("❌ Invalid selection.")


def parse_kv_json(s: str):
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        raise ValueError("Invalid JSON. Provide a valid JSON object string.")


def main():
    ap = argparse.ArgumentParser(
        description="Apply ROI mask from YAML and run RT-DETR detection, exporting an annotated MP4."
    )

    ap.add_argument("--video", help="Input video path")
    ap.add_argument("--yaml", dest="yaml_path", help="YAML config path (FishEye + Track.Recinto)")
    ap.add_argument("--model", dest="model_path", help="RT-DETR weights (.pt)")

    ap.add_argument("--out-dir", default="", help="Output directory (default: next to input video)")

    ap.add_argument("--classes", default="", help="Comma-separated class names (order must match model)")
    ap.add_argument("--thresholds", default="", help='JSON dict: {"class":0.8,...}')
    ap.add_argument("--colors", default="", help='JSON dict: {"class":[B,G,R],...}')

    ap.add_argument("--tag", default="masked", help="Tag used in output filename")
    ap.add_argument("--try-repair", action="store_true", help="Try to repair MP4 using ffmpeg faststart if it fails to open")
    ap.add_argument("--ffmpeg", default="", help="ffmpeg path (optional, uses PATH if empty)")

    ap.add_argument("--profiles", default="", help="Optional profiles.json to select from a menu")

    args = ap.parse_args()

    # If profiles.json provided -> menu selection fills missing values
    if args.profiles:
        profiles = load_profiles(args.profiles)
        p = pick_profile_menu(profiles)
        if not p:
            print("👋 Bye.")
            return

        # Fill from profile if missing
        video = args.video or p.get("video", "")
        yaml_path = args.yaml_path or p.get("yaml", "")
        model_path = args.model_path or p.get("model", "")
        out_dir = args.out_dir or p.get("out_dir", "")

        classes = p.get("classes", [])
        thresholds_map = p.get("thresholds", {})
        colors_map = p.get("colors", {})
        tag = args.tag or p.get("tag", "masked")
        try_repair = args.try_repair or bool(p.get("try_repair", False))
        ffmpeg_path = args.ffmpeg or p.get("ffmpeg", "")

        if not classes:
            raise SystemExit("Profile has no 'classes'. Add classes in profiles.json.")
    else:
        # CLI mode
        if not args.video or not args.yaml_path or not args.model_path:
            raise SystemExit("Missing args. Provide --video --yaml --model (or use --profiles).")

        video = args.video
        yaml_path = args.yaml_path
        model_path = args.model_path
        out_dir = args.out_dir

        classes = [c.strip() for c in (args.classes or "").split(",") if c.strip()]
        thresholds_map = parse_kv_json(args.thresholds)
        colors_map = parse_kv_json(args.colors)
        tag = args.tag
        try_repair = args.try_repair
        ffmpeg_path = args.ffmpeg

        if not classes:
            raise SystemExit("Missing --classes. Provide comma-separated classes in model order (or use --profiles).")

        # Convert colors from [B,G,R] lists to tuples
        for k, v in list(colors_map.items()):
            if isinstance(v, list) and len(v) == 3:
                colors_map[k] = (int(v[0]), int(v[1]), int(v[2]))

    procesar_video(
        video_path=video,
        yaml_path=yaml_path,
        model_path=model_path,
        out_dir=out_dir,
        clases=classes,
        thresholds_map=thresholds_map,
        colors_map=colors_map,
        tag=tag,
        try_repair=try_repair,
        ffmpeg_path=ffmpeg_path,
    )


if __name__ == "__main__":
    main()