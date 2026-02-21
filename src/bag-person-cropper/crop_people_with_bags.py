import os
import csv
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import RTDETR, YOLO


# ============================================================
# CONFIG (PLACEHOLDERS) — CAMBIA ESTO EN TU PC
# ============================================================
MODEL_PATH = r"C:\path\to\person_detector.pt"  # modelo detector de PERSONAS (por defecto clase 0)
IMAGES_DIR = r"C:\path\to\dataset\images"
LABELS_DIR = r"C:\path\to\dataset\labels"

OUTPUT_DIR = r"C:\path\to\output\bag_person_crops"

# Detector personas
CONF_PERSON = 0.80
PERSON_CLASS_ID_MODEL = 0   # en el MODELO: 0 = persona (ajusta si tu modelo usa otro id)

# Labels de entrada (YOLO)
BAG_CLASS_ID = 0            # en tus labels: 0 = bolsa (ajusta si corresponde)

# Padding base alrededor del recorte final (se aplica asimétrico)
PADDING = 10

# Criterio de asignación bolsa->persona:
#   'center': el centro de la bolsa debe caer dentro de la caja de persona
#   'iou':    se asigna si IoU >= IOU_THRESHOLD
MATCH_MODE = "center"
IOU_THRESHOLD = 0.10


# ============================================================
# Helpers
# ============================================================
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def xywhn_to_xyxy_abs(xc, yc, w, h, W, H):
    x = xc * W
    y = yc * H
    bw = w * W
    bh = h * H
    return [x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2]


def xyxy_to_yolo_norm(x1, y1, x2, y2, W, H):
    ww = max(0.0, x2 - x1)
    hh = max(0.0, y2 - y1)
    if ww <= 0 or hh <= 0:
        return None
    xc = x1 + ww / 2.0
    yc = y1 + hh / 2.0
    return xc / W, yc / H, ww / W, hh / H


def crop_image(img, box):
    x1, y1, x2, y2 = map(int, box)
    return img[y1:y2, x1:x2]


def ensure_dirs(output_dir):
    os.makedirs(os.path.join(output_dir, "crops_with_bags", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "crops_with_bags", "labels"), exist_ok=True)

    classes_path = os.path.join(output_dir, "crops_with_bags", "labels", "classes.txt")
    if not os.path.isfile(classes_path):
        with open(classes_path, "w", encoding="utf-8") as f:
            f.write("bag\n")  # id 0 (output labels are always bag-only)


def load_bags_from_label(txt_path, W, H, bag_class_id):
    bags = []
    if not os.path.isfile(txt_path):
        return bags

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue
            try:
                cid = int(float(p[0]))
            except ValueError:
                continue
            if cid != bag_class_id:
                continue

            xc, yc, w, h = map(float, p[1:])
            x1, y1, x2, y2 = xywhn_to_xyxy_abs(xc, yc, w, h, W, H)
            bags.append([x1, y1, x2, y2])

    return bags


def load_model(path):
    """
    Try RTDETR first; if it fails, fallback to YOLO.
    """
    try:
        return RTDETR(path), "rtdetr"
    except Exception:
        return YOLO(path), "yolo"


def assign_bags_to_persons_many_to_one(person_boxes, person_confs, bag_boxes, mode="center", iou_thr=0.10):
    """
    Many bags -> one person mapping.
    Returns: dict { idx_person: [idx_bag, ...] }
    """
    mapping = {}
    for j, bb in enumerate(bag_boxes):
        best = None  # (score_tuple, idx_person)
        cx = (bb[0] + bb[2]) / 2.0
        cy = (bb[1] + bb[3]) / 2.0

        for i, pb in enumerate(person_boxes):
            if mode == "center":
                if not (pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]):
                    continue
                iou_val = iou(pb, bb)
                score = (iou_val, float(person_confs[i]))
            else:
                iou_val = iou(pb, bb)
                if iou_val < iou_thr:
                    continue
                score = (iou_val, float(person_confs[i]))

            if best is None or score > best[0]:
                best = (score, i)

        if best is not None:
            _, i_best = best
            mapping.setdefault(i_best, []).append(j)

    return mapping


def tight_union_person_bags(person_box, bag_boxes_for_person, base_pad, W, H):
    """
    Union(person, all assigned bags) then asym padding biased towards bags.
    """
    x1, y1, x2, y2 = person_box

    if bag_boxes_for_person:
        bx1 = min(bb[0] for bb in bag_boxes_for_person)
        by1 = min(bb[1] for bb in bag_boxes_for_person)
        bx2 = max(bb[2] for bb in bag_boxes_for_person)
        by2 = max(bb[3] for bb in bag_boxes_for_person)

        ux1, uy1, ux2, uy2 = min(x1, bx1), min(y1, by1), max(x2, bx2), max(y2, by2)

        bcx = sum((bb[0] + bb[2]) / 2 for bb in bag_boxes_for_person) / len(bag_boxes_for_person)
        bcy = sum((bb[1] + bb[3]) / 2 for bb in bag_boxes_for_person) / len(bag_boxes_for_person)
    else:
        ux1, uy1, ux2, uy2 = x1, y1, x2, y2
        bcx = (x1 + x2) / 2
        bcy = (y1 + y2) / 2

    pcx = (x1 + x2) / 2
    pcy = (y1 + y2) / 2

    near_mul, far_mul = 1.3, 0.6
    min_pad, max_pad = 4, 24

    pad_left = int(np.clip(base_pad * (near_mul if bcx < pcx else far_mul), min_pad, max_pad))
    pad_right = int(np.clip(base_pad * (near_mul if bcx >= pcx else far_mul), min_pad, max_pad))
    pad_top = int(np.clip(base_pad * (near_mul if bcy < pcy else far_mul), min_pad, max_pad))
    pad_bot = int(np.clip(base_pad * (near_mul if bcy >= pcy else far_mul), min_pad, max_pad))

    X1 = max(0, int(np.floor(ux1) - pad_left))
    Y1 = max(0, int(np.floor(uy1) - pad_top))
    X2 = min(W - 1, int(np.ceil(ux2) + pad_right))
    Y2 = min(H - 1, int(np.ceil(uy2) + pad_bot))

    if X2 <= X1:
        X2 = min(W - 1, X1 + 1)
    if Y2 <= Y1:
        Y2 = min(H - 1, Y1 + 1)

    return [X1, Y1, X2, Y2]


def write_crop_and_labels(img, crop_box, base_name, out_images_dir, out_labels_dir, bag_boxes_for_person):
    crop = crop_image(img, crop_box)
    out_img = os.path.join(out_images_dir, f"{base_name}.jpg")
    cv2.imwrite(out_img, crop)

    cx1, cy1, cx2, cy2 = map(int, crop_box)
    cW = max(1, cx2 - cx1)
    cH = max(1, cy2 - cy1)

    txt_path = os.path.join(out_labels_dir, f"{base_name}.txt")
    lines = []

    for bb in bag_boxes_for_person:
        x1 = clamp(bb[0] - cx1, 0, cW)
        y1 = clamp(bb[1] - cy1, 0, cH)
        x2 = clamp(bb[2] - cx1, 0, cW)
        y2 = clamp(bb[3] - cy1, 0, cH)

        if (x2 - x1) <= 1 or (y2 - y1) <= 1:
            continue

        yolo = xyxy_to_yolo_norm(x1, y1, x2, y2, cW, cH)
        if yolo is None:
            continue

        xc, yc, w, h = yolo
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")  # output class 0 = bag

    with open(txt_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln)

    return out_img, txt_path


def collect_images(images_dir):
    imgs = []
    for ext in VALID_EXTS:
        imgs.extend(glob(os.path.join(images_dir, f"*{ext}")))
    imgs.sort()
    return imgs


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.isdir(IMAGES_DIR):
        raise SystemExit(f"IMAGES_DIR not found: {IMAGES_DIR}")
    if not os.path.isdir(LABELS_DIR):
        raise SystemExit(f"LABELS_DIR not found: {LABELS_DIR}")

    ensure_dirs(OUTPUT_DIR)
    out_images_dir = os.path.join(OUTPUT_DIR, "crops_with_bags", "images")
    out_labels_dir = os.path.join(OUTPUT_DIR, "crops_with_bags", "labels")

    model, kind = load_model(MODEL_PATH)
    print(f"Loaded model as: {kind}")

    imgs = collect_images(IMAGES_DIR)
    if not imgs:
        print("⚠️ No images found.")
        return

    resumen_csv = os.path.join(OUTPUT_DIR, "summary_people_with_bags.csv")

    with open(resumen_csv, "w", newline="", encoding="utf-8") as cf:
        wr = csv.writer(cf)
        wr.writerow(["image", "total_people", "people_with_bags", "person_indices"])

        for img_path in tqdm(imgs, desc="Processing"):
            img = cv2.imread(img_path)
            if img is None:
                continue

            H, W = img.shape[:2]
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(LABELS_DIR, f"{base}.txt")

            # 1) load bags from YOLO label
            bag_boxes = load_bags_from_label(label_path, W, H, BAG_CLASS_ID)
            if not bag_boxes:
                wr.writerow([os.path.basename(img_path), 0, 0, ""])
                continue

            # 2) detect people
            res = model.predict(img, conf=CONF_PERSON, imgsz=640, verbose=False)
            if not res:
                wr.writerow([os.path.basename(img_path), 0, 0, ""])
                continue

            det = res[0]
            if det.boxes is None or len(det.boxes) == 0:
                wr.writerow([os.path.basename(img_path), 0, 0, ""])
                continue

            boxes = det.boxes.xyxy.cpu().numpy()
            confs = det.boxes.conf.cpu().numpy()

            # Filter by model class id if available
            if hasattr(det.boxes, "cls") and det.boxes.cls is not None:
                cls_ids = det.boxes.cls.cpu().numpy().astype(int)
                mask = (cls_ids == PERSON_CLASS_ID_MODEL)
                boxes = boxes[mask]
                confs = confs[mask]

            total_people = len(boxes)
            if total_people == 0:
                wr.writerow([os.path.basename(img_path), 0, 0, ""])
                continue

            # 3) assign bags to people
            mapping = assign_bags_to_persons_many_to_one(
                np.array(boxes), confs, np.array(bag_boxes),
                mode=MATCH_MODE, iou_thr=IOU_THRESHOLD
            )

            if not mapping:
                wr.writerow([os.path.basename(img_path), total_people, 0, ""])
                continue

            # 4) crop each person that has >=1 bag
            person_indices = []
            for i_person, bag_idx_list in mapping.items():
                bbs_for_person = [bag_boxes[j] for j in bag_idx_list]
                crop_box = tight_union_person_bags(boxes[i_person], bbs_for_person, PADDING, W, H)

                out_name = f"{base}_person_{i_person}"
                write_crop_and_labels(img, crop_box, out_name, out_images_dir, out_labels_dir, bbs_for_person)
                person_indices.append(i_person)

            wr.writerow([
                os.path.basename(img_path),
                total_people,
                len(person_indices),
                ",".join(map(str, sorted(set(person_indices))))
            ])

    print("\n✅ Done")
    print(f"Images:  {out_images_dir}")
    print(f"Labels:  {out_labels_dir}")
    print(f"Summary: {resumen_csv}")


if __name__ == "__main__":
    main()