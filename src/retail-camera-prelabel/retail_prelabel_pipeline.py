import os
import re
import shutil
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import RTDETR


# ============================================================
# CONFIG (PLACEHOLDERS) — CAMBIA ESTO EN TU PC
# ============================================================
# Estructura esperada:
# dataset_root/
#   cam1/images/
#   cam1/output/labels/
#   cam1/output/deliveries/
#   cam2/images/ ...
DATASET_ROOT = r"C:\path\to\your_dataset_root"  # <-- CAMBIA

# Cámaras disponibles (puedes agregar/quitar)
CAMERAS = ["cam1", "cam2", "cam6"]

# YAML por cámara (opcional). Si no usas máscara, deja None o elimina la cámara del dict.
YAML_PATHS = {
    "cam1": r"C:\path\to\fisheye_cam1.yaml",  # <-- CAMBIA o pon None
    "cam2": r"C:\path\to\fisheye_cam2.yaml",  # <-- CAMBIA o pon None
    "cam6": r"C:\path\to\fisheye_cam6.yaml",  # <-- CAMBIA o pon None
}

# Modelo por cámara (placeholder). Puedes usar el mismo .pt para todas si quieres.
MODEL_PATHS = {
    "cam1": r"C:\path\to\model_cam1.pt",  # <-- CAMBIA
    "cam2": r"C:\path\to\model_cam2.pt",  # <-- CAMBIA
    "cam6": r"C:\path\to\model_cam6.pt",  # <-- CAMBIA
}

# Umbral por cámara
CONF_THRESHOLDS = {"cam1": 0.60, "cam2": 0.60, "cam6": 0.60}

# Clases (genéricas). Ajusta a tu proyecto.
# Ojo: estas keys deben coincidir con los nombres de clase que entrega tu modelo (results.names).
CLASS_DICT = {
    "class_a": 0,
    "class_b": 1,
    "class_c": 2,
    "class_d": 3,
}


# ============================================================
# RENOMBRADO (timestamp normalization)
# 2025-06-01-14-12-54-020339.jpg -> 2025-06-01_14-12-54-020339.jpg
# ============================================================
PATRON_FECHA = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2}-\d+)")

def renombrar_imagenes(images_dir: str) -> int:
    cambios = 0
    for archivo in os.listdir(images_dir):
        ruta_original = os.path.join(images_dir, archivo)
        if not os.path.isfile(ruta_original):
            continue
        nuevo_nombre = PATRON_FECHA.sub(r"\1_\2", archivo)
        if nuevo_nombre != archivo:
            ruta_nueva = os.path.join(images_dir, nuevo_nombre)
            if os.path.exists(ruta_nueva):
                # No pisa
                continue
            os.rename(ruta_original, ruta_nueva)
            cambios += 1
    return cambios


# ============================================================
# MÁSCARA / CROP (opcional vía YAML)
# ============================================================
def aplicar_mascara(images_dir: str, output_root: str, yaml_path: str) -> str:
    """
    Aplica rotación + crop + máscara poligonal, guarda en:
      output_root/masked/images/
    Retorna el nuevo images_dir (masked/images).
    """
    import yaml
    from PIL import Image

    if not yaml_path or not os.path.exists(yaml_path):
        print("⚠️ YAML no existe o no está configurado. Se omite máscara.")
        return images_dir

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    masked_images_dir = os.path.join(output_root, "masked", "images")
    os.makedirs(masked_images_dir, exist_ok=True)

    archivos = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for filename in tqdm(archivos, desc="🖼️ Enmascarando"):
        image_path = os.path.join(images_dir, filename)
        im = Image.open(image_path)

        # Se espera que el YAML tenga:
        # FishEye.Angle
        # FishEye.CropSize.Left/Right/Top/Bottom
        # Track.Recinto (polígono)
        img = im.rotate(cfg["FishEye"]["Angle"])

        left = cfg["FishEye"]["CropSize"]["Left"]
        right = img.size[0] - cfg["FishEye"]["CropSize"]["Right"]
        top = cfg["FishEye"]["CropSize"]["Top"]
        bottom = img.size[1] - cfg["FishEye"]["CropSize"]["Bottom"]
        img_crop = img.crop((left, top, right, bottom))

        img_np = cv2.cvtColor(np.array(img_crop), cv2.COLOR_RGB2BGR)

        mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        recinto = np.array(cfg["Track"]["Recinto"], np.int32)
        cv2.fillPoly(mask, [recinto], 255)
        img_masked = cv2.bitwise_and(img_np, img_np, mask=mask)

        cv2.imwrite(os.path.join(masked_images_dir, filename), img_masked)

    return masked_images_dir


# ============================================================
# PREETIQUETADO (RT-DETR -> YOLO)
# ============================================================
def write_classes_txt(labels_dir: str, class_names: list[str]):
    os.makedirs(labels_dir, exist_ok=True)
    with open(os.path.join(labels_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")

def preetiquetar(
    images_dir: str,
    labels_dir: str,
    model_path: str,
    conf_th: float,
    clases_a_incluir=None,
    reindexar=False,
    mover_sin_etiquetas=True,
):
    """
    Genera YOLO labels. Si una imagen no tiene detecciones y mover_sin_etiquetas=True,
    la mueve a output/no_detections/.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RTDETR(model_path).to(device)
    os.makedirs(labels_dir, exist_ok=True)

    # classes.txt
    if clases_a_incluir:
        class_names = clases_a_incluir
    else:
        class_names = list(CLASS_DICT.keys())
    write_classes_txt(labels_dir, class_names)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        print("⚠️ No hay imágenes disponibles para preetiquetar.")
        return 0

    no_det_dir = None
    if mover_sin_etiquetas:
        no_det_dir = os.path.join(os.path.dirname(labels_dir), "no_detections")
        os.makedirs(no_det_dir, exist_ok=True)

    created = 0

    # invert dict para mapear id->nombre (según CLASS_DICT)
    inv_id_to_name = {v: k for k, v in CLASS_DICT.items()}

    for filename in tqdm(image_files, desc="📸 Prelabel"):
        base = os.path.splitext(filename)[0].lower()
        txt_path = os.path.join(labels_dir, f"{base}.txt")

        if os.path.exists(txt_path):
            continue

        path = os.path.join(images_dir, filename)
        img = cv2.imread(path)
        if img is None:
            continue

        H, W = img.shape[:2]
        try:
            preds = model(img)[0].boxes.data.cpu().numpy()
        except Exception as e:
            print(f"⚠️ Error con {filename}: {e}")
            continue

        yolo_lines = []
        for pred in preds:
            x1, y1, x2, y2, score, cls = pred
            if score < conf_th:
                continue

            cls = int(cls)

            # Convertir ID del modelo a nombre de clase:
            # - OJO: esto asume que el modelo fue entrenado con IDs que calzan con CLASS_DICT.
            # Si tu modelo usa otros IDs/nombres, ajusta CLASS_DICT o el mapeo.
            if cls not in inv_id_to_name:
                continue

            nombre_clase = inv_id_to_name[cls]

            if clases_a_incluir and nombre_clase not in clases_a_incluir:
                continue

            id_final = (
                clases_a_incluir.index(nombre_clase)
                if reindexar and clases_a_incluir
                else CLASS_DICT[nombre_clase]
            )

            xc = ((x1 + x2) / 2) / W
            yc = ((y1 + y2) / 2) / H
            ww = (x2 - x1) / W
            hh = (y2 - y1) / H

            yolo_lines.append(f"{id_final} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

        if yolo_lines:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
            created += 1
        else:
            if mover_sin_etiquetas and no_det_dir:
                shutil.move(path, os.path.join(no_det_dir, filename))

    print(f"✅ {created} etiquetas generadas.")
    return created


# ============================================================
# PARES + ENTREGA (100) + ZIP
# ============================================================
def verificar_pares(images_dir: str, labels_dir: str):
    imgs = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    lbls = {
        os.path.splitext(f)[0]
        for f in os.listdir(labels_dir)
        if f.endswith(".txt") and f != "classes.txt"
    }
    return sorted(imgs & lbls)

def agrupar_y_comprimir(pares, images_dir: str, labels_dir: str, deliveries_dir: str, batch_size: int = 100):
    print(f"🔗 {len(pares)} pares válidos (imágenes + etiquetas)")

    if len(pares) < batch_size:
        total_imgs = len([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print("❌ No hay suficientes pares.")
        print(f"📸 Pero hay {total_imgs} imágenes en la carpeta 'images'.")
        return

    while True:
        cantidad = input(f"📦 ¿Cuántas carpetas deseas crear ({batch_size} c/u)? (0 o ENTER para volver): ").strip()
        if cantidad in ["", "0"]:
            print("⏪ Cancelado.")
            return
        try:
            num = int(cantidad)
            if num * batch_size > len(pares):
                print("❌ No hay suficientes pares. Ingresa un número menor.")
                continue
            break
        except ValueError:
            print("⚠️ Ingresa un número válido.")

    base = input("📌 Nombre base (ej: R001_cam1_20250521): ").strip()
    if len(base) < 4 or not base[1:4].isdigit():
        print("❌ Formato inválido. Usa algo como R001_cam1_YYYYMMDD")
        return

    letra = base[0]
    numero = int(base[1:4])
    sufijo = base[4:]

    zips_dir = os.path.join(deliveries_dir, "zips")
    os.makedirs(zips_dir, exist_ok=True)
    os.makedirs(deliveries_dir, exist_ok=True)

    # clases.txt
    classes_src = os.path.join(labels_dir, "classes.txt")
    if not os.path.exists(classes_src):
        write_classes_txt(labels_dir, list(CLASS_DICT.keys()))
        classes_src = os.path.join(labels_dir, "classes.txt")

    pares.sort()
    total_imgs, total_lbls = 0, 0

    for i in range(num):
        nuevo_num = numero + i
        nuevo_nombre = f"{letra}{nuevo_num:03d}{sufijo}"
        carpeta = os.path.join(deliveries_dir, nuevo_nombre)

        img_out = os.path.join(carpeta, "images")
        lbl_out = os.path.join(carpeta, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        shutil.copy2(classes_src, os.path.join(lbl_out, "classes.txt"))

        subset = pares[i * batch_size : (i + 1) * batch_size]
        for name in subset:
            moved_img = False
            for ext in [".jpg", ".jpeg", ".png"]:
                src = os.path.join(images_dir, name + ext)
                if os.path.exists(src):
                    shutil.move(src, os.path.join(img_out, name + ext))
                    total_imgs += 1
                    moved_img = True
                    break
            if not moved_img:
                print(f"⚠️ Imagen no encontrada: {name}")

            label_src = os.path.join(labels_dir, name + ".txt")
            if os.path.exists(label_src):
                shutil.move(label_src, os.path.join(lbl_out, name + ".txt"))
                total_lbls += 1
            else:
                print(f"⚠️ Label no encontrada: {name}")

        zip_base = os.path.join(zips_dir, os.path.basename(carpeta))
        shutil.make_archive(zip_base, "zip", carpeta)
        print(f"✅ Carpeta comprimida: {nuevo_nombre}")

    print(f"🏁 Total imágenes movidas: {total_imgs}, total etiquetas movidas: {total_lbls}")


# ============================================================
# MENÚ
# ============================================================
def ejecutar_menu():
    # Selección de cámara
    while True:
        print("\n🎥 Selecciona cámara:")
        for i, c in enumerate(CAMERAS, 1):
            print(f"  {i}. {c}")
        print("  0. Salir")

        try:
            op = int(input("Opción: "))
            if op == 0:
                print("👋 Saliendo.")
                return
            if 1 <= op <= len(CAMERAS):
                break
        except ValueError:
            pass
        print("❌ Opción inválida.")

    cam = CAMERAS[op - 1]
    cam_root = os.path.join(DATASET_ROOT, cam)

    images_dir = os.path.join(cam_root, "images")
    output_root = os.path.join(cam_root, "output")
    labels_dir = os.path.join(output_root, "labels")
    deliveries_dir = os.path.join(output_root, "deliveries")

    model_path = MODEL_PATHS.get(cam)
    conf_th = CONF_THRESHOLDS.get(cam, 0.60)
    yaml_path = YAML_PATHS.get(cam)

    # Validaciones
    if not os.path.isdir(DATASET_ROOT):
        print(f"❌ DATASET_ROOT no existe: {DATASET_ROOT}")
        return
    if not os.path.isdir(images_dir):
        print(f"❌ Falta carpeta: {images_dir}")
        print("   Se espera: <DATASET_ROOT>/<cam>/images/")
        return
    if not model_path or not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado para {cam}: {model_path}")
        print("   Edita MODEL_PATHS y apunta a tu .pt")
        return

    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(deliveries_dir, exist_ok=True)

    print(f"\n🧠 Modelo: {os.path.basename(model_path)} | conf_th={conf_th}")

    # Paso 1: renombrar
    print("\n🟩 PASO 1: Renombrar imágenes")
    ren = renombrar_imagenes(images_dir)
    print(f"✅ {ren} imágenes renombradas.")

    # Paso 2: máscara (opcional)
    aplicar = input("¿Deseas aplicar máscara/crop con YAML? (s/n): ").strip().lower()
    if aplicar == "s":
        print("\n🟩 PASO 2: Máscara YAML")
        images_dir = aplicar_mascara(images_dir, cam_root, yaml_path)
        print(f"✅ Usando images_dir: {images_dir}")
    else:
        print("\n⏭️  Máscara omitida.")

    # Paso 3: preetiquetar
    pre = input("\n¿Deseas preetiquetar (RT-DETR -> YOLO)? (s/n): ").strip().lower()
    if pre != "s":
        print("⏪ Cancelado.")
        return

    aplicar_filtro = input("¿Filtrar por clases específicas? (s/n): ").strip().lower() == "s"
    if aplicar_filtro:
        disponibles = list(CLASS_DICT.keys())
        print("Clases disponibles:", ", ".join(disponibles))
        clases_input = input("Ingresa clases separadas por coma (ej: class_a,class_c): ").lower()
        clases_a_incluir = [c.strip() for c in clases_input.split(",") if c.strip() in disponibles]
        if not clases_a_incluir:
            print("⚠️ No se ingresaron clases válidas. Se etiquetará todo.")
            clases_a_incluir = None
            reindexar = False
        else:
            reindexar = input("¿Reindexar desde 0 (YOLO)? (s/n): ").strip().lower() == "s"
    else:
        clases_a_incluir = None
        reindexar = False

    mover_sin = input("¿Mover imágenes sin detecciones a output/no_detections? (s/n): ").strip().lower() == "s"

    print("\n🟩 PASO 3: Pre-etiquetado")
    preetiquetar(
        images_dir=images_dir,
        labels_dir=labels_dir,
        model_path=model_path,
        conf_th=conf_th,
        clases_a_incluir=clases_a_incluir,
        reindexar=reindexar,
        mover_sin_etiquetas=mover_sin,
    )

    # Paso 4: verificar
    print("\n🟩 PASO 4: Verificar pares")
    pares = verificar_pares(images_dir, labels_dir)
    print(f"✅ Pairs válidos: {len(pares)}")

    # Paso 5: agrupar + zip
    print("\n🟩 PASO 5: Agrupar + ZIP")
    agrupar_y_comprimir(pares, images_dir, labels_dir, deliveries_dir, batch_size=100)

    print("\n✅ Proceso finalizado.")


if __name__ == "__main__":
    ejecutar_menu()