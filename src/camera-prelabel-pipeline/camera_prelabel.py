import os
import shutil
import re
import zipfile
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import RTDETR


# ============================================================
# CONFIG (PLACEHOLDERS) — CAMBIA ESTO EN TU PC
# ============================================================
# Dataset root genérico (1 carpeta por cámara)
# dataset_root/
#   cam1/images/
#   cam1/output/labels/
#   cam1/output/deliveries/
#   cam2/images/ ...
DATASET_ROOT = r"C:\path\to\your_dataset_root"   # <-- CAMBIA ESTO

# Cámaras disponibles
CAMERAS = ["cam1", "cam2", "cam6"]

# YAML por cámara (opcional). Si no quieres máscara, déjalo None.
YAML_PATHS = {
    "cam1": r"C:\path\to\fisheye_cam1.yaml",   # <-- CAMBIA ESTO o pon None
    "cam2": r"C:\path\to\fisheye_cam2.yaml",   # <-- CAMBIA ESTO o pon None
    "cam6": r"C:\path\to\fisheye_cam6.yaml",   # <-- CAMBIA ESTO o pon None
}

# Modelo por cámara (placeholder). Puedes usar el mismo para todas si quieres.
MODEL_PATHS = {
    "cam1": r"C:\path\to\model_cam1.pt",       # <-- CAMBIA ESTO
    "cam2": r"C:\path\to\model_cam2.pt",       # <-- CAMBIA ESTO
    "cam6": r"C:\path\to\model_cam6.pt",       # <-- CAMBIA ESTO
}

CONF_THRESHOLDS = {
    "cam1": 0.60,
    "cam2": 0.60,
    "cam6": 0.60,
}

# Clases (genéricas). Ajusta nombres/IDs.
CLASS_DICT = {"persona": 0, "carro": 1, "canasto": 2, "bolsa": 3}


# ============================================================
# RENOMBRADO
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
                # Si ya existe, no pisa
                continue
            os.rename(ruta_original, ruta_nueva)
            cambios += 1
    return cambios


# ============================================================
# PREETIQUETADO (RT-DETR -> YOLO)
# ============================================================
def preetiquetar(
    images_dir: str,
    labels_dir: str,
    model_path: str,
    conf_th: float,
    clases_a_incluir=None,
    reindexar=False,
    usar_sin_etiquetas=True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RTDETR(model_path).to(device)
    os.makedirs(labels_dir, exist_ok=True)

    # classes.txt (para entregar)
    with open(os.path.join(labels_dir, "classes.txt"), "w", encoding="utf-8") as f:
        if clases_a_incluir:
            f.write("\n".join(clases_a_incluir) + "\n")
        else:
            f.write("\n".join(CLASS_DICT.keys()) + "\n")

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        print("⚠️ No hay imágenes disponibles para preetiquetar.")
        return 0

    if usar_sin_etiquetas:
        sin_labels_dir = os.path.join(os.path.dirname(labels_dir), "sin_etiquetas")
        os.makedirs(sin_labels_dir, exist_ok=True)

    creadas = 0
    for filename in tqdm(image_files, desc="📸 Etiquetando"):
        name = os.path.splitext(filename)[0].lower()
        txt_path = os.path.join(labels_dir, f"{name}.txt")

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
            print(f"Error con {filename}: {e}")
            continue

        yolo = []
        for pred in preds:
            x1, y1, x2, y2, score, cls = pred
            if score < conf_th:
                continue

            cls = int(cls)
            # Nombre de clase según CLASS_DICT
            inv = [k for k, v in CLASS_DICT.items() if v == cls]
            if not inv:
                continue
            nombre_clase = inv[0]

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

            yolo.append(f"{id_final} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

        if yolo:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo))
            creadas += 1
        else:
            if usar_sin_etiquetas:
                shutil.move(path, os.path.join(sin_labels_dir, filename))

    print(f"✅ {creadas} etiquetas generadas.")
    return creadas


def verificar_pares(images_dir: str, labels_dir: str):
    imgs = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    lbls = {
        os.path.splitext(f)[0]
        for f in os.listdir(labels_dir)
        if f.endswith(".txt") and f != "classes.txt"
    }
    return sorted(imgs & lbls)


# ============================================================
# AGRUPAR Y COMPRIMIR
# ============================================================
def agrupar_y_comprimir(pares, images_dir: str, labels_dir: str, salida_base: str):
    print(f"🔗 {len(pares)} pares válidos (imágenes + etiquetas)")

    if len(pares) < 100:
        total_imgs = len([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print("❌ No hay suficientes pares para crear una entrega (mínimo 100).")
        print(f"📸 Pero hay {total_imgs} imágenes en la carpeta 'images'.")
        return

    while True:
        try:
            cantidad = input("📦 ¿Cuántas carpetas deseas crear (100 c/u)? (0 o ENTER para volver): ").strip()
            if cantidad == "" or cantidad == "0":
                print("⏪ Volviendo.")
                return
            num = int(cantidad)
            if num * 100 > len(pares):
                print("❌ No hay suficientes pares. Ingresa un número menor.")
                continue
            break
        except ValueError:
            print("⚠️ Ingresa un número válido.")

    base = input("📌 Nombre base (ej: C001_cam1_20250521): ").strip()
    letra = base[0]
    numero = int(base[1:4])
    sufijo = base[4:]

    carpeta_zip = os.path.join(salida_base, "zips")
    os.makedirs(carpeta_zip, exist_ok=True)
    os.makedirs(salida_base, exist_ok=True)

    total_imgs, total_lbls = 0, 0

    for i in range(num):
        nuevo_num = numero + i
        nuevo_nombre = f"{letra}{nuevo_num:03d}{sufijo}"
        carpeta = os.path.join(salida_base, nuevo_nombre)

        img_out = os.path.join(carpeta, "images")
        lbl_out = os.path.join(carpeta, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        # copiar classes.txt
        classes_src = os.path.join(labels_dir, "classes.txt")
        if os.path.exists(classes_src):
            shutil.copy2(classes_src, os.path.join(lbl_out, "classes.txt"))

        subset = pares[i * 100 : (i + 1) * 100]

        for name in subset:
            # mover imagen
            moved = False
            for ext in [".jpg", ".jpeg", ".png"]:
                src = os.path.join(images_dir, name + ext)
                if os.path.exists(src):
                    shutil.move(src, os.path.join(img_out, name + ext))
                    total_imgs += 1
                    moved = True
                    break
            if not moved:
                print(f"⚠️ Imagen no encontrada: {name}")

            # mover label
            label_src = os.path.join(labels_dir, name + ".txt")
            if os.path.exists(label_src):
                shutil.move(label_src, os.path.join(lbl_out, name + ".txt"))
                total_lbls += 1
            else:
                print(f"⚠️ Label no encontrada: {name}")

        # zip carpeta
        shutil.make_archive(os.path.join(carpeta_zip, os.path.basename(carpeta)), "zip", carpeta)
        print(f"✅ Carpeta comprimida: {nuevo_nombre}")

    print(f"🏁 Total imágenes movidas: {total_imgs}, total etiquetas movidas: {total_lbls}")


# ============================================================
# MÁSCARA / CROP (opcional via YAML)
# ============================================================
def aplicar_mascara(images_dir: str, output_root: str, yaml_path: str):
    import yaml
    from PIL import Image

    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    masked_dir = os.path.join(output_root, "images")
    os.makedirs(masked_dir, exist_ok=True)

    archivos = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for filename in tqdm(archivos, desc="🖼️ Enmascarando"):
        image_path = os.path.join(images_dir, filename)
        im = Image.open(image_path)

        # Ejemplo de config esperado (igual a tu lógica)
        img = im.rotate(config["FishEye"]["Angle"])

        left = config["FishEye"]["CropSize"]["Left"]
        right = img.size[0] - config["FishEye"]["CropSize"]["Right"]
        top = config["FishEye"]["CropSize"]["Top"]
        bottom = img.size[1] - config["FishEye"]["CropSize"]["Bottom"]

        img_crop = img.crop((left, top, right, bottom))
        img_np = cv2.cvtColor(np.array(img_crop), cv2.COLOR_RGB2BGR)

        mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        recinto = np.array(config["Track"]["Recinto"], np.int32)
        cv2.fillPoly(mask, [recinto], 255)

        img_masked = cv2.bitwise_and(img_np, img_np, mask=mask)
        cv2.imwrite(os.path.join(masked_dir, filename), img_masked)


# ============================================================
# MENÚ
# ============================================================
def ejecutar_con_menu():
    # Seleccionar cámara
    while True:
        print("\n🎥 Selecciona cámara:")
        for i, c in enumerate(CAMERAS, 1):
            print(f"  {i}. {c}")
        print("  0. Salir")

        try:
            seleccion = int(input("Opción: "))
            if seleccion == 0:
                print("👋 Saliendo.")
                return
            if 1 <= seleccion <= len(CAMERAS):
                break
        except ValueError:
            pass
        print("❌ Opción inválida.")

    cam = CAMERAS[seleccion - 1]
    root = os.path.join(DATASET_ROOT, cam)

    images_dir = os.path.join(root, "images")
    labels_dir = os.path.join(root, "output", "labels")
    deliveries_dir = os.path.join(root, "output", "deliveries")

    model_path = MODEL_PATHS.get(cam)
    conf_th = CONF_THRESHOLDS.get(cam, 0.60)
    yaml_path = YAML_PATHS.get(cam)

    # Validaciones
    if not os.path.isdir(DATASET_ROOT):
        print(f"❌ DATASET_ROOT no existe: {DATASET_ROOT}")
        return
    if not os.path.isdir(images_dir):
        print(f"❌ No existe carpeta de imágenes: {images_dir}")
        print("   Se espera: <DATASET_ROOT>/<cam>/images/")
        return
    if not model_path or not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado para {cam}: {model_path}")
        print("   Edita MODEL_PATHS (ruta placeholder) y apunta a tu .pt")
        return

    print(f"\n🧠 Modelo: {os.path.basename(model_path)} | conf_th={conf_th}")

    # Paso 1: renombrar
    print("\n🔄 Renombrando imágenes...")
    ren = renombrar_imagenes(images_dir)
    print(f"✅ {ren} imágenes renombradas.")

    # Paso 2: máscara (opcional)
    aplicar = input("¿Deseas aplicar máscara con YAML? (s/n): ").strip().lower()
    if aplicar == "s":
        if yaml_path and os.path.exists(yaml_path):
            print("🧩 Aplicando máscara...")
            aplicar_mascara(images_dir, root, yaml_path)
            images_dir = os.path.join(root, "images")
        else:
            print("⚠️ YAML no configurado o no existe. Se omite máscara.")

    # Paso 3: preetiquetar
    pre = input("¿Deseas preetiquetar las imágenes? (s/n): ").strip().lower()
    if pre != "s":
        print("⏪ Cancelado por el usuario.")
        return

    aplicar_filtro = input("¿Deseas filtrar por clases específicas? (s/n): ").strip().lower() == "s"

    if aplicar_filtro:
        disponibles = list(CLASS_DICT.keys())
        print("Clases disponibles:", ", ".join(disponibles))
        clases_input = input("Ingresa clases separadas por coma (ej: bolsa,carro): ").lower()
        clases_a_incluir = [c.strip() for c in clases_input.split(",") if c.strip() in disponibles]

        if not clases_a_incluir:
            print("❌ No se ingresaron clases válidas. Se etiquetará todo.")
            clases_a_incluir = None
            reindexar = False
        else:
            reindexar = input("¿Deseas reindexar desde 0 (YOLO)? (s/n): ").strip().lower() == "s"
    else:
        clases_a_incluir = None
        reindexar = False

    usar_sin_etiquetas = input("¿Mover imágenes sin etiquetas a 'sin_etiquetas'? (s/n): ").strip().lower() == "s"

    print("\n🔍 Preetiquetando imágenes...")
    preetiquetar(
        images_dir=images_dir,
        labels_dir=labels_dir,
        model_path=model_path,
        conf_th=conf_th,
        clases_a_incluir=clases_a_incluir,
        reindexar=reindexar,
        usar_sin_etiquetas=usar_sin_etiquetas,
    )

    # Paso 4: verificar
    print("\n📂 Verificando pares...")
    pares = verificar_pares(images_dir, labels_dir)
    print(f"🔗 {len(pares)} pares válidos.")

    # Paso 5: agrupar + zip
    agrupar_y_comprimir(pares, images_dir, labels_dir, deliveries_dir)
    print("✅ Proceso finalizado.")


if __name__ == "__main__":
    ejecutar_con_menu()