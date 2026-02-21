import os
import re
import shutil
import zipfile
from glob import glob

import cv2
from ultralytics import RTDETR

# ============================================================
# CONFIG (EJEMPLO / PLACEHOLDERS) — CAMBIA ESTO EN TU PC
# ============================================================
# Estructura esperada:
# dataset_root/
#   images/
#   labels/            (se crea si no existe)
#   deliveries/        (se crea si no existe)
#   zips/              (se crea si no existe)
#   classes.txt        (se genera automáticamente)

DATASET_ROOT = r"C:\path\to\your_dataset"          # <-- CAMBIA ESTO
IMAGES_DIR   = os.path.join(DATASET_ROOT, "images")
LABELS_DIR   = os.path.join(DATASET_ROOT, "labels")
CLASSES_PATH = os.path.join(DATASET_ROOT, "classes.txt")

DELIVERIES_DIR = os.path.join(DATASET_ROOT, "deliveries")  # carpetas de 100
ZIPS_DIR       = os.path.join(DATASET_ROOT, "zips")        # zips finales

WEIGHTS_PATH = r"C:\path\to\your_model.pt"         # <-- CAMBIA ESTO (peso del modelo)
CONF_TH = 0.80

# Tus 6 clases (genéricas; ajusta los nombres a tu caso)
CLASS_DICT = {
    "ClassA": 0,
    "ClassB": 1,
    "ClassC": 2,
    "ClassD": 3,
    "ClassE": 4,
    "ClassF": 5,
}
ADMITTED_CLASSES = list(CLASS_DICT.keys())

# ------------------------------------------------------------
# Renombrado: 2025-06-01-14-12-54-020339.jpg -> 2025-06-01_14-12-54-020339.jpg
# ------------------------------------------------------------
PATRON_FECHA = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2}-\d+)")

def nuevo_nombre_destino(nombre: str) -> str:
    """Devuelve el nombre con '_' entre fecha y hora si aplica al patrón."""
    return PATRON_FECHA.sub(r"\1_\2", nombre)

def detectar_conflictos():
    """Lista (src_abs, dst_abs) donde el nuevo nombre ya existe."""
    conflictos = []
    for archivo in os.listdir(IMAGES_DIR):
        src = os.path.join(IMAGES_DIR, archivo)
        if not os.path.isfile(src):
            continue
        dst_name = nuevo_nombre_destino(archivo)
        if dst_name == archivo:
            continue
        dst = os.path.join(IMAGES_DIR, dst_name)
        if os.path.exists(dst):
            conflictos.append((src, dst))
    return conflictos

def resolver_conflictos(conflictos):
    """
    Opción de borrar destinos existentes (y su label) para poder renombrar sin choques.
    No borra los archivos de origen.
    """
    if not conflictos:
        print("✅ No hay conflictos de nombres.")
        return

    print(f"⚠️ Conflictos detectados: {len(conflictos)} (destino ya existe con el nombre renombrado).")
    print("Ejemplos (hasta 10):")
    for i, (src, dst) in enumerate(conflictos[:10], 1):
        print(f"  {i:02d}. {os.path.basename(src)} -> {os.path.basename(dst)}  (DESTINO YA EXISTE)")

    resp = input("\n¿Deseas BORRAR los destinos existentes y sus labels para poder renombrar? (s/n): ").strip().lower()
    if resp != "s":
        print("⏭️  No se eliminaron destinos. El renombrado saltará esos casos.")
        return

    confirm = input("Escribe 'BORRAR' para confirmar eliminación permanente de destinos: ").strip()
    if confirm != "BORRAR":
        print("❌ Confirmación inválida. No se eliminó nada.")
        return

    borrados = 0
    for _, dst in conflictos:
        try:
            if os.path.exists(dst):
                os.remove(dst)
                borrados += 1

            base = os.path.splitext(os.path.basename(dst))[0]
            lbl_path = os.path.join(LABELS_DIR, base + ".txt")
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
        except Exception as e:
            print(f"  ⚠️ No se pudo borrar {dst}: {e}")

    print(f"🗑️  Destinos eliminados: {borrados}")

def renombrar_imagenes():
    """
    Renombra archivos en IMAGES_DIR reemplazando '-' por '_' entre fecha y hora.
    Si el destino existe, lo salta.
    """
    cambios = 0
    for archivo in os.listdir(IMAGES_DIR):
        src = os.path.join(IMAGES_DIR, archivo)
        if not os.path.isfile(src):
            continue
        dst_name = nuevo_nombre_destino(archivo)
        if dst_name == archivo:
            continue
        dst = os.path.join(IMAGES_DIR, dst_name)
        if os.path.exists(dst):
            print(f"[SKIP] Ya existe destino: {dst_name}")
            continue
        os.rename(src, dst)
        cambios += 1
    return cambios

def calcular_centro(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

class Detector:
    def __init__(self, weight_path: str, confidence_threshold: float):
        print(f"\n🔄 Cargando modelo desde: {weight_path}")
        self.model = RTDETR(weight_path)
        self.conf_th = float(confidence_threshold)
        self.device = "cpu"  # cambia a "0" si quieres GPU (si tienes CUDA)

    def inferencia(self, img):
        results = self.model.predict(
            img,
            imgsz=640,
            conf=self.conf_th,
            verbose=False,
            device=self.device,
            half=False,
        )[0]
        bboxes = results.boxes.xyxy.cpu().detach().numpy().tolist()
        class_ids = results.boxes.cls.cpu().detach().numpy().tolist()
        class_labels = [results.names[int(el)] for el in class_ids]
        return bboxes, class_labels

def etiquetar():
    """
    Crea labels en formato YOLO (txt) para cada imagen que no tenga label aún.
    Si no hay detecciones admitidas, borra el txt vacío.
    """
    image_paths = (
        glob(os.path.join(IMAGES_DIR, "*.png"))
        + glob(os.path.join(IMAGES_DIR, "*.jpg"))
        + glob(os.path.join(IMAGES_DIR, "*.jpeg"))
    )

    os.makedirs(LABELS_DIR, exist_ok=True)

    detector = Detector(weight_path=WEIGHTS_PATH, confidence_threshold=CONF_TH)
    count = 0

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        base = os.path.splitext(filename)[0]
        txt_outpath = os.path.join(LABELS_DIR, base + ".txt")

        if os.path.exists(txt_outpath):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        Habs, Wabs = img.shape[:2]
        bboxes, class_labels = detector.inferencia(img)

        wrote = False
        with open(txt_outpath, "w", encoding="utf-8") as f:
            for cl, bbox in zip(class_labels, bboxes):
                if cl not in ADMITTED_CLASSES:
                    continue

                cl_id = CLASS_DICT[cl]
                xc, yc = calcular_centro(bbox[0], bbox[1], bbox[2], bbox[3])
                wr = bbox[2] - bbox[0]
                hr = bbox[3] - bbox[1]

                x1 = xc / Wabs
                y1 = yc / Habs
                w1 = wr / Wabs
                h1 = hr / Habs

                f.write(f"{cl_id} {x1:.4f} {y1:.4f} {w1:.4f} {h1:.4f}\n")
                wrote = True

        if wrote:
            count += 1
        else:
            if os.path.exists(txt_outpath) and os.path.getsize(txt_outpath) == 0:
                os.remove(txt_outpath)

    # Genera classes.txt
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        for name in CLASS_DICT:
            f.write(name + "\n")

    return count

def verificar_pares():
    imagenes = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    labels = sorted([f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")])

    nombres_img = {os.path.splitext(f)[0] for f in imagenes}
    nombres_lbl = {os.path.splitext(f)[0] for f in labels}

    pares = list(nombres_img & nombres_lbl)
    return pares, len(imagenes), len(labels)

def crear_carpetas_de_entrega(pares):
    print(f"\n🧮 Total de pares válidos: {len(pares)}")
    if len(pares) < 100:
        print("❌ No hay suficientes pares para crear una carpeta (mínimo 100).")
        return False

    opcion = input("¿Deseas agrupar los datos en carpetas de 100 para entregar? (s/n): ").strip().lower()
    if opcion != "s":
        return False

    os.makedirs(DELIVERIES_DIR, exist_ok=True)
    os.makedirs(ZIPS_DIR, exist_ok=True)

    nombre_base = input("➡️ Nombre base de la PRIMERA carpeta (ej: P001_project_YYYYMMDD): ").strip()
    if not re.match(r"^P\d{3}_", nombre_base):
        print("❌ El nombre debe comenzar con 'P' + 3 dígitos + '_' (ej: P001_...).")
        return False

    num_carpetas = int(input("¿Cuántas carpetas crear (100 pares por carpeta)? "))
    if num_carpetas * 100 > len(pares):
        print("❌ No hay suficientes pares para esa cantidad de carpetas.")
        return False

    print("\n🔍 Carpetas que se generarán:")
    for i in range(num_carpetas):
        nombre = f"{nombre_base[:1]}{int(nombre_base[1:4]) + i:03d}_{'_'.join(nombre_base.split('_')[1:])}"
        print(f"  - {nombre}")
    confirmar = input("¿Confirmas? (s/n): ").strip().lower()
    if confirmar != "s":
        print("❌ Operación cancelada.")
        return False

    pares.sort()
    total_creadas = 0

    for i in range(num_carpetas):
        subset = pares[i * 100 : (i + 1) * 100]
        if not subset:
            break

        nombre = f"{nombre_base[:1]}{int(nombre_base[1:4]) + i:03d}_{'_'.join(nombre_base.split('_')[1:])}"
        carpeta = os.path.join(DELIVERIES_DIR, nombre)

        os.makedirs(os.path.join(carpeta, "images"), exist_ok=True)
        os.makedirs(os.path.join(carpeta, "labels"), exist_ok=True)

        shutil.copy2(CLASSES_PATH, os.path.join(carpeta, "labels", "classes.txt"))

        for base in subset:
            # Mover imagen
            imagen_movida = False
            for ext in [".jpg", ".jpeg", ".png"]:
                src_img = os.path.join(IMAGES_DIR, base + ext)
                if os.path.exists(src_img):
                    shutil.move(src_img, os.path.join(carpeta, "images", base + ext))
                    imagen_movida = True
                    break
            if not imagen_movida:
                print(f"⚠️ Imagen no encontrada para: {base}")

            # Mover label
            lbl_src = os.path.join(LABELS_DIR, base + ".txt")
            if os.path.exists(lbl_src):
                shutil.move(lbl_src, os.path.join(carpeta, "labels", base + ".txt"))
            else:
                print(f"⚠️ Label no encontrada para: {base}")

        total_creadas += 1

    if total_creadas == 0:
        print("❌ No se crearon carpetas nuevas.")
        return False

    total_usadas = total_creadas * 100
    sobrantes = len(pares) - total_usadas
    if sobrantes > 0:
        print(f"\n⚠️ Quedaron {sobrantes} pares sin mover (se quedan en la carpeta original).")
    else:
        print("\n✅ Todo fue agrupado sin sobrantes.")

    print(f"✅ Se han creado {total_creadas} carpetas en {DELIVERIES_DIR}")
    return True

def comprimir_carpetas():
    print("\n📦 Comprobando carpetas para comprimir...")
    for folder in os.listdir(DELIVERIES_DIR):
        folder_path = os.path.join(DELIVERIES_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        zip_path = os.path.join(ZIPS_DIR, folder + ".zip")
        if os.path.exists(zip_path):
            continue

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, folder_path)
                    zipf.write(abs_path, os.path.join(folder, rel_path))

        print(f"✅ Carpeta comprimida: {folder}")

def main():
    # Validaciones mínimas
    if not os.path.isdir(DATASET_ROOT):
        print(f"❌ DATASET_ROOT no existe: {DATASET_ROOT}")
        print("   Edita DATASET_ROOT en el script y apunta a tu dataset.")
        return
    if not os.path.isdir(IMAGES_DIR):
        print(f"❌ IMAGES_DIR no existe: {IMAGES_DIR}")
        print("   Se espera una carpeta 'images' dentro de DATASET_ROOT.")
        return

    os.makedirs(LABELS_DIR, exist_ok=True)

    print("\n🟩 PASO 1: Conflictos y renombrado de imágenes")
    conflictos = detectar_conflictos()
    print(f"🔎 Conflictos detectados: {len(conflictos)}" if conflictos else "✅ No hay conflictos previos.")
    resolver_conflictos(conflictos)

    renombrados = renombrar_imagenes()
    print(f"✅ {renombrados} imágenes renombradas." if renombrados else "⚠️ No hubo cambios de renombrado.")

    print("\n🟩 PASO 2: Pre-etiquetado (RT-DETR) -> YOLO labels")
    nuevas = etiquetar()
    print(f"✅ {nuevas} labels creadas." if nuevas else "⚠️ No se crearon labels nuevas.")

    print("\n🟩 PASO 3: Verificación de pares (imagen + label)")
    pares, n_imgs, n_lbls = verificar_pares()
    if pares:
        print(f"✅ {len(pares)} pares válidos de {n_imgs} imágenes y {n_lbls} labels")
    else:
        print("❌ No hay pares válidos (revisa inferencia/clases/directorios).")
        return

    print("\n🟩 PASO 4: Agrupación para entrega")
    creado = crear_carpetas_de_entrega(pares)

    if creado:
        print("\n🟩 PASO 5: Compresión ZIP")
        comprimir_carpetas()
    else:
        print("\n⚠️ No se crearon carpetas nuevas, omitiendo compresión.")

    print("\n✅ Proceso completo.")

if __name__ == "__main__":
    main()