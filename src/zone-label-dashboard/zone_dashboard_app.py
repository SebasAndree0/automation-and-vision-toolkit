import os
import re
import math
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import StringVar, BooleanVar

import yaml
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict


# ============================================================
# CONFIG (PLACEHOLDERS) — CAMBIA ESTO EN TU PC
# ============================================================
# Cada "proyecto/cámara" define:
#  - images: carpeta de imágenes
#  - labels: carpeta de labels YOLO (.txt)
#  - yaml: archivo YAML con zonas (rectangles o zonas)
#  - image_base: imagen de referencia para dibujar zonas
#  - classes_txt: classes.txt (una clase por línea)
#
# output_dir: carpeta donde se guardan los PNG finales
CONFIG = {
    "camera_a": {
        "images": r"C:\path\to\dataset\camera_a\images",
        "labels": r"C:\path\to\dataset\camera_a\labels",
        "yaml": r"C:\path\to\zones\camera_a_zones.yaml",
        "image_base": r"C:\path\to\reference_images\camera_a_sample.jpg",
        "classes_txt": r"C:\path\to\dataset\camera_a\classes.txt",
    },
    "camera_b": {
        "images": r"C:\path\to\dataset\camera_b\images",
        "labels": r"C:\path\to\dataset\camera_b\labels",
        "yaml": r"C:\path\to\zones\camera_b_zones.yaml",
        "image_base": r"C:\path\to\reference_images\camera_b_sample.jpg",
        "classes_txt": r"C:\path\to\dataset\camera_b\classes.txt",
    },
    # output base
    "output_dir": r"C:\path\to\output\results",
    # logo es opcional; NO se usa por defecto (lo dejamos por compatibilidad)
    "logo": None,
}


# ============================================================
# YAML helpers
# ============================================================
def rect_key(x: str):
    m = re.match(r"rect_(\d+)", str(x))
    return int(m.group(1)) if m else str(x)


def cargar_zonas(yaml_path: str):
    """
    Soporta 2 formatos:
      1) {"rectangles": {"rect_1":[x1,y1,x2,y2], "rect_2":[...], ...}}
      2) {"zonas": [ [x1,y1,x2,y2],  OR  [[x,y],[x,y],...] polygon, ... ]}
    Retorna (zonas, nombres)
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "rectangles" in raw:
        rects = raw["rectangles"] or {}
        zonas, nombres = [], []
        for k in sorted(rects, key=rect_key):
            zonas.append(rects[k])
            nombres.append(str(k))
        return zonas, nombres

    if "zonas" in raw:
        zonas = raw["zonas"] or []
        nombres = [f"Zona {i+1}" for i in range(len(zonas))]
        return zonas, nombres

    return [], []


def cargar_clases(classes_txt: str):
    with open(classes_txt, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


# ============================================================
# Dataset helpers
# ============================================================
def get_imagen_shape(images_dir: str):
    """
    Busca una imagen cualquiera para inferir (W,H).
    """
    if not os.path.isdir(images_dir):
        return None, None

    for f in os.listdir(images_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            p = os.path.join(images_dir, f)
            img = cv2.imread(p)
            if img is not None:
                h, w = img.shape[:2]
                return w, h
    return None, None


def iterar_labels_dir(labels_dir: str):
    if not os.path.isdir(labels_dir):
        return
    for f in os.listdir(labels_dir):
        if f.endswith(".txt") and f != "classes.txt":
            yield os.path.join(labels_dir, f)


def contar_etiquetas_por_zona(labels_dir: str, zonas, clases, images_dir: str):
    """
    Cuenta:
      - por zona y por clase
      - total por zona
      - global por clase
    Determina zona según el centro del bbox YOLO.
    """
    conteo_zona_clase = [defaultdict(int) for _ in zonas]
    conteo_zona_total = [0 for _ in zonas]
    conteo_global = defaultdict(int)

    w, h = get_imagen_shape(images_dir)
    if not w or not h:
        print("⚠️ No se pudo obtener tamaño de imagen desde images_dir.")
        return conteo_zona_clase, conteo_zona_total, conteo_global

    for label_path in iterar_labels_dir(labels_dir):
        with open(label_path, "r", encoding="utf-8") as f:
            for linea in f:
                datos = linea.strip().split()
                if len(datos) < 5:
                    continue

                try:
                    clase_idx = int(float(datos[0]))
                    x_c, y_c = float(datos[1]), float(datos[2])
                except ValueError:
                    continue

                if clase_idx < 0 or clase_idx >= len(clases):
                    continue

                clase_nombre = clases[clase_idx]
                cx, cy = int(x_c * w), int(y_c * h)

                conteo_global[clase_nombre] += 1

                for idx, zona in enumerate(zonas):
                    # Polígono: [[x,y], [x,y], ...]
                    if isinstance(zona, list) and zona and isinstance(zona[0], list):
                        poly = np.array(zona, dtype=np.int32)
                        if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                            conteo_zona_clase[idx][clase_nombre] += 1
                            conteo_zona_total[idx] += 1
                            break

                    # Rect: [x1,y1,x2,y2]
                    elif isinstance(zona, list) and len(zona) == 4:
                        x1, y1, x2, y2 = zona
                        x1, x2 = sorted([int(x1), int(x2)])
                        y1, y2 = sorted([int(y1), int(y2)])
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            conteo_zona_clase[idx][clase_nombre] += 1
                            conteo_zona_total[idx] += 1
                            break

    return conteo_zona_clase, conteo_zona_total, conteo_global


# ============================================================
# Drawing: zones on reference image
# ============================================================
def dibujar_zonas_en_imagen(yaml_path: str, img_path: str, save_path: str, highlighted_indices: set[int]):
    """
    Dibuja zonas (rect o polygon) en una imagen base.
    - highlight: indices en color A
    - no highlight: indices en color B
    """
    zonas, _ = cargar_zonas(yaml_path)

    pil_img = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)

    # Colores
    COLOR_A = (209, 123, 74, 255)   # naranjo
    COLOR_B = (0, 162, 255, 255)    # azul
    TEXT = (255, 255, 255, 255)

    try:
        font_label = ImageFont.truetype("arialbd.ttf", 17)
    except Exception:
        font_label = ImageFont.load_default()

    def text_size(font, text):
        try:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            return font.getsize(text)

    for idx, coords in enumerate(zonas):
        border = COLOR_A if idx in highlighted_indices else COLOR_B
        label = str(idx + 1)

        # Polygon
        if isinstance(coords, list) and coords and isinstance(coords[0], list):
            pts = [tuple(p) for p in coords]
            draw.polygon(pts, outline=border, width=3)

            # label at centroid
            cx = int(sum(p[0] for p in coords) / len(coords))
            cy = int(sum(p[1] for p in coords) / len(coords))
            lw, lh = text_size(font_label, label)
            draw.rectangle([cx - lw // 2 - 6, cy - lh // 2 - 4, cx + lw // 2 + 6, cy + lh // 2 + 4], fill=border[:-1] + (90,))
            draw.text((cx - lw // 2, cy - lh // 2), label, font=font_label, fill=TEXT)

        # Rect
        elif isinstance(coords, list) and len(coords) == 4:
            x0, y0, x1, y1 = [int(round(a)) for a in coords]
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])

            draw.rectangle([x0, y0, x1, y1], outline=border, width=3)

            lw, lh = text_size(font_label, label)
            lx = x0 + (x1 - x0 - lw) // 2
            ly = y0 + 6
            draw.rectangle([lx - 6, ly - 4, lx + lw + 8, ly + lh + 6], fill=border[:-1] + (90,))
            draw.text((lx, ly), label, font=font_label, fill=TEXT)

    pil_img.convert("RGB").save(save_path)


# ============================================================
# Drawing: dashboard stats
# ============================================================
def draw_dashboard(
    zona_names,
    clases,
    conteo_zona_clase,
    conteo_zona_total,
    conteo_global,
    out_path: str,
    highlighted_indices: set[int],
    n_cols: int = 2,
):
    """
    Genera un PNG con un panel por zona + resumen global.
    Usa OpenCV para dibujar texto y cajas (rápido y simple).
    """
    azul = (0, 162, 255)
    naranjo = (209, 123, 74)
    fondo_panel = (36, 38, 45)
    verde_oscuro = (44, 150, 80)
    rojo = (10, 60, 200)
    celeste = (192, 232, 255)
    bg_color = (28, 28, 34)
    border_color = (70, 168, 255)
    font_color = (255, 255, 255)

    n_zonas = len(zona_names)
    n_rows = math.ceil(max(n_zonas, 1) / n_cols)

    rect_w = 700
    rect_h = 85 + 56 + 56 * len(clases) + 80
    panel_gap_x = 50
    panel_gap_y = 60
    margin_x = 90
    margin_y = 100
    header_h = 90

    title_text = "Resumen estadístico"
    grid_w = n_cols * rect_w + (n_cols - 1) * panel_gap_x
    grid_h = n_rows * rect_h + (n_rows - 1) * panel_gap_y

    global_panel_h = 90 + 58 * len(clases) + 80

    dash_w = margin_x * 2 + grid_w
    dash_h = margin_y + header_h + grid_h + panel_gap_y + global_panel_h + margin_y

    dash = np.full((dash_h, dash_w, 3), bg_color, np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.19
    header_font = 2.03

    (tw, th), _ = cv2.getTextSize(title_text, font, header_font, 3)
    cv2.putText(dash, title_text.upper(), (int((dash_w - tw) // 2), margin_y + th + 14),
                font, header_font, border_color, 3, cv2.LINE_AA)

    # Paneles por zona
    for i in range(n_zonas):
        col = i % n_cols
        row = i // n_cols
        x0 = margin_x + col * (rect_w + panel_gap_x)
        y0 = margin_y + header_h + row * (rect_h + panel_gap_y)

        # Marcado => AZUL, no marcado => NARANJO (mismo comportamiento que tu script)
        if i in highlighted_indices:
            panel_border = azul
            rect_color = azul
            total_color = verde_oscuro
        else:
            panel_border = naranjo
            rect_color = naranjo
            total_color = naranjo

        cv2.rectangle(dash, (x0, y0), (x0 + rect_w, y0 + rect_h), fondo_panel, -1, cv2.LINE_AA)
        cv2.rectangle(dash, (x0, y0), (x0 + rect_w, y0 + rect_h), panel_border, 3, cv2.LINE_AA)

        zona_name = f"Zona {i+1}"
        (z_w, _), _ = cv2.getTextSize(zona_name, font, 1.25, 2)
        cv2.putText(dash, zona_name, (x0 + int((rect_w - z_w) // 2), y0 + 59),
                    font, 1.25, rect_color, 2, cv2.LINE_AA)

        y_head = y0 + 85 + 18
        (nw, _), _ = cv2.getTextSize("Cantidad", font, font_scale, 2)
        cx = x0 + 54
        qx = x0 + rect_w - nw - 24

        cv2.putText(dash, "Clase", (cx, y_head), font, font_scale, celeste, 2, cv2.LINE_AA)
        cv2.putText(dash, "Cantidad", (qx, y_head), font, font_scale, celeste, 2, cv2.LINE_AA)
        cv2.line(dash, (cx, y_head + 10), (qx + nw, y_head + 10), panel_border, 1)

        y_start = y_head + 46
        for idx_c, c in enumerate(clases):
            val = conteo_zona_clase[i][c]
            cy = y_start + idx_c * 56
            cv2.putText(dash, c, (cx, cy), font, font_scale, font_color, 2, cv2.LINE_AA)
            cv2.putText(dash, str(val), (qx, cy), font, font_scale, font_color, 2, cv2.LINE_AA)

        sep_y = y_start + len(clases) * 56 + 10
        cv2.line(dash, (cx, sep_y), (qx + nw, sep_y), panel_border, 2)

        tz = f"Total zona: {conteo_zona_total[i]}"
        (tzw, _), _ = cv2.getTextSize(tz, font, font_scale + 0.07, 2)
        tz_x = x0 + int((rect_w - tzw) // 2)
        tz_y = y0 + rect_h - 28
        cv2.putText(dash, tz, (tz_x, tz_y), font, font_scale + 0.07, total_color, 2, cv2.LINE_AA)

    # Panel global
    gp_x = margin_x
    gp_y = margin_y + header_h + grid_h + panel_gap_y + 42
    gp_w = grid_w
    gp_h = global_panel_h

    cv2.rectangle(dash, (gp_x, gp_y), (gp_x + gp_w, gp_y + gp_h), fondo_panel, -1, cv2.LINE_AA)
    cv2.rectangle(dash, (gp_x, gp_y), (gp_x + gp_w, gp_y + gp_h), rojo, 3, cv2.LINE_AA)

    (glw, _), _ = cv2.getTextSize("RESUMEN GLOBAL", font, 1.28, 2)
    cv2.putText(dash, "RESUMEN GLOBAL", (gp_x + int((gp_w - glw) // 2), gp_y + 59),
                font, 1.28, rojo, 2, cv2.LINE_AA)

    g_cx = gp_x + 74
    g_qx = gp_x + gp_w - 180
    y_g = gp_y + 115

    cv2.putText(dash, "Clase", (g_cx, y_g), font, font_scale, celeste, 2, cv2.LINE_AA)
    cv2.putText(dash, "Cantidad", (g_qx, y_g), font, font_scale, celeste, 2, cv2.LINE_AA)
    cv2.line(dash, (g_cx, y_g + 10), (g_qx + nw, y_g + 10), rojo, 1)

    y_global_start = y_g + 48
    for idx, c in enumerate(clases):
        val = conteo_global[c]
        cy = y_global_start + idx * 58
        cv2.putText(dash, c, (g_cx, cy), font, font_scale, font_color, 2, cv2.LINE_AA)
        cv2.putText(dash, str(val), (g_qx, cy), font, font_scale, font_color, 2, cv2.LINE_AA)

    y_tot = y_global_start + len(clases) * 58 + 12
    cv2.line(dash, (g_cx, y_tot), (g_qx + nw, y_tot), rojo, 2)

    total_global = sum(conteo_global[c] for c in clases)
    tg = f"Total global: {total_global}"
    (tgw, _), _ = cv2.getTextSize(tg, font, font_scale + 0.15, 2)
    cv2.putText(dash, tg, (gp_x + int((gp_w - tgw) // 2), y_tot + 44),
                font, font_scale + 0.15, celeste, 2, cv2.LINE_AA)

    cv2.imwrite(out_path, dash)


# ============================================================
# GUI
# ============================================================
class ZoneDashboardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Zone Label Dashboard")
        self.state("zoomed")
        self.config(bg="#222c38")

        # list projects (exclude meta keys)
        self.project_keys = [k for k in CONFIG.keys() if k not in ("output_dir", "logo")]
        if not self.project_keys:
            self.project_keys = ["camera_a"]

        self.project_var = StringVar(value=self.project_keys[0])
        self.color_vars: list[BooleanVar] = []
        self.zone_names: list[str] = []

        self.status_var = StringVar(value="Listo para usar.")

        self._create_widgets()
        self._load_project_config()

    def _create_widgets(self):
        frame_top = ttk.Frame(self)
        frame_top.pack(pady=16)

        ttk.Label(frame_top, text="Selecciona proyecto/cámara:", font=("Segoe UI", 16)).pack(side="left", padx=14)

        # Dropdown (más genérico que botones fijos)
        self.combo = ttk.Combobox(frame_top, textvariable=self.project_var, values=self.project_keys, state="readonly", width=30)
        self.combo.pack(side="left", padx=10)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self._on_project_change())

        self.frame_scroll = tk.Frame(self, bg="#222c38")
        self.frame_scroll.pack(pady=16, padx=16, fill="both", expand=True)

        self.canvas = tk.Canvas(self.frame_scroll, bg="#222c38", highlightthickness=0)
        self.scroll_y = tk.Scrollbar(self.frame_scroll, orient="vertical", command=self.canvas.yview)

        self.inner_frame = ttk.Frame(self.canvas)
        self.inner_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_y.pack(side="right", fill="y")

        self.label_info = ttk.Label(self, text="", background="#222c38", foreground="white")
        self.label_info.pack(pady=2)

        frame_btn = ttk.Frame(self)
        frame_btn.pack(pady=22)

        ttk.Button(frame_btn, text="Generar imágenes", command=self._generate_outputs, width=28).pack()

        self.status_lbl = ttk.Label(self, textvariable=self.status_var, background="#222c38", foreground="white", font=("Segoe UI", 12))
        self.status_lbl.pack(pady=12)

    def _on_project_change(self):
        self._load_project_config()
        self.status_var.set(f"Configuración de '{self.project_var.get()}' cargada.")

    def _load_project_config(self):
        project = self.project_var.get()
        cfg = CONFIG.get(project, {})

        yaml_path = cfg.get("yaml")
        if not yaml_path or not os.path.exists(yaml_path):
            self.zone_names = []
            self._render_zone_checks([])
            self.label_info.config(text="⚠️ YAML no configurado o no existe. Edita CONFIG.")
            return

        zonas, nombres = cargar_zonas(yaml_path)
        self.zone_names = nombres

        self._render_zone_checks(nombres)
        self.label_info.config(text=f"Total de zonas: {len(nombres)}\nMarcados = azul, desmarcados = naranjo")

    def _render_zone_checks(self, nombres):
        for w in self.inner_frame.winfo_children():
            w.destroy()

        self.color_vars = []
        for idx, _ in enumerate(nombres):
            var = BooleanVar(value=False)
            chk = ttk.Checkbutton(self.inner_frame, text=f"Zona {idx+1}", variable=var)
            chk.grid(row=idx // 8, column=idx % 8, sticky="w", padx=7, pady=5)
            self.color_vars.append(var)

    def _validate_paths(self, cfg: dict) -> tuple[bool, str]:
        required = ["images", "labels", "yaml", "image_base", "classes_txt"]
        for k in required:
            p = cfg.get(k)
            if not p:
                return False, f"Falta configurar '{k}' en CONFIG."
            if k in ("images", "labels"):
                if not os.path.isdir(p):
                    return False, f"No existe carpeta '{k}': {p}"
            else:
                if not os.path.exists(p):
                    return False, f"No existe archivo '{k}': {p}"

        out_dir = CONFIG.get("output_dir")
        if not out_dir:
            return False, "Falta 'output_dir' en CONFIG."
        os.makedirs(out_dir, exist_ok=True)
        return True, "OK"

    def _generate_outputs(self):
        project = self.project_var.get()
        cfg = CONFIG.get(project, {})

        ok, msg = self._validate_paths(cfg)
        if not ok:
            messagebox.showerror("Config inválida", msg)
            self.status_var.set("⚠️ Config inválida. Revisa CONFIG.")
            return

        zonas, zone_names = cargar_zonas(cfg["yaml"])
        clases = cargar_clases(cfg["classes_txt"])

        highlighted = {i for i, var in enumerate(self.color_vars) if var.get()}

        conteo_zona_clase, conteo_zona_total, conteo_global = contar_etiquetas_por_zona(
            cfg["labels"], zonas, clases, cfg["images"]
        )

        out_dir = CONFIG["output_dir"]
        dash_path = os.path.join(out_dir, f"{project}_dashboard.png")
        zones_img_path = os.path.join(out_dir, f"{project}_zones.png")

        draw_dashboard(
            zone_names, clases, conteo_zona_clase, conteo_zona_total, conteo_global,
            dash_path, highlighted, n_cols=2
        )
        dibujar_zonas_en_imagen(cfg["yaml"], cfg["image_base"], zones_img_path, highlighted)

        done_msg = (
            f"¡Listo!\n\n"
            f"→ Imagen con zonas: {zones_img_path}\n"
            f"→ Dashboard estadístico: {dash_path}"
        )
        self.status_var.set(done_msg)
        messagebox.showinfo("Generación exitosa", done_msg)


def main():
    out_dir = CONFIG.get("output_dir")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    app = ZoneDashboardApp()
    app.mainloop()


if __name__ == "__main__":
    main()