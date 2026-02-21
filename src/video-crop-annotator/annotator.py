import cv2
import os
import tkinter as tk
from tkinter import filedialog

# ------------------------------------------------------------
# Config (SIN rutas fijas del usuario)
# ------------------------------------------------------------
# Carpeta base donde se guardan los crops (se crea si no existe)
OUTPUT_BASE = os.path.join(os.getcwd(), "output_crops")
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Eventos mouse multiplataforma
EVENT_LBUTTONUP = getattr(cv2, "EVENT_LBUTTONUP", 4)
EVENT_LBUTTONDOWN = getattr(cv2, "EVENT_LBUTTONDOWN", 1)
EVENT_MOUSEMOVE = getattr(cv2, "EVENT_MOUSEMOVE", 0)

# UI / Estado
rect = None
frame = None
drawing = False
dragging = False
resizing = False
selected_corner = -1
offset = (0, 0, 0, 0)
corner_size = 10

local_actual = None
tipo_actual = None
opciones_tipo = ["Persona", "Trabajador", "Mechero"]
personas_actual = []
persona_idx = 0
contadores = {}

COLOR_MAIN = (255, 110, 20)
COLOR_BG = (252, 252, 255)
COLOR_FG = (33, 33, 33)
COLOR_PLUS = (230, 220, 230)
MENU_FONT = cv2.FONT_HERSHEY_DUPLEX

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def font_scale_to_fit(text, font, target_width, max_scale=0.82, min_scale=0.22):
    scale = max_scale
    (w, _), _ = cv2.getTextSize(text, font, scale, 2)
    while w > target_width and scale > min_scale:
        scale -= 0.01
        (w, _), _ = cv2.getTextSize(text, font, scale, 2)
    return max(scale, min_scale)

def get_corners(r):
    x1, y1, x2, y2 = r
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def point_near(p1, p2, d=corner_size):
    return abs(p1[0] - p2[0]) < d and abs(p1[1] - p2[1]) < d

def draw_rect(img, r):
    x1, y1, x2, y2 = map(int, r)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for (cx, cy) in get_corners(r):
        cv2.rectangle(
            img,
            (int(cx) - corner_size, int(cy) - corner_size),
            (int(cx) + corner_size, int(cy) + corner_size),
            (0, 0, 255),
            -1,
        )

# ------------------------------------------------------------
# Mouse callback (dibujar / mover / redimensionar)
# ------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global rect, drawing, dragging, resizing, selected_corner, offset

    if drawing:
        if event == EVENT_MOUSEMOVE and rect is not None:
            rect[2], rect[3] = x, y
        elif event == EVENT_LBUTTONUP and rect is not None:
            rect[2], rect[3] = x, y
            drawing = False
        return

    # No estamos dibujando: seleccionar / arrastrar / redimensionar o empezar rect nuevo
    if rect is not None:
        corners = get_corners(rect)

        if event == EVENT_LBUTTONDOWN:
            # ¿clic cerca de esquina?
            for idx, corner in enumerate(corners):
                if point_near((x, y), corner):
                    resizing = True
                    selected_corner = idx
                    return

            # ¿clic dentro del rect?
            x1, y1, x2, y2 = rect
            if min(x1, x2) < x < max(x1, x2) and min(y1, y2) < y < max(y1, y2):
                dragging = True
                offset = (x - x1, y - y1, x2 - x, y2 - y)
                return
            else:
                rect = None  # clic afuera: borrar rect

    if rect is None and event == EVENT_LBUTTONDOWN:
        drawing = True
        rect = [x, y, x, y]
        return

    # mover/redimensionar
    if rect is not None and event == EVENT_MOUSEMOVE:
        if dragging:
            dx, dy, dx2, dy2 = offset
            rect[0], rect[1], rect[2], rect[3] = x - dx, y - dy, x + dx2, y + dy2
        elif resizing and selected_corner != -1:
            if selected_corner == 0:
                rect[0], rect[1] = x, y
            elif selected_corner == 1:
                rect[2], rect[1] = x, y
            elif selected_corner == 2:
                rect[2], rect[3] = x, y
            elif selected_corner == 3:
                rect[0], rect[3] = x, y

    if event == EVENT_LBUTTONUP:
        dragging = False
        resizing = False
        selected_corner = -1

# ------------------------------------------------------------
# Selección video / locales / tipo
# ------------------------------------------------------------
def seleccionar_video():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Selecciona el video",
        filetypes=[
            ("Archivos de video", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV"),
            ("Todos los archivos", "*.*"),
        ],
    )
    root.destroy()
    return video_path

def listar_locales():
    return [d for d in os.listdir(OUTPUT_BASE) if os.path.isdir(os.path.join(OUTPUT_BASE, d))]

def seleccionar_local_interactivo(frame_):
    locales = listar_locales()
    idx_sel = 0 if locales else -1
    creando = False
    nuevo_local = ""
    h, w = frame_.shape[:2]
    menu_w, menu_h = 460, 260
    x0, y0 = (w - menu_w) // 2, (h - menu_h) // 2

    while True:
        temp = frame_.copy()
        cv2.rectangle(temp, (x0 - 5, y0 - 5), (x0 + menu_w + 5, y0 + menu_h + 5), (200, 120, 50), -1)
        overlay = temp.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + menu_w, y0 + menu_h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.97, temp, 0.03, 0, temp)
        cv2.rectangle(temp, (x0, y0), (x0 + menu_w, y0 + menu_h), COLOR_MAIN, 2)

        titulo = "Seleccionar Local"
        (titulo_w, _), _ = cv2.getTextSize(titulo, MENU_FONT, 1, 2)
        cv2.putText(temp, titulo, (x0 + (menu_w - titulo_w) // 2, y0 + 44), MENU_FONT, 1, COLOR_MAIN, 2, cv2.LINE_AA)

        for i, local in enumerate(locales):
            y_text = y0 + 80 + i * 36
            if i == idx_sel:
                cv2.rectangle(temp, (x0 + 14, y_text - 20), (x0 + menu_w - 14, y_text + 7), (255, 210, 110), -1)
            cv2.putText(temp, local, (x0 + 20, y_text), MENU_FONT, 0.7, COLOR_FG, 2, cv2.LINE_AA)

        y_nuevo = y0 + 80 + len(locales) * 36
        if idx_sel == len(locales):
            cv2.rectangle(temp, (x0 + 14, y_nuevo - 20), (x0 + menu_w - 14, y_nuevo + 7), (220, 220, 255), -1)
        cv2.putText(temp, "[Crear nuevo local]", (x0 + 20, y_nuevo), MENU_FONT, 0.7, (60, 90, 200), 2, cv2.LINE_AA)

        if creando:
            y_box = y_nuevo + 36
            box_w = 300
            box_h = 36
            x_box = x0 + menu_w // 2 - box_w // 2
            cv2.rectangle(temp, (x_box, y_box), (x_box + box_w, y_box + box_h), (220, 220, 255), -1)
            fs_nombre = font_scale_to_fit(nuevo_local + "|", MENU_FONT, box_w - 20)
            cv2.putText(temp, "Nombre: " + nuevo_local + "|", (x_box + 8, y_box + 26), MENU_FONT, fs_nombre, (33, 33, 90), 2)

        cv2.imshow("Selecciona y ajusta", temp)
        key = cv2.waitKeyEx(25)

        if creando:
            if key in [27, ord("q")]:
                return None
            elif key == 13:
                nombre = nuevo_local.strip().replace(" ", "_")
                if nombre:
                    return nombre
                creando = False
                nuevo_local = ""
            elif key in [8, 255]:
                nuevo_local = nuevo_local[:-1]
            elif key == 32:
                nuevo_local += " "
            elif 32 < key < 127:
                nuevo_local += chr(key)
            continue

        if key in [27, ord("q")]:
            return None
        if key == 13:
            if idx_sel == len(locales):
                creando = True
                nuevo_local = ""
            elif 0 <= idx_sel < len(locales):
                return locales[idx_sel]
        if key in [2490368, 0x26, 0x480000]:  # Arriba
            idx_sel = max(0, idx_sel - 1)
        if key in [2621440, 0x28, 0x500000]:  # Abajo
            idx_sel = min(len(locales), idx_sel + 1)

def pedir_tipo_persona(frame_):
    h, w = frame_.shape[:2]
    menu_w = 540
    menu_h = 160
    x0, y0 = (w - menu_w) // 2, (h - menu_h) // 2
    idx_sel = 0
    tipos = opciones_tipo
    font_size = 0.8
    espaciado = 44

    type_texts = []
    type_ws = []
    for tipo in tipos:
        (tw, _), _ = cv2.getTextSize(tipo, MENU_FONT, font_size, 2)
        type_texts.append((tipo, tw))
        type_ws.append(tw)
    total_width = sum(type_ws) + espaciado * (len(tipos) - 1)
    base_x = x0 + (menu_w - total_width) // 2

    while True:
        temp = frame_.copy()
        cv2.rectangle(temp, (x0 - 5, y0 - 5), (x0 + menu_w + 5, y0 + menu_h + 5), (200, 120, 50), -1)
        overlay = temp.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + menu_w, y0 + menu_h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.97, temp, 0.03, 0, temp)
        cv2.rectangle(temp, (x0, y0), (x0 + menu_w, y0 + menu_h), COLOR_MAIN, 2)

        titulo = "Tipo de etiqueta:"
        (titulo_w, _), _ = cv2.getTextSize(titulo, MENU_FONT, 1, 2)
        titulo_x = x0 + (menu_w - titulo_w) // 2
        cv2.putText(temp, titulo, (titulo_x, y0 + 44), MENU_FONT, 1, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.putText(temp, titulo, (titulo_x, y0 + 44), MENU_FONT, 1, COLOR_MAIN, 2, cv2.LINE_AA)

        x_text = base_x
        for i, (tipo, tw) in enumerate(type_texts):
            if i == idx_sel:
                cv2.putText(temp, tipo, (x_text, y0 + 105), MENU_FONT, font_size, (255, 255, 255), 7, cv2.LINE_AA)
                clr = COLOR_MAIN
            else:
                clr = COLOR_FG
            cv2.putText(temp, tipo, (x_text, y0 + 105), MENU_FONT, font_size, clr, 2, cv2.LINE_AA)
            x_text += tw + espaciado

        cv2.imshow("Selecciona y ajusta", temp)
        key = cv2.waitKeyEx(20)
        if key in [27, ord("q")]:
            return None
        if key == 13:
            return tipos[idx_sel]
        if key in [2490368, 0x26, 0x480000, 2424832, 0x25, 0x4B0000]:
            idx_sel = max(0, idx_sel - 1)
        if key in [2621440, 0x28, 0x500000, 2555904, 0x27, 0x4D0000]:
            idx_sel = min(len(tipos) - 1, idx_sel + 1)

def pedir_nombre_local(frame_):
    nombre = ""
    h, w = frame_.shape[:2]
    box_h = 80
    min_box_w = 250
    espacio_minimo = 54
    y0 = h - box_h - 30
    global _clicked_x_overlay
    _clicked_x_overlay = False

    def on_mouse(event, mx, my, flags, param):
        global _clicked_x_overlay
        if event == cv2.EVENT_LBUTTONDOWN:
            cur_text = nombre if nombre else "Nuevo local:"
            fs = font_scale_to_fit(cur_text, MENU_FONT, 9999, max_scale=0.95, min_scale=0.5)
            (tw, _), _ = cv2.getTextSize(cur_text, MENU_FONT, fs, 2)
            cur_box_w = max(min_box_w, tw + espacio_minimo + 50)
            x0b = w - cur_box_w - 30
            x_box0 = int(x0b + tw + espacio_minimo)
            x_box1 = x_box0 + 28
            if x_box0 <= mx <= x_box1 and y0 + 10 <= my <= y0 + 38:
                _clicked_x_overlay = True

    cv2.setMouseCallback("Selecciona y ajusta", on_mouse)

    while True:
        cur_text = nombre if nombre else "Nuevo local:"
        fs = font_scale_to_fit(cur_text, MENU_FONT, 9999, max_scale=0.95, min_scale=0.5)
        (tw, _), _ = cv2.getTextSize(cur_text, MENU_FONT, fs, 2)
        box_w = max(min_box_w, tw + espacio_minimo + 50)
        x0b = w - box_w - 30

        temp = frame_.copy()
        cv2.rectangle(temp, (x0b + 6, y0 + 6), (x0b + box_w + 6, y0 + box_h + 6), (190, 190, 190), -1)
        overlay = temp.copy()
        cv2.rectangle(overlay, (x0b, y0), (x0b + box_w, y0 + box_h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.97, temp, 0.03, 0, temp)
        cv2.rectangle(temp, (x0b, y0), (x0b + box_w, y0 + box_h), COLOR_MAIN, 2)

        x_x = int(x0b + tw + espacio_minimo + 8)
        x_box0 = int(x0b + tw + espacio_minimo)
        x_box1 = x_box0 + 28

        cv2.putText(temp, "X", (x_x, y0 + 30), MENU_FONT, 0.8, (10, 90, 200), 2)
        cv2.rectangle(temp, (x_box0, y0 + 10), (x_box1, y0 + 38), (10, 90, 200), 1)

        fs_title = font_scale_to_fit("Nuevo local:", MENU_FONT, box_w - espacio_minimo - 10)
        cv2.putText(temp, "Nuevo local:", (x0b + 13, y0 + 32), MENU_FONT, fs_title, COLOR_MAIN, 2)

        fs_nombre = font_scale_to_fit(nombre + "|", MENU_FONT, box_w - espacio_minimo - 10)
        cv2.putText(temp, nombre + "|", (x0b + 13, y0 + 65), cv2.FONT_HERSHEY_SIMPLEX, fs_nombre, COLOR_FG, 2)

        cv2.imshow("Selecciona y ajusta", temp)
        key = cv2.waitKeyEx(20)

        if _clicked_x_overlay:
            _clicked_x_overlay = False
            cv2.setMouseCallback("Selecciona y ajusta", mouse_callback)
            return None

        if key in [27, ord("q")]:
            cv2.setMouseCallback("Selecciona y ajusta", mouse_callback)
            return None
        elif key == 13:
            if nombre.strip() != "":
                cv2.setMouseCallback("Selecciona y ajusta", mouse_callback)
                return nombre.strip().replace(" ", "_")
        elif key in [8, 255]:
            nombre = nombre[:-1]
        elif key == 32:
            nombre += " "
        elif 32 < key < 127:
            nombre += chr(key)

# ------------------------------------------------------------
# Personas / miniaturas
# ------------------------------------------------------------
def listar_personas(local, tipo):
    carpeta = os.path.join(OUTPUT_BASE, local, tipo)
    if not os.path.exists(carpeta):
        return []

    personas = [d for d in os.listdir(carpeta) if os.path.isdir(os.path.join(carpeta, d))]

    def num_key(n):
        try:
            return int("".join([c for c in n if c.isdigit()]))
        except:
            return 99999

    return sorted(personas, key=num_key)

def cargar_miniatura(local, tipo, persona):
    ruta = os.path.join(OUTPUT_BASE, local, tipo, persona)
    if not os.path.exists(ruta):
        return None
    archivos = sorted([f for f in os.listdir(ruta) if f.lower().endswith(".png")])
    if archivos:
        path_img = os.path.join(ruta, archivos[0])
        img = cv2.imread(path_img)
        if img is not None:
            return cv2.resize(img, (44, 44), interpolation=cv2.INTER_AREA)
    return None

def banner_personas(frame_, tipo, personas, idx_sel):
    banner_h = 82
    w = frame_.shape[1]
    overlay = frame_.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), COLOR_MAIN, -1)
    cv2.addWeighted(overlay, 0.92, frame_, 0.08, 0, frame_)

    txt_tipo = f"{tipo}"
    cv2.putText(frame_, txt_tipo, (14, 50), MENU_FONT, 0.8, (255, 255, 255), 7, cv2.LINE_AA)
    cv2.putText(frame_, txt_tipo, (14, 50), MENU_FONT, 0.8, (60, 90, 200), 2, cv2.LINE_AA)

    base_x = 168
    btn_w = 120
    btn_h = 34
    thumb_size = 44

    for i, nombre in enumerate(personas):
        xx = base_x + i * (btn_w + thumb_size + 18)
        yy = 12

        if i == idx_sel:
            cv2.rectangle(frame_, (xx, yy), (xx + btn_w, yy + btn_h), (255, 180, 80), -1)
            cv2.rectangle(frame_, (xx, yy), (xx + btn_w, yy + btn_h), (255, 140, 40), 2)
            color_text = (60, 60, 60)
        else:
            cv2.rectangle(frame_, (xx, yy), (xx + btn_w, yy + btn_h), (255, 255, 255), -1)
            cv2.rectangle(frame_, (xx, yy), (xx + btn_w, yy + btn_h), COLOR_MAIN, 2)
            color_text = COLOR_FG

        fs_btn = font_scale_to_fit(nombre, MENU_FONT, btn_w - 12, max_scale=0.62, min_scale=0.36)
        cv2.putText(frame_, nombre, (xx + 10, yy + 25), MENU_FONT, fs_btn, color_text, 2, cv2.LINE_AA)

        thumb = cargar_miniatura(local_actual, tipo_actual, nombre)
        if thumb is not None:
            th, tw, _ = thumb.shape
            thumb_x = xx + (btn_w - tw) // 2
            thumb_y = yy + btn_h + 2
            frame_[thumb_y : thumb_y + th, thumb_x : thumb_x + tw] = thumb
            cv2.rectangle(frame_, (thumb_x, thumb_y), (thumb_x + tw, thumb_y + th), (60, 90, 200), 2)
        else:
            thumb_x = xx + (btn_w - thumb_size) // 2
            thumb_y = yy + btn_h + 2
            cv2.rectangle(frame_, (thumb_x, thumb_y), (thumb_x + thumb_size, thumb_y + thumb_size), (180, 180, 180), 2)

    # Botón "+"
    xx = base_x + len(personas) * (btn_w + thumb_size + 18)
    yy = 12
    cv2.rectangle(frame_, (xx, yy), (xx + btn_w, yy + btn_h), COLOR_PLUS, -1)
    cv2.rectangle(frame_, (xx, yy), (xx + btn_w, yy + btn_h), COLOR_MAIN, 2)
    cv2.putText(frame_, "+", (xx + 45, yy + 27), MENU_FONT, 1.1, (80, 80, 100), 2, cv2.LINE_AA)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    global rect, frame, local_actual, tipo_actual, personas_actual, persona_idx, contadores
    global drawing, dragging, resizing, selected_corner, offset

    video_path = seleccionar_video()
    if not video_path:
        print("No seleccionaste ningún video.")
        return

    cv2.namedWindow("Selecciona y ajusta", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Selecciona y ajusta", 1920, 1080)
    cv2.setMouseCallback("Selecciona y ajusta", mouse_callback)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el video.")
        return

    seleccion_inicial = False

    while True:
        if cv2.getWindowProperty("Selecciona y ajusta", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return

        display = frame.copy()

        if tipo_actual and personas_actual:
            banner_personas(display, tipo_actual, personas_actual, persona_idx)

        # Barra de tiempo y progreso
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_count > 0 and fps > 0:
            dur = frame_count / fps
            cur = pos_frame / fps

            def fmt(t):
                return f"{int(t // 60):02}:{int(t % 60):02}"

            txt_barra = f"{fmt(cur)} / {fmt(dur)}"
            barra_y = display.shape[0] - 54
            bar_w, bar_h = 320, 16
            bar_x1 = display.shape[1] - 24
            bar_x0 = bar_x1 - bar_w

            cv2.rectangle(display, (bar_x0, barra_y), (bar_x1, barra_y + bar_h), (210, 210, 230), -1)
            progreso = min(max(cur / dur, 0), 1)
            cv2.rectangle(display, (bar_x0, barra_y), (int(bar_x0 + bar_w * progreso), barra_y + bar_h), (10, 90, 200), -1)
            cv2.rectangle(display, (bar_x0, barra_y), (bar_x1, barra_y + bar_h), (70, 70, 140), 2)

            (tw, _), _ = cv2.getTextSize(txt_barra, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            time_x = bar_x1 - tw
            time_y = barra_y - 7
            cv2.putText(display, txt_barra, (time_x, time_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(display, txt_barra, (time_x, time_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 90, 200), 2, cv2.LINE_AA)

        cv2.putText(
            display,
            "[ENTER] Guardar  [ESPACIO] Sig.frame  [A] Atrás  [Q/W/E] Cambiar tipo  [L] Cambiar local  [ESC] Salir",
            (20, display.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )

        if rect is not None:
            draw_rect(display, rect)

        cv2.imshow("Selecciona y ajusta", display)
        key = cv2.waitKey(1) & 0xFF

        # Atajos tipo
        if key in [ord("q"), ord("Q")]:
            if local_actual:
                tipo_actual = "Persona"
                personas_actual = listar_personas(local_actual, tipo_actual)
                if not personas_actual:
                    personas_actual = [f"{tipo_actual}1"]
                    tipo_path = os.path.join(OUTPUT_BASE, local_actual, tipo_actual)
                    os.makedirs(os.path.join(tipo_path, personas_actual[0]), exist_ok=True)
                persona_idx = 0

        elif key in [ord("w"), ord("W")]:
            if local_actual:
                tipo_actual = "Trabajador"
                personas_actual = listar_personas(local_actual, tipo_actual)
                if not personas_actual:
                    personas_actual = [f"{tipo_actual}1"]
                    tipo_path = os.path.join(OUTPUT_BASE, local_actual, tipo_actual)
                    os.makedirs(os.path.join(tipo_path, personas_actual[0]), exist_ok=True)
                persona_idx = 0

        elif key in [ord("e"), ord("E")]:
            if local_actual:
                tipo_actual = "Mechero"
                personas_actual = listar_personas(local_actual, tipo_actual)
                if not personas_actual:
                    personas_actual = [f"{tipo_actual}1"]
                    tipo_path = os.path.join(OUTPUT_BASE, local_actual, tipo_actual)
                    os.makedirs(os.path.join(tipo_path, personas_actual[0]), exist_ok=True)
                persona_idx = 0

        # Cambiar local
        elif key in [ord("l"), ord("L")]:
            while True:
                locales = listar_locales()
                if not locales:
                    local = pedir_nombre_local(frame)
                    if local is None:
                        break
                    local_actual = local
                    os.makedirs(os.path.join(OUTPUT_BASE, local_actual), exist_ok=True)
                    break
                else:
                    sel = seleccionar_local_interactivo(frame)
                    if sel is not None:
                        local_actual = sel
                    break

            tipo_actual = pedir_tipo_persona(frame)
            if tipo_actual is None:
                print("No seleccionaste tipo.")
                continue

            tipo_path = os.path.join(OUTPUT_BASE, local_actual, tipo_actual)
            os.makedirs(tipo_path, exist_ok=True)

            personas_actual = listar_personas(local_actual, tipo_actual)
            if not personas_actual:
                personas_actual = [f"{tipo_actual}1"]
                os.makedirs(os.path.join(tipo_path, personas_actual[0]), exist_ok=True)

            persona_idx = 0
            seleccion_inicial = True

        # Guardar recorte (teclas 1..9)
        elif key in [49, 50, 51, 52, 53, 54, 55, 56, 57]:
            idx = key - 49

            if not seleccion_inicial:
                while True:
                    locales = listar_locales()
                    if not locales:
                        local = pedir_nombre_local(frame)
                        if local is None:
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                        local_actual = local
                        os.makedirs(os.path.join(OUTPUT_BASE, local_actual), exist_ok=True)
                        break
                    else:
                        sel = seleccionar_local_interactivo(frame)
                        if sel is not None:
                            local_actual = sel
                        break

                tipo_actual = pedir_tipo_persona(frame)
                if tipo_actual is None:
                    print("No seleccionaste tipo.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                tipo_path = os.path.join(OUTPUT_BASE, local_actual, tipo_actual)
                os.makedirs(tipo_path, exist_ok=True)

                personas_actual = listar_personas(local_actual, tipo_actual)
                if not personas_actual:
                    personas_actual = [f"{tipo_actual}1"]
                    os.makedirs(os.path.join(tipo_path, personas_actual[0]), exist_ok=True)

                persona_idx = 0
                seleccion_inicial = True

            # seleccionar persona por índice o crear nueva en "+"
            if idx < len(personas_actual):
                persona_idx = idx
            elif idx == len(personas_actual):
                nueva = f"{tipo_actual}{len(personas_actual) + 1}"
                personas_actual.append(nueva)
                tipo_path = os.path.join(OUTPUT_BASE, local_actual, tipo_actual)
                os.makedirs(os.path.join(tipo_path, nueva), exist_ok=True)
                persona_idx = len(personas_actual) - 1
                contadores[(local_actual, tipo_actual, nueva)] = 1

            # Guardar crop si hay rect válido
            if rect is not None and persona_idx < len(personas_actual):
                x1, y1, x2, y2 = map(int, rect)
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])

                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    nombre_per = personas_actual[persona_idx]
                    keyc = (local_actual, tipo_actual, nombre_per)
                    ruta_crop = os.path.join(OUTPUT_BASE, local_actual, tipo_actual, nombre_per)
                    os.makedirs(ruta_crop, exist_ok=True)

                    contadores.setdefault(
                        keyc,
                        len([f for f in os.listdir(ruta_crop) if f.startswith("crop_") and f.endswith(".png")]) + 1,
                    )

                    filename = os.path.join(ruta_crop, f"crop_{contadores.get(keyc, 1):04d}.png")
                    crop = frame[y1:y2, x1:x2].copy()
                    cv2.imwrite(filename, crop)
                    print(f"Guardado: {filename}")

                    contadores[keyc] = contadores.get(keyc, 1) + 1
                    rect = None

        # (Opcional) Cambiar tipo con T (mantengo tu lógica)
        elif key in [ord("t"), ord("T")]:
            if not local_actual:
                continue
            tipo_nuevo = pedir_tipo_persona(frame)
            if tipo_nuevo is None:
                continue
            tipo_actual = tipo_nuevo
            tipo_path = os.path.join(OUTPUT_BASE, local_actual, tipo_actual)
            os.makedirs(tipo_path, exist_ok=True)
            personas_actual = listar_personas(local_actual, tipo_actual)
            if not personas_actual:
                personas_actual = [f"{tipo_actual}1"]
                os.makedirs(os.path.join(tipo_path, personas_actual[0]), exist_ok=True)
            persona_idx = 0

        # Siguiente frame
        elif key == 32:
            ret, next_frame = cap.read()
            if not ret:
                print("Fin del video.")
                cv2.destroyWindow("Selecciona y ajusta")
                import tkinter.messagebox as msg

                root = tk.Tk()
                root.withdraw()
                res = msg.askyesno("Video finalizado", "¿Deseas cargar un nuevo video?")
                root.destroy()
                if res:
                    cap.release()
                    cv2.destroyAllWindows()
                    main()
                else:
                    cap.release()
                    cv2.destroyAllWindows()
                return

            frame = next_frame
            rect = None
            drawing = False
            dragging = False
            resizing = False
            selected_corner = -1

        # Frame anterior
        elif key in [ord("a"), ord("A")]:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos - 2)
                ret, prev_frame = cap.read()
                if ret:
                    frame = prev_frame
                    rect = None
                    drawing = False
                    dragging = False
                    resizing = False
                    selected_corner = -1
                else:
                    print("No se pudo retroceder el frame.")
            else:
                print("Ya estás en el primer frame.")

        # Salir
        elif key == 27:
            break

        # Cambiar persona con flechas (si hay)
        elif key in [2555904, 0x27, 0x4D0000]:
            if personas_actual:
                persona_idx = (persona_idx + 1) % len(personas_actual)
        elif key in [2424832, 0x25, 0x4B0000]:
            if personas_actual:
                persona_idx = (persona_idx - 1) % len(personas_actual)

        # Crear nueva persona con N (mantengo tu lógica)
        elif key in [ord("n"), ord("N")]:
            if tipo_actual and local_actual:
                nueva = f"{tipo_actual}{len(personas_actual) + 1}"
                personas_actual.append(nueva)
                tipo_path = os.path.join(OUTPUT_BASE, local_actual, tipo_actual)
                os.makedirs(os.path.join(tipo_path, nueva), exist_ok=True)
                persona_idx = len(personas_actual) - 1
                contadores[(local_actual, tipo_actual, nueva)] = 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()