# -*- coding: utf-8 -*-
import os
import re
import math
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import yaml

# ===================== CONFIG (GENÉRICO / PORTABLE) =====================
def _default_dir(p: str) -> str:
    """Create folder if possible and return absolute path; fallback to cwd."""
    try:
        Path(p).mkdir(parents=True, exist_ok=True)
        return str(Path(p).resolve())
    except Exception:
        return str(Path(".").resolve())

# Defaults safe for any user/repo
DIR_IMG_DEFECTO  = _default_dir("./examples/images")
DIR_YAML_DEFECTO = _default_dir("./examples/yamls")

def parse_args():
    ap = argparse.ArgumentParser(description="ROI YAML Annotator (polygons/rectangles/circles)")
    ap.add_argument("--img-dir", default=DIR_IMG_DEFECTO, help="Default image folder")
    ap.add_argument("--yaml-dir", default=DIR_YAML_DEFECTO, help="Default yaml folder")
    return ap.parse_args()

args = parse_args()
DIR_IMG_DEFECTO  = args.img_dir
DIR_YAML_DEFECTO = args.yaml_dir

COLORES_LISTA = ["Celeste", "Verde", "Rojo", "Amarillo", "Naranja", "Magenta", "Cian", "Blanco"]
COLORES_HEX = {
    "Celeste": "#3ec7eb", "Verde": "#0bc759", "Rojo": "#fc4a50", "Amarillo": "#fcf440",
    "Naranja": "#ffb347", "Magenta": "#f252c6", "Cian": "#5de2e6", "Blanco": "#FFFFFF"
}
get_color = lambda c: COLORES_HEX.get(c, c)

# Vértices del polígono
POLY_POINT_RADIUS = 6
POLY_LINE_WIDTH   = 2
HIT_TOLERANCE     = 10

# ===================== ESTADO =====================
ultima_dir_imagen = DIR_IMG_DEFECTO
ultima_dir_yaml   = DIR_YAML_DEFECTO
ruta_yaml         = None

root = tk.Tk()
root.title("ROI YAML Annotator")

# Polígonos
polygons, polygons_disp = {}, {}
id_points, id_lines, id_polygons = {}, {}, {}
polygons_backup = {}

# Rects
rectangles, id_rectangles = {}, {}
rect_start = None
drawing_rect = rect_dragging = rect_resizing = False
rect_corner = None
rect_selected = None
rect_offset = (0, 0)
rectangles_backup = {}

# Circles
circles, id_circles = {}, {}
circle_start = None
drawing_circle = circle_dragging = circle_resizing = False
circle_selected = None
circle_offset = (0, 0)
circles_backup = {}

# Modo / imagen
img_tk = None
img_width = img_height = None
pil_img_global = None
undo_stack = []
yaml_ya_guardado = False

mode = tk.StringVar(value="")
rect_var = tk.StringVar()
poly_var = tk.StringVar()
circle_var = tk.StringVar()
color_seleccionado = tk.StringVar(value="Celeste")

# ===================== HELPERS =====================
def next_available(prefix, d):
    i = 1
    while f"{prefix}_{i}" in d:
        i += 1
    return f"{prefix}_{i}"

def sort_keys_numeric(d, prefix):
    def keyfn(k):
        try:
            return int(k.split("_")[1])
        except:
            return 10**9
    return sorted([k for k in d.keys() if k.startswith(prefix+"_")], key=keyfn)

def hay_dibujos_en_pantalla():
    return bool(polygons) or bool(rectangles) or bool(circles)

def guardar_estado_actual():
    global polygons_backup, rectangles_backup, circles_backup
    polygons_backup   = polygons.copy()
    rectangles_backup = rectangles.copy()
    circles_backup    = circles.copy()

# ===== centroid helpers (para ubicar etiqueta dentro del polígono) =====
def point_in_polygon(x, y, pts):
    inside = False
    n = len(pts)
    for i in range(n):
        x1,y1 = pts[i]
        x2,y2 = pts[(i+1)%n]
        if ((y1>y)!=(y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-9)+x1):
            inside = not inside
    return inside

def polygon_centroid(pts):
    A = 0.0; cx = 0.0; cy = 0.0
    n = len(pts)
    for i in range(n):
        x0,y0 = pts[i]; x1,y1 = pts[(i+1)%n]
        cross = x0*y1 - x1*y0
        A += cross; cx += (x0+x1)*cross; cy += (y0+y1)*cross
    if abs(A) < 1e-7:
        sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
        return sx/n, sy/n
    A *= 0.5; cx /= (6.0*A); cy /= (6.0*A); return cx, cy

def poly_label_point(pts):
    cx, cy = polygon_centroid(pts)
    if point_in_polygon(cx, cy, pts):
        return int(cx), int(cy)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    bx, by = (min(xs)+max(xs))//2, (min(ys)+max(ys))//2
    if point_in_polygon(bx, by, pts):
        return bx, by
    x0,y0 = pts[0]
    for t in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:
        tx = x0 + (cx-x0)*t; ty = y0 + (cy-y0)*t
        if point_in_polygon(tx, ty, pts): return int(tx), int(ty)
    return bx, by

def hit_polygon_vertex(x, y):
    radius = POLY_POINT_RADIUS + 6
    for key, pts in polygons_disp.items():
        for i, (px, py) in enumerate(pts):
            if math.hypot(x - px, y - py) <= radius:
                return key, i
    return None, None

# ===================== IMAGEN =====================
def openfile():
    global img_tk, img_width, img_height, ultima_dir_imagen, pil_img_global
    fname = filedialog.askopenfilename(
        initialdir=ultima_dir_imagen, title="Open image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not fname: return
    ultima_dir_imagen = os.path.dirname(fname)
    pil_img_global = Image.open(fname).convert("RGB")
    img_width, img_height = pil_img_global.width, pil_img_global.height
    C.config(width=img_width, height=img_height)
    img_tk = ImageTk.PhotoImage(pil_img_global)
    C.delete("all")
    C.create_image(0, 0, anchor="nw", image=img_tk)
    C.tag_lower("all")
    redraw_all_labels()

# ===================== YAML =====================
def save_yaml(auto=False):
    global ruta_yaml, ultima_dir_yaml, yaml_ya_guardado
    hay_poligonos   = any(polygons.get(k) and polygons[k]['pts'] for k in polygons)
    hay_rectangulos = any(rectangles.get(k) and len(rectangles[k].get('coords',[]))==4 for k in rectangles)
    hay_circulos    = any(circles.get(k) and "center" in circles[k] and "radius" in circles[k] for k in circles)
    if not (hay_poligonos or hay_rectangulos or hay_circulos):
        return

    data = {
        'polygons':   {k:{'pts':polygons[k]['pts'], 'color':polygons[k]['color']} for k in polygons},
        'rectangles': {k:{'coords':v['coords'], 'color':v['color']} for k,v in rectangles.items() if v.get('coords')},
        'circles':    {k:{'center':v['center'], 'radius':v['radius'], 'color':v['color']} for k,v in circles.items() if "center" in v and "radius" in v}
    }

    if ruta_yaml is None and not yaml_ya_guardado:
        f = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML","*.yaml")],
            initialdir=ultima_dir_yaml
        )
        if not f: return
        if not f.lower().endswith('.yaml'): f += ".yaml"
        ruta_yaml = f
        ultima_dir_yaml = os.path.dirname(f)
        yaml_ya_guardado = True
    elif ruta_yaml is None:
        return

    with open(ruta_yaml, "w", encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True)

def load_yaml():
    global ruta_yaml, ultima_dir_yaml, yaml_ya_guardado
    if hay_dibujos_en_pantalla():
        guardar_estado_actual()
    clear_all(auto=False)

    f = filedialog.askopenfilename(filetypes=[("YAML","*.yaml")], initialdir=ultima_dir_yaml)
    if not f: return
    ruta_yaml = f
    ultima_dir_yaml = os.path.dirname(f)
    yaml_ya_guardado = True

    with open(f, encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    # Polígonos
    for name, item in data.get('polygons', {}).items():
        polygons[name] = {"pts": item["pts"], "color": item.get("color", "Celeste")}
        polygons_disp[name] = []
        id_points[name] = []
        id_lines[name] = []
        id_polygons[name] = None
        for (x,y) in item["pts"]:
            px, py = int(round(x)), int(round(y))
            polygons_disp[name].append((px,py))
            p_id = C.create_oval(px-POLY_POINT_RADIUS, py-POLY_POINT_RADIUS,
                                 px+POLY_POINT_RADIUS, py+POLY_POINT_RADIUS,
                                 fill="#2196f3", outline="white", width=2)
            id_points[name].append(p_id)
        n = len(polygons_disp[name])
        for i in range(n-1):
            idl = C.create_line(polygons_disp[name][i], polygons_disp[name][i+1],
                                width=POLY_LINE_WIDTH, fill=get_color(item.get("color","Celeste")))
            id_lines[name].append(idl)
        if n>2:
            idl = C.create_line(polygons_disp[name][0], polygons_disp[name][-1],
                                width=POLY_LINE_WIDTH, fill=get_color(item.get("color","Celeste")))
            id_lines[name].append(idl)
            id_polygons[name] = C.create_polygon(*[xy for t in polygons_disp[name] for xy in t],
                                                 fill=get_color(item.get("color","Celeste")),
                                                 outline=get_color(item.get("color","Celeste")), width=2, stipple='gray25')
        draw_poly_label(name)

    # Rectángulos
    for name,item in data.get('rectangles',{}).items():
        if "coords" in item and len(item["coords"])==4:
            rectangles[name] = {"coords": item["coords"], "color": item.get("color","Celeste")}
            id_rectangles[name] = C.create_rectangle(*item["coords"], outline=get_color(item.get("color","Celeste")), width=2)
            draw_rect_label(name)

    # Círculos
    for name,item in data.get('circles',{}).items():
        if "center" in item and "radius" in item:
            circles[name] = {"center": item["center"], "radius": item["radius"], "color": item.get("color","Celeste")}
            draw_circle(name)

    rebuild_radios_all()
    save_yaml(auto=True)

def copiar_ruta_yaml():
    if ruta_yaml:
        root.clipboard_clear()
        root.clipboard_append(ruta_yaml)
        root.update()
        messagebox.showinfo("YAML path copied", f"YAML path:\n{ruta_yaml}")
    else:
        messagebox.showwarning("No YAML","No YAML selected/saved yet.")

def clear_all(auto=True):
    global yaml_ya_guardado, ruta_yaml

    # Polígonos
    for idl in id_lines.values():
        for i in idl: C.delete(i)
    for idp in id_points.values():
        for i in idp: C.delete(i)
    for k,pid in list(id_polygons.items()):
        if isinstance(pid,int): C.delete(pid)
        if id_polygons.get(k+"_label"):
            C.delete(id_polygons[k+"_label"])
            id_polygons.pop(k+"_label", None)
    id_lines.clear(); id_points.clear(); polygons.clear(); polygons_disp.clear(); id_polygons.clear()

    # Rects
    for i in id_rectangles.values():
        if isinstance(i,int): C.delete(i)
    for k in [k for k in list(id_rectangles.keys()) if k.endswith("_label")]:
        C.delete(id_rectangles[k]); id_rectangles.pop(k)
    rectangles.clear(); id_rectangles.clear()

    # Círculos
    for i in id_circles.values():
        if isinstance(i,int): C.delete(i)
    for k in [k for k in list(id_circles.keys()) if k.endswith("_label")]:
        C.delete(id_circles[k]); id_circles.pop(k)
    circles.clear(); id_circles.clear()

    if auto: save_yaml(auto=True)
    yaml_ya_guardado = False
    ruta_yaml = None
    rebuild_radios_all()

# ===================== MODOS =====================
def set_mode_poly():
    mode.set("poly")
    frame_radios_rect.pack_forget()
    frame_radios_circle.pack_forget()
    frame_radios_poly.pack(fill="y")

def set_mode_rect():
    mode.set("rect")
    frame_radios_poly.pack_forget()
    frame_radios_circle.pack_forget()
    frame_radios_rect.pack(fill="y")

def set_mode_circle():
    mode.set("circle")
    frame_radios_poly.pack_forget()
    frame_radios_rect.pack_forget()
    frame_radios_circle.pack(fill="y")

# ===================== POLÍGONOS =====================
poly_mode = False
poly_actual = None
poly_temp_points = []

def start_poly():
    global poly_mode, poly_actual, poly_temp_points
    poly_actual = poly_var.get()
    if poly_actual in polygons_disp and len(polygons_disp[poly_actual])>2:
        poly_mode = False
        poly_temp_points = []
    else:
        poly_mode = True
        poly_temp_points = polygons_disp.get(poly_actual, []).copy()

def close_poly():
    global poly_mode, poly_actual, poly_temp_points
    if len(poly_temp_points)>2 and poly_actual:
        color = color_seleccionado.get()
        real_pts = [[int(round(px)), int(round(py))] for px,py in poly_temp_points]
        polygons[poly_actual] = {"pts": real_pts, "color": color}

        if id_polygons.get(poly_actual): C.delete(id_polygons[poly_actual])
        if id_polygons.get(poly_actual+"_label"):
            C.delete(id_polygons[poly_actual+"_label"]); id_polygons.pop(poly_actual+"_label")

        for pid in id_points.get(poly_actual, []): C.delete(pid)
        for lid in id_lines.get(poly_actual, []): C.delete(lid)
        id_points[poly_actual] = []; id_lines[poly_actual] = []

        n = len(poly_temp_points)
        for i,(px,py) in enumerate(poly_temp_points):
            p_id = C.create_oval(px-POLY_POINT_RADIUS, py-POLY_POINT_RADIUS,
                                 px+POLY_POINT_RADIUS, py+POLY_POINT_RADIUS,
                                 fill="#2196f3", outline="white", width=2)
            id_points[poly_actual].append(p_id)
            if i>0:
                idl = C.create_line(poly_temp_points[i-1], (px,py), width=POLY_LINE_WIDTH, fill=get_color(color))
                id_lines[poly_actual].append(idl)
        if n>2:
            idl = C.create_line(poly_temp_points[0], poly_temp_points[-1], width=POLY_LINE_WIDTH, fill=get_color(color))
            id_lines[poly_actual].append(idl)
            id_polygons[poly_actual] = C.create_polygon(
                *[xy for t in poly_temp_points for xy in t],
                fill=get_color(color), outline=get_color(color), width=2, stipple='gray25'
            )
        polygons_disp[poly_actual] = poly_temp_points.copy()
        draw_poly_label(poly_actual)
        poly_mode = False
        poly_temp_points = []
        rebuild_poly_radios()
        save_yaml(auto=True)

def undo_poly_func(event=None):
    global poly_temp_points, poly_mode, poly_actual
    if mode.get() != "poly": return
    if poly_mode and poly_temp_points:
        poly_temp_points.pop()
        redraw_temp_poly()
        if not poly_temp_points:
            poly_mode = True
            poly_actual = poly_var.get()
    else:
        key = poly_var.get()
        if key in polygons:
            pts = polygons[key]['pts']
            if len(pts)>1:
                pts.pop()
                polygons_disp[key].pop()
                redraw_open_poly(key, polygons_disp[key], polygons[key]['color'])
                poly_mode = True
                poly_actual = key
                poly_temp_points = polygons_disp[key].copy()
                save_yaml(auto=True)
            else:
                delete_polygon(key)

def delete_polygon(key):
    if key in polygons:
        if id_polygons.get(key): C.delete(id_polygons[key]); id_polygons[key]=None
        if id_polygons.get(key+"_label"): C.delete(id_polygons[key+"_label"]); id_polygons.pop(key+"_label", None)
        for pid in id_points.get(key, []): C.delete(pid)
        for lid in id_lines.get(key, []): C.delete(lid)
        polygons.pop(key, None)
        polygons_disp.pop(key, None)
        id_points.pop(key, None)
        id_lines.pop(key, None)
        id_polygons.pop(key, None)
        rebuild_poly_radios()
        save_yaml(auto=True)

def redraw_temp_poly():
    key = poly_actual
    for pid in id_points.get(key, []): C.delete(pid)
    for lid in id_lines.get(key, []): C.delete(lid)
    id_points[key] = []; id_lines[key] = []
    for i,(px,py) in enumerate(poly_temp_points):
        p_id = C.create_oval(px-POLY_POINT_RADIUS, py-POLY_POINT_RADIUS,
                             px+POLY_POINT_RADIUS, py+POLY_POINT_RADIUS,
                             fill="#2196f3", outline="white", width=2)
        id_points[key].append(p_id)
        if i>0:
            idl = C.create_line(poly_temp_points[i-1], (px,py), width=POLY_LINE_WIDTH, fill=get_color(color_seleccionado.get()))
            id_lines[key].append(idl)
    if id_polygons.get(key+"_label"):
        C.delete(id_polygons[key+"_label"]); id_polygons.pop(key+"_label", None)
    if poly_temp_points:
        draw_poly_label(key, poly_temp_points)

def redraw_open_poly(key, pts_disp, color):
    for pid in id_points.get(key, []): C.delete(pid)
    for lid in id_lines.get(key, []): C.delete(lid)
    id_points[key] = []; id_lines[key] = []
    if id_polygons.get(key): C.delete(id_polygons[key]); id_polygons[key] = None
    if id_polygons.get(key+"_label"):
        C.delete(id_polygons[key+"_label"]); id_polygons.pop(key+"_label", None)
    for i,(px,py) in enumerate(pts_disp):
        p_id = C.create_oval(px-POLY_POINT_RADIUS, py-POLY_POINT_RADIUS,
                             px+POLY_POINT_RADIUS, py+POLY_POINT_RADIUS,
                             fill="#2196f3", outline="white", width=2)
        id_points[key].append(p_id)
        if i>0:
            idl = C.create_line(pts_disp[i-1], (px,py), width=POLY_LINE_WIDTH, fill=get_color(color))
            id_lines[key].append(idl)
    if pts_disp:
        draw_poly_label(key, pts_disp)

def draw_poly_label(key, pts=None):
    if pts is None:
        pts = polygons_disp.get(key, [])
    if pts:
        lx, ly = poly_label_point(pts)
        label = key
        if id_polygons.get(key+"_label"):
            C.delete(id_polygons[key+"_label"])
        id_polygons[key+"_label"] = C.create_text(lx, ly, text=label, fill="yellow", font=("Arial", 16, "bold"))

def add_polygon_named():
    nmax = 0
    for k in polygons.keys():
        if k.lower().startswith("caja "):
            try: nmax = max(nmax, int(k.split(" ",1)[1]))
            except: pass
    sugerido = f"Caja {nmax+1}" if nmax>=0 else "Nuevo"
    name = simpledialog.askstring("New polygon", "Name:", initialvalue=sugerido, parent=root)
    if not name: return
    if name in polygons:
        messagebox.showerror("Duplicate name", f"'{name}' already exists.")
        return
    polygons[name] = {"pts": [], "color": color_seleccionado.get()}
    polygons_disp[name] = []
    id_points[name] = []; id_lines[name] = []; id_polygons[name] = None
    rebuild_poly_radios()
    poly_var.set(name); set_mode_poly()
    global poly_mode, poly_actual, poly_temp_points
    poly_mode = True; poly_actual = name; poly_temp_points = []

# ===================== RECTÁNGULOS =====================
def draw_rect_label(key):
    if key not in rectangles: return
    coords = rectangles[key].get("coords",[])
    if len(coords)==4:
        x0,y0,x1,y1 = coords
        cx,cy = (x0+x1)//2, (y0+y1)//2
        label = key.replace("rect_","")
        if id_rectangles.get(key+"_label"):
            C.delete(id_rectangles[key+"_label"])
        id_rectangles[key+"_label"] = C.create_text(cx, cy, text=label, fill="yellow", font=("Arial",16,"bold"))
    else:
        if id_rectangles.get(key+"_label"):
            C.delete(id_rectangles[key+"_label"]); id_rectangles.pop(key+"_label", None)

def set_color_rect(event=None):
    if mode.get()=="rect":
        key = rect_var.get()
        if key and key in rectangles:
            rectangles[key]["color"] = color_seleccionado.get()
            if id_rectangles.get(key):
                C.itemconfig(id_rectangles[key], outline=get_color(color_seleccionado.get()))
            save_yaml(auto=True)
    elif mode.get()=="poly":
        key = poly_var.get()
        if key and key in polygons:
            polygons[key]["color"] = color_seleccionado.get()
            save_yaml(auto=True)
    elif mode.get()=="circle":
        key = circle_var.get()
        if key and key in circles:
            circles[key]["color"] = color_seleccionado.get()
            draw_circle(key); save_yaml(auto=True)

def add_rectangle():
    name = next_available("rect", rectangles)
    rectangles[name] = {"coords": [], "color": color_seleccionado.get()}
    rebuild_rect_radios()
    rect_var.set(name); set_mode_rect()

# ===================== CÍRCULOS =====================
def draw_circle(key):
    if key not in circles or "center" not in circles[key] or "radius" not in circles[key]:
        return
    cx, cy = circles[key]["center"]
    r  = circles[key]["radius"]
    color = circles[key].get("color","Celeste")
    if id_circles.get(key): C.delete(id_circles[key])
    if id_circles.get(key+"_label"): C.delete(id_circles[key+"_label"])
    id_circles[key] = C.create_oval(cx-r, cy-r, cx+r, cy+r, outline=get_color(color), width=3)
    label = key.replace("circle_","")
    id_circles[key+"_label"] = C.create_text(cx, cy, text=label, fill="yellow", font=("Arial",16,"bold"))

def add_circle():
    name = next_available("circle", circles)
    circles[name] = {}
    rebuild_circle_radios()
    circle_var.set(name); set_mode_circle()

# ===================== PICK / CONTEXT =====================
def pick_shape_at(x,y):
    # círculos
    for k in sort_keys_numeric(circles, "circle"):
        c = circles[k]
        if "center" in c and "radius" in c:
            cx,cy = c["center"]; r=c["radius"]
            if math.hypot(x-cx, y-cy) <= r + HIT_TOLERANCE:
                return ("circle", k)
    # rects
    for k in sort_keys_numeric(rectangles, "rect"):
        coords = rectangles[k].get("coords",[])
        if len(coords)==4:
            x0,y0,x1,y1 = coords
            if min(x0,x1)-HIT_TOLERANCE <= x <= max(x0,x1)+HIT_TOLERANCE and \
               min(y0,y1)-HIT_TOLERANCE <= y <= max(y0,y1)+HIT_TOLERANCE:
                return ("rect", k)
    # polígonos
    for k,disp in polygons_disp.items():
        if len(disp)>2 and point_in_polygon(x,y,disp):
            return ("poly", k)
    return (None, None)

menu_ctx = tk.Menu(root, tearoff=0)
_ctx_target = {"type":None, "key":None}

def ctx_rename_poly():
    key = _ctx_target.get("key")
    if not key or key not in polygons:
        return
    nuevo = simpledialog.askstring("Rename polygon", "New name:", initialvalue=key, parent=root)
    if not nuevo or nuevo == key:
        return
    if nuevo in polygons:
        messagebox.showerror("Name already used", f"'{nuevo}' already exists.")
        return
    polygons[nuevo]      = polygons.pop(key)
    polygons_disp[nuevo] = polygons_disp.pop(key)
    id_points[nuevo]     = id_points.pop(key, [])
    id_lines[nuevo]      = id_lines.pop(key, [])
    if key in id_polygons:
        id_polygons[nuevo] = id_polygons.pop(key)
    if id_polygons.get(key+"_label"):
        C.delete(id_polygons[key+"_label"])
        id_polygons.pop(key+"_label", None)
    draw_poly_label(nuevo)
    if poly_var.get() == key:
        poly_var.set(nuevo)
    rebuild_poly_radios(); save_yaml(auto=True)

def delete_vertex_and_split(key, vi):
    if key not in polygons_disp:
        return
    pts = polygons_disp[key][:]
    if len(pts) <= 3:
        delete_polygon(key)
        return
    pts.pop(vi)
    start = vi % len(pts)
    open_pts = pts[start:] + pts[:start]
    polygons[key]['pts'] = [[int(x), int(y)] for (x, y) in open_pts]
    polygons_disp[key]   = open_pts
    redraw_open_poly(key, open_pts, polygons[key]['color'])
    global poly_mode, poly_actual, poly_temp_points
    poly_var.set(key)
    set_mode_poly()
    poly_mode = True
    poly_actual = key
    poly_temp_points = open_pts[:]
    save_yaml(auto=True)

def on_right_click(event):
    vk, vi = hit_polygon_vertex(event.x, event.y)
    menu_ctx.delete(0, "end")
    if vk is not None:
        _ctx_target["type"] = "poly_vertex"
        _ctx_target["key"]  = vk
        _ctx_target["index"] = vi
        menu_ctx.add_command(
            label=f"Delete vertex {vi+1} of {vk} (split)",
            command=lambda: delete_vertex_and_split(_ctx_target["key"], _ctx_target["index"])
        )
        menu_ctx.tk_popup(event.x_root, event.y_root)
        return

    t, k = pick_shape_at(event.x, event.y)
    if not t:
        return
    _ctx_target.clear(); _ctx_target["type"]=t; _ctx_target["key"]=k

    if t == "poly":
        menu_ctx.add_command(label=f"Duplicate {k}", command=lambda: duplicate_selected(target=_ctx_target))
        menu_ctx.add_command(label=f"Rename {k}", command=ctx_rename_poly)
        menu_ctx.add_command(label=f"Delete {k}", command=lambda: borrar_objeto_seleccionado(target=_ctx_target))
    elif t == "rect":
        menu_ctx.add_command(label=f"Duplicate {k}", command=lambda: duplicate_selected(target=_ctx_target))
        menu_ctx.add_command(label=f"Delete {k}", command=lambda: borrar_objeto_seleccionado(target=_ctx_target))
    elif t == "circle":
        menu_ctx.add_command(label=f"Duplicate {k}", command=lambda: duplicate_selected(target=_ctx_target))
        menu_ctx.add_command(label=f"Delete {k}", command=lambda: borrar_objeto_seleccionado(target=_ctx_target))

    menu_ctx.tk_popup(event.x_root, event.y_root)

# ===================== CANVAS: CLICK/DRAG =====================
def canvas_click(event):
    global poly_temp_points, poly_mode, poly_actual
    global rect_start, drawing_rect, rect_selected, rect_dragging, rect_resizing, rect_corner, rect_offset
    global circle_start, drawing_circle, circle_selected, circle_dragging, circle_resizing, circle_offset

    if not poly_mode:
        t,k = pick_shape_at(event.x, event.y)
        if t=="circle":
            circle_var.set(k); set_mode_circle()
        elif t=="rect":
            rect_var.set(k); set_mode_rect()
        elif t=="poly":
            poly_var.set(k); set_mode_poly()
            return

    if mode.get()=="poly":
        if not poly_mode or not poly_actual:
            return
        poly_temp_points.append((event.x, event.y))
        redraw_temp_poly()
        if len(poly_temp_points)>2:
            x0,y0 = poly_temp_points[0]
            xn,yn = poly_temp_points[-1]
            if math.hypot(x0-xn, y0-yn) < 10:
                close_poly()
        return

    elif mode.get()=="rect":
        key = rect_var.get()
        if key and key in rectangles:
            rect = rectangles[key].get("coords",[])
            if not rect or len(rect)!=4:
                rect_start = (event.x, event.y)
                drawing_rect = True
                return
            x0,y0,x1,y1 = rect
            tol = HIT_TOLERANCE
            if abs(event.x - x0) < tol and abs(event.y - y0) < tol:
                rect_corner = 'tl'; rect_resizing = True; rect_selected = key
            elif abs(event.x - x1) < tol and abs(event.y - y0) < tol:
                rect_corner = 'tr'; rect_resizing = True; rect_selected = key
            elif abs(event.x - x0) < tol and abs(event.y - y1) < tol:
                rect_corner = 'bl'; rect_resizing = True; rect_selected = key
            elif abs(event.x - x1) < tol and abs(event.y - y1) < tol:
                rect_corner = 'br'; rect_resizing = True; rect_selected = key
            elif min(x0,x1) < event.x < max(x0,x1) and min(y0,y1) < event.y < max(y0,y1):
                rect_dragging = True; rect_offset = (event.x - x0, event.y - y0); rect_selected = key
        return

    elif mode.get()=="circle":
        key = circle_var.get()
        if key and key in circles:
            circ = circles[key]
            if "center" not in circ or "radius" not in circ:
                circle_start = (event.x, event.y); drawing_circle = True; return
            cx,cy = circ["center"]; r = circ["radius"]; tol = HIT_TOLERANCE
            dist = math.hypot(event.x - cx, event.y - cy)
            if abs(dist - r) < tol:
                circle_resizing = True; circle_selected = key
            elif dist < r:
                circle_dragging = True; circle_offset = (event.x - cx, event.y - cy); circle_selected = key

def canvas_drag(event):
    global drawing_rect, rect_start, rect_selected, rect_dragging, rect_resizing, rect_corner, rect_offset
    global drawing_circle, circle_start, circle_selected, circle_dragging, circle_resizing, circle_offset

    if mode.get()=="rect":
        key = rect_var.get()
        if key and key in rectangles:
            if drawing_rect and rect_start:
                x0,y0 = rect_start; x1,y1 = event.x, event.y
                rectangles[key]["coords"] = [x0,y0,x1,y1]
                color = rectangles[key]["color"]
                if id_rectangles.get(key): C.delete(id_rectangles[key])
                if id_rectangles.get(key + "_label"):
                    C.delete(id_rectangles[key + "_label"]); id_rectangles.pop(key + "_label", None)
                id_rectangles[key] = C.create_rectangle(x0,y0,x1,y1, outline=get_color(color), width=2)
                draw_rect_label(key)
            elif rect_dragging and rect_selected == key:
                x0,y0,x1,y1 = rectangles[key]["coords"]; w,h = x1-x0, y1-y0
                nx0,ny0 = event.x - rect_offset[0], event.y - rect_offset[1]
                nx1,ny1 = nx0 + w, ny0 + h
                rectangles[key]["coords"] = [nx0,ny0,nx1,ny1]
                C.coords(id_rectangles[key], nx0,ny0,nx1,ny1)
                if id_rectangles.get(key + "_label"):
                    C.delete(id_rectangles[key + "_label"]); id_rectangles.pop(key + "_label", None)
                draw_rect_label(key)
            elif rect_resizing and rect_selected == key:
                if rect_corner == 'tl':
                    rectangles[key]["coords"][0] = event.x
                    rectangles[key]["coords"][1] = event.y
                elif rect_corner == 'tr':
                    rectangles[key]["coords"][2] = event.x
                    rectangles[key]["coords"][1] = event.y
                elif rect_corner == 'bl':
                    rectangles[key]["coords"][0] = event.x
                    rectangles[key]["coords"][3] = event.y
                elif rect_corner == 'br':
                    rectangles[key]["coords"][2] = event.x
                    rectangles[key]["coords"][3] = event.y
                C.coords(id_rectangles[key], *rectangles[key]["coords"])
                if id_rectangles.get(key + "_label"):
                    C.delete(id_rectangles[key + "_label"]); id_rectangles.pop(key + "_label", None)
                draw_rect_label(key)

    elif mode.get()=="circle":
        key = circle_var.get()
        if key and key in circles:
            circ = circles[key]
            if drawing_circle and circle_start:
                cx0,cy0 = circle_start
                radius = int(math.hypot(event.x - cx0, event.y - cy0))
                circ["center"] = [cx0,cy0]; circ["radius"] = radius; circ["color"] = color_seleccionado.get()
                draw_circle(key)
            elif circle_dragging and circle_selected == key:
                dx,dy = event.x - circle_offset[0], event.y - circle_offset[1]
                circ["center"] = [dx,dy]; draw_circle(key)
            elif circle_resizing and circle_selected == key:
                cx,cy = circ["center"]
                new_r = int(math.hypot(event.x - cx, event.y - cy))
                circ["radius"] = new_r; draw_circle(key)

def canvas_release(event):
    global drawing_rect, rect_start, rect_dragging, rect_resizing, rect_corner, rect_selected
    global drawing_circle, circle_start, circle_dragging, circle_resizing, circle_selected

    if mode.get()=="rect":
        key = rect_var.get()
        if key and key in rectangles:
            if drawing_rect and rect_start:
                x0,y0 = rect_start; x1,y1 = event.x, event.y
                if abs(x1-x0)<5 or abs(y1-y0)<5:
                    if id_rectangles.get(key): C.delete(id_rectangles[key]); id_rectangles.pop(key, None)
                    if id_rectangles.get(key + "_label"): C.delete(id_rectangles[key + "_label"]); id_rectangles.pop(key + "_label", None)
                    rectangles[key]["coords"] = []
                else:
                    rectangles[key]["coords"] = [x0,y0,x1,y1]
                    if id_rectangles.get(key + "_label"):
                        C.delete(id_rectangles[key + "_label"]); id_rectangles.pop(key + "_label", None)
                    draw_rect_label(key)
                drawing_rect = False; rect_start = None
            rect_dragging = False; rect_resizing = False; rect_corner = None; rect_selected = None
            save_yaml(auto=True)

    elif mode.get()=="circle":
        key = circle_var.get()
        if key and key in circles:
            circ = circles[key]
            if drawing_circle and circle_start:
                cx0,cy0 = circle_start
                radius = int(math.hypot(event.x - cx0, event.y - cy0))
                if radius < 5:
                    if id_circles.get(key): C.delete(id_circles[key]); id_circles.pop(key, None)
                    if id_circles.get(key + "_label"): C.delete(id_circles[key + "_label"]); id_circles.pop(key + "_label", None)
                    circ.clear()
                else:
                    circ["center"] = [cx0,cy0]; circ["radius"] = radius; circ["color"] = color_seleccionado.get()
                    draw_circle(key)
                drawing_circle = False; circle_start = None
            circle_dragging = False; circle_resizing = False; circle_selected = None
            save_yaml(auto=True)

# ===================== BORRAR / UNDO / DUPLICAR =====================
def borrar_objeto_seleccionado(event=None, target=None):
    if target:
        t, key = target.get("type"), target.get("key")
    else:
        if mode.get()=="circle":
            t, key = "circle", circle_var.get()
        elif mode.get()=="rect":
            t, key = "rect", rect_var.get()
        elif mode.get()=="poly":
            t, key = "poly", poly_var.get()
        else:
            return

    if t=="circle" and key in circles:
        if id_circles.get(key): C.delete(id_circles[key]); id_circles.pop(key, None)
        if id_circles.get(key+"_label"): C.delete(id_circles[key+"_label"]); id_circles.pop(key+"_label", None)
        circles.pop(key, None); rebuild_circle_radios(); save_yaml(auto=True); return

    if t=="rect" and key in rectangles:
        undo_stack.append((key, rectangles[key].copy()))
        rectangles[key]["coords"]=[]
        if id_rectangles.get(key): C.delete(id_rectangles[key]); id_rectangles.pop(key, None)
        if id_rectangles.get(key+"_label"): C.delete(id_rectangles[key+"_label"]); id_rectangles.pop(key+"_label", None)
        draw_rect_label(key); rebuild_rect_radios(); save_yaml(auto=True); return

    if t=="poly" and key in polygons:
        delete_polygon(key)

def undo_rect(event=None):
    if mode.get()!="rect": return
    if undo_stack:
        key,data = undo_stack.pop()
        rectangles[key]=data; color=data["color"]
        if id_rectangles.get(key): C.delete(id_rectangles[key])
        if id_rectangles.get(key+"_label"): C.delete(id_rectangles[key+"_label"]); id_rectangles.pop(key+"_label", None)
        if "coords" in data and len(data["coords"])==4:
            id_rectangles[key]=C.create_rectangle(*data["coords"], outline=get_color(color), width=2)
            draw_rect_label(key)
        rebuild_rect_radios(); save_yaml(auto=True)

def undo_current(event=None):
    if mode.get()=="poly":
        undo_poly_func()
    elif mode.get()=="rect":
        undo_rect()
    elif mode.get()=="circle":
        borrar_objeto_seleccionado()

def duplicate_selected(event=None, target=None):
    if target:
        t, key = target.get("type"), target.get("key")
    else:
        if mode.get()=="circle":
            t, key = "circle", circle_var.get()
        elif mode.get()=="rect":
            t, key = "rect", rect_var.get()
        elif mode.get()=="poly":
            t, key = "poly", poly_var.get()
        else:
            return

    if t=="circle" and key in circles and "center" in circles[key]:
        newk = next_available("circle", circles)
        base = circles[key].copy()
        cx,cy = base["center"]; base["center"]= [cx+15, cy+15]
        circles[newk]=base
        draw_circle(newk); rebuild_circle_radios(); circle_var.set(newk); save_yaml(auto=True); return

    if t=="rect" and key in rectangles and rectangles[key].get("coords"):
        newk = next_available("rect", rectangles)
        x0,y0,x1,y1 = rectangles[key]["coords"]
        rectangles[newk] = {"coords":[x0+15,y0+15,x1+15,y1+15], "color": rectangles[key]["color"]}
        id_rectangles[newk] = C.create_rectangle(*rectangles[newk]["coords"], outline=get_color(rectangles[newk]["color"]), width=2)
        draw_rect_label(newk); rebuild_rect_radios(); rect_var.set(newk); save_yaml(auto=True); return

    if t=="poly" and key in polygons:
        m = re.match(r"(?i)\s*caja\s+(\d+)\s*$", key)
        if m:
            n = int(m.group(1)) + 1
            new_name = f"Caja {n}"
            while new_name in polygons:
                n += 1; new_name = f"Caja {n}"
        else:
            new_name = key + " (copy)"; i = 2
            while new_name in polygons:
                new_name = f"{key} (copy {i})"; i += 1

        polygons[new_name] = {"pts":[p[:] for p in polygons[key]["pts"]], "color": polygons[key]["color"]}
        polygons_disp[new_name] = [(x+15,y+15) for (x,y) in polygons_disp[key]]
        id_points[new_name]=[]; id_lines[new_name]=[]
        for (px,py) in polygons_disp[new_name]:
            pid = C.create_oval(px-POLY_POINT_RADIUS, py-POLY_POINT_RADIUS,
                                px+POLY_POINT_RADIUS, py+POLY_POINT_RADIUS,
                                fill="#2196f3", outline="white", width=2)
            id_points[new_name].append(pid)
        for i in range(len(polygons_disp[new_name])-1):
            id_lines[new_name].append(C.create_line(polygons_disp[new_name][i], polygons_disp[new_name][i+1],
                                                    width=POLY_LINE_WIDTH, fill=get_color(polygons[new_name]["color"])))
        id_lines[new_name].append(C.create_line(polygons_disp[new_name][0], polygons_disp[new_name][-1],
                                                width=POLY_LINE_WIDTH, fill=get_color(polygons[new_name]["color"])))
        id_polygons[new_name] = C.create_polygon(*[xy for t in polygons_disp[new_name] for xy in t],
                                                 fill=get_color(polygons[new_name]["color"]),
                                                 outline=get_color(polygons[new_name]["color"]), width=2, stipple='gray25')
        draw_poly_label(new_name); rebuild_poly_radios(); poly_var.set(new_name); save_yaml(auto=True)
        return

# ===================== RADIOS =====================
def rebuild_rect_radios():
    for w in frame_radios_rect.winfo_children(): w.destroy()
    for k in sort_keys_numeric(rectangles, "rect"):
        tk.Radiobutton(frame_radios_rect, text=k.replace("rect_","Rect "),
                       variable=rect_var, value=k, command=lambda v=k: None,
                       bg="#222", fg="cyan", selectcolor="#222", activebackground="#444").pack(anchor="w")
    if not rect_var.get():
        keys = sort_keys_numeric(rectangles, "rect")
        if keys: rect_var.set(keys[0])

def rebuild_circle_radios():
    for w in frame_radios_circle.winfo_children(): w.destroy()
    for k in sort_keys_numeric(circles, "circle"):
        tk.Radiobutton(frame_radios_circle, text=k.replace("circle_","Circle "),
                       variable=circle_var, value=k, command=lambda v=k: None,
                       bg="#222", fg="magenta", selectcolor="#222", activebackground="#444").pack(anchor="w")
    if not circle_var.get():
        keys = sort_keys_numeric(circles, "circle")
        if keys: circle_var.set(keys[0])

def rebuild_poly_radios():
    for w in frame_radios_poly.winfo_children(): w.destroy()
    zonas = ["Entrance 1","Entrance 2","ROI Area","Exit 1","Exit 2","Excluded Zone"]
    for z in zonas:
        tk.Radiobutton(frame_radios_poly, text=z, variable=poly_var, value=z,
                       command=start_poly, bg="#222", fg="cyan",
                       selectcolor="#222", activebackground="#444").pack(anchor="w")
    for k in polygons.keys():
        if k not in zonas:
            tk.Radiobutton(frame_radios_poly, text=k, variable=poly_var, value=k,
                           command=start_poly, bg="#222", fg="magenta",
                           selectcolor="#222", activebackground="#444").pack(anchor="w")
    if not poly_var.get():
        if zonas: poly_var.set(zonas[0])

def rebuild_radios_all():
    rebuild_rect_radios(); rebuild_circle_radios(); rebuild_poly_radios()

# ===================== REDIBUJAR =====================
def redraw_all_labels():
    for key in rectangles: draw_rect_label(key)
    for key in polygons_disp: draw_poly_label(key)
    for key in circles: draw_circle(key)

# ===================== SNAPSHOT PRO =====================
def _measure_text(draw, font, text):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        return bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        try:
            bbox = font.getbbox(text)
            return bbox[2]-bbox[0], bbox[3]-bbox[1]
        except Exception:
            return font.getsize(text)

def snapshot_pro():
    global ruta_yaml, img_tk, img_width, img_height
    if img_tk is None:
        messagebox.showerror("Error","Load a base image first."); return
    if ruta_yaml is None or not os.path.exists(ruta_yaml):
        messagebox.showerror("Error","Load a YAML first."); return
    fpath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")], initialdir=ultima_dir_imagen)
    if not fpath: return
    if not fpath.lower().endswith('.png'): fpath += '.png'

    pil_img = pil_img_global.copy()
    draw = ImageDraw.Draw(pil_img)
    with open(ruta_yaml,"r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    try: font = ImageFont.truetype("arial.ttf", size=32)
    except: font = ImageFont.load_default()

    # Rects
    for name,v in data.get("rectangles",{}).items():
        coords = v.get("coords",[])
        if coords and len(coords)==4:
            color = COLORES_HEX.get(v.get("color","Celeste"), "#3ec7eb")
            x0,y0,x1,y1 = [int(round(a)) for a in coords]
            x0,x1 = sorted([x0,x1]); y0,y1 = sorted([y0,y1])
            draw.rectangle([x0,y0,x1,y1], outline=color, width=4)
            label = name.replace("rect_","")
            tw,th = _measure_text(draw, font, label)
            cx,cy = (x0+x1)//2,(y0+y1)//2
            draw.rectangle([cx-2,cy-2,cx+tw+8,cy+th+4], fill=color)
            draw.text((cx+4,cy+2), label, fill="black", font=font)

    # Polígonos
    for name,v in data.get("polygons",{}).items():
        pts = v.get("pts",[])
        if len(pts)>2:
            color = COLORES_HEX.get(v.get("color","Celeste"), "#3ec7eb")
            pts_real = [(int(round(x)), int(round(y))) for x,y in pts]
            draw.polygon(pts_real, outline=color, width=4)
            label = name
            lx, ly = poly_label_point(pts_real)
            tw,th = _measure_text(draw, font, label)
            draw.rectangle([lx,ly,lx+tw+8,ly+th+4], fill=color)
            draw.text((lx+4,ly+2), label, fill="black", font=font)

    # Círculos
    for name,v in data.get("circles",{}).items():
        if "center" in v and "radius" in v:
            color = COLORES_HEX.get(v.get("color","Celeste"), "#3ec7eb")
            cx,cy = [int(round(c)) for c in v["center"]]
            r = int(round(v["radius"]))
            draw.ellipse([cx-r,cy-r,cx+r,cy+r], outline=color, width=4)
            label = name.replace("circle_","")
            tw,th = _measure_text(draw, font, label)
            draw.rectangle([cx,cy,cx+tw+8,cy+th+4], fill=color)
            draw.text((cx+4,cy+2), label, fill="black", font=font)

    pil_img.save(fpath)
    messagebox.showinfo("Snapshot", f"✅ Saved snapshot:\n{fpath}")

# ===================== UI =====================
topf = tk.Frame(root, bg="#222"); topf.pack(side="top", fill="x")

tk.Button(topf, text="Open image", command=openfile).pack(side="left")
tk.Button(topf, text="Save YAML", command=lambda: save_yaml(auto=False)).pack(side="left")
tk.Button(topf, text="Load YAML", command=load_yaml).pack(side="left")
tk.Button(topf, text="Clear", command=clear_all).pack(side="left")
tk.Button(topf, text="Copy YAML path", command=copiar_ruta_yaml, bg="#222", fg="#FFD700", activebackground="#444").pack(side="left", padx=10)

tk.Button(topf, text="Polygon mode",  command=set_mode_poly,   bg="#555", fg="#FFD700").pack(side="left", padx=10)
tk.Button(topf, text="Rectangle mode",command=set_mode_rect,   bg="#333", fg="#33ffaa").pack(side="left", padx=5)
tk.Button(topf, text="Circle mode",   command=set_mode_circle, bg="#222", fg="#fb00c0").pack(side="left", padx=5)

cb_color = ttk.Combobox(topf, values=COLORES_LISTA, textvariable=color_seleccionado, width=15, state="readonly")
cb_color.pack(side="left", padx=10); cb_color.bind("<<ComboboxSelected>>", set_color_rect)

tk.Button(topf, text="Snapshot PRO", command=snapshot_pro, fg="#111", bg="#FFD700", activebackground="#444").pack(side="left", padx=10)

tk.Button(topf, text="+ Rectangle", command=add_rectangle).pack(side="left", padx=(15,2))
tk.Button(topf, text="+ Circle",    command=add_circle).pack(side="left", padx=2)
tk.Button(topf, text="+ Polygon (named)", command=add_polygon_named).pack(side="left", padx=2)
tk.Button(topf, text="Delete selected (Del)", command=borrar_objeto_seleccionado).pack(side="left", padx=(10,2))

# Sidebar
sidebar = tk.Frame(root, bg="#232323"); sidebar.pack(side="right", fill="y")
frame_radios_rect   = tk.Frame(sidebar, bg="#222")
frame_radios_poly   = tk.Frame(sidebar, bg="#222")
frame_radios_circle = tk.Frame(sidebar, bg="#222")

# Canvas
C = tk.Canvas(root, bg="black")
C.pack(fill="both", expand=True)

# Eventos de canvas
C.bind("<Button-1>", canvas_click)
C.bind("<B1-Motion>", canvas_drag)
C.bind("<ButtonRelease-1>", canvas_release)
C.bind("<Button-3>", on_right_click)

# Bindings
root.bind("<Control-s>", lambda e: save_yaml(auto=False))
root.bind("<Delete>", borrar_objeto_seleccionado)
root.bind("<Control-z>", undo_current)
root.bind("<Control-d>", duplicate_selected)

# Init
rebuild_radios_all()
set_mode_poly()
root.mainloop()