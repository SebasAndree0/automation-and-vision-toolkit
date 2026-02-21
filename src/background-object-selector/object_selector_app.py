import os
import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, Canvas

import numpy as np
from PIL import Image, ImageTk
from scipy.ndimage import label, find_objects
from rembg import new_session, remove


# ============================================================
# Utils: crop by alpha (tight crop)
# ============================================================
def crop_to_alpha(img: Image.Image, alpha_threshold: int = 10) -> Image.Image:
    """
    Crops an RGBA image tightly around pixels with alpha > alpha_threshold.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    arr = np.array(img)
    alpha = arr[:, :, 3]
    mask = alpha > alpha_threshold

    if not mask.any():
        return img

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img.crop((x0, y0, x1, y1))


# ============================================================
# Utils: connected components on alpha mask
# ============================================================
def extract_main_objects(
    img: Image.Image,
    alpha_threshold: int = 10,
    min_object_side: int = 50,
):
    """
    Finds connected components ("islands") in the alpha mask and extracts them as separate RGBA images.

    Returns a list of dicts:
    {
      "img": PIL.Image (RGBA)  # cropped object image
      "slice": tuple(slice, slice)  # region in original
      "mask": np.ndarray(bool)  # mask of component in that slice
    }
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    arr = np.array(img)
    alpha = arr[:, :, 3]
    mask = alpha > alpha_threshold

    labeled, _ = label(mask)
    slices = find_objects(labeled)

    objects = []
    for idx, slc in enumerate(slices):
        if slc is None:
            continue

        subarr = arr[slc[0], slc[1], :]

        # Filter out small components by side length (keeps logic simple & predictable)
        if subarr.shape[0] < min_object_side or subarr.shape[1] < min_object_side:
            continue

        # Exact mask for this component
        mask_obj = (labeled[slc] == (idx + 1))

        # Apply alpha mask
        subarr_obj = subarr.copy()
        subarr_obj[~mask_obj] = [0, 0, 0, 0]

        subimg = Image.fromarray(subarr_obj)
        subimg = crop_to_alpha(subimg, alpha_threshold)

        objects.append({"img": subimg, "slice": slc, "mask": mask_obj})

    return objects


# ============================================================
# GUI App
# ============================================================
class BackgroundObjectSelectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Background Object Selector")
        self.root.geometry("1100x700")
        self.root.config(bg="#181824")

        # Settings (you can tweak these)
        self.alpha_threshold = 10
        self.min_object_side = 50
        self.rembg_model_name = "u2net"  # common default in rembg

        # State
        self.original_no_bg: Image.Image | None = None
        self.processed_image: Image.Image | None = None
        self.objects = []  # list of dicts
        self.objects_active = []
        self.thumbnails_widgets = []

        # Main image panel
        frm_main = tk.Frame(self.root, bg="#181824")
        frm_main.pack(side="left", fill="both", expand=True)

        self.panel = tk.Label(frm_main, bg="#222")
        self.panel.pack(fill="both", expand=True, padx=30, pady=30)

        self.panel_width = 700
        self.panel_height = 600

        # Side panel
        frm_side = tk.Frame(self.root, bg="#1a1a2e", width=240)
        frm_side.pack(side="right", fill="y")

        lbl = tk.Label(
            frm_side,
            text="Detected objects:",
            bg="#1a1a2e",
            fg="white",
            font=("Arial", 12, "bold"),
        )
        lbl.pack(pady=10)

        # Scrollable thumbnails
        self.canvas = Canvas(frm_side, bg="#1a1a2e", width=220, height=600, highlightthickness=0)
        self.scrollbar = Scrollbar(frm_side, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#1a1a2e")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Buttons
        btn_load = tk.Button(
            frm_side,
            text="Load image",
            command=self.load_image,
            bg="#29b6f6",
            fg="white",
            font=("Arial", 11, "bold"),
        )
        btn_load.pack(pady=6, fill="x", padx=6)

        btn_restore = tk.Button(
            frm_side,
            text="Restore all",
            command=self.restore_all,
            bg="#43a047",
            fg="white",
            font=("Arial", 11, "bold"),
        )
        btn_restore.pack(pady=6, fill="x", padx=6)

        btn_save = tk.Button(
            frm_side,
            text="Save PNG",
            command=self.save_image,
            bg="#ffca28",
            fg="#181824",
            font=("Arial", 11, "bold"),
        )
        btn_save.pack(pady=6, fill="x", padx=6)

        self.lbl_info = tk.Label(frm_side, text="", bg="#1a1a2e", fg="#ffca28", font=("Arial", 10))
        self.lbl_info.pack(pady=8)

    # ---------------- UI actions ----------------
    def load_image(self):
        img_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )
        if not img_path:
            return

        self.lbl_info.config(text="Removing background... please wait.")
        self.root.update()

        img = Image.open(img_path).convert("RGBA")

        # Remove background
        session = new_session(model_name=self.rembg_model_name)
        img_no_bg = remove(img, session=session)

        # Tight crop
        img_no_bg = crop_to_alpha(img_no_bg, alpha_threshold=self.alpha_threshold)

        self.original_no_bg = img_no_bg

        # Extract objects
        self.objects = extract_main_objects(
            img_no_bg,
            alpha_threshold=self.alpha_threshold,
            min_object_side=self.min_object_side,
        )
        self.objects_active = [True] * len(self.objects)

        self.lbl_info.config(
            text=f"{len(self.objects)} objects detected. Double-click a thumbnail to toggle."
        )

        self.render_thumbnails()
        self.update_main_preview()

    def render_thumbnails(self):
        # Clear old thumbnails
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.thumbnails_widgets = []

        for idx, obj in enumerate(self.objects):
            thumb = obj["img"].copy()
            thumb.thumbnail((80, 80), Image.LANCZOS)

            imgtk = ImageTk.PhotoImage(thumb)

            lbl = tk.Label(self.scrollable_frame, image=imgtk, bd=4, relief="solid")
            lbl.image = imgtk  # keep ref

            lbl.grid(row=idx, column=0, pady=6, padx=6)

            self._update_border(lbl, self.objects_active[idx])

            lbl.bind("<Double-Button-1>", lambda e, i=idx: self.toggle_object(i))

            self.thumbnails_widgets.append(lbl)

    def _update_border(self, widget, active: bool):
        if active:
            widget.config(highlightbackground="#43a047", highlightthickness=2, bd=4, relief="solid")
        else:
            widget.config(highlightbackground="#f44336", highlightthickness=2, bd=4, relief="solid")

    def toggle_object(self, idx: int):
        self.objects_active[idx] = not self.objects_active[idx]
        self._update_border(self.thumbnails_widgets[idx], self.objects_active[idx])
        self.update_main_preview()

    def restore_all(self):
        self.objects_active = [True] * len(self.objects)
        for idx, lbl in enumerate(self.thumbnails_widgets):
            self._update_border(lbl, True)
        self.update_main_preview()

    def update_main_preview(self):
        if not self.objects or self.original_no_bg is None:
            return

        arr = np.array(self.original_no_bg)
        alpha = arr[:, :, 3]
        mask_total = np.zeros_like(alpha, dtype=bool)

        for obj, active in zip(self.objects, self.objects_active):
            if active:
                mask_total[obj["slice"]] |= obj["mask"]

        arr_out = arr.copy()
        arr_out[~mask_total] = [0, 0, 0, 0]

        img_final = Image.fromarray(arr_out)

        w, h = img_final.size
        scale = min(self.panel_width / w, self.panel_height / h, 1.0)
        img_disp = img_final.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(img_disp)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)

        self.processed_image = img_final

    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("No image", "No processed image to save.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            title="Save image as...",
        )
        if path:
            self.processed_image.save(path)
            messagebox.showinfo("Saved", f"Image saved to:\n{path}")


def main():
    root = tk.Tk()
    app = BackgroundObjectSelectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()