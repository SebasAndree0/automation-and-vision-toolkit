import os
import argparse
import cv2
import numpy as np


VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def apply_rect_mask(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Returns a masked image: everything black except the rectangle [x1:x2, y1:y2].
    """
    h, w = img.shape[:2]

    # Clamp to image bounds
    x1c = max(0, min(w, x1))
    x2c = max(0, min(w, x2))
    y1c = max(0, min(h, y1))
    y2c = max(0, min(h, y2))

    # Ensure proper ordering
    x1c, x2c = sorted([x1c, x2c])
    y1c, y2c = sorted([y1c, y2c])

    out = np.zeros_like(img)
    if x2c > x1c and y2c > y1c:
        out[y1c:y2c, x1c:x2c] = img[y1c:y2c, x1c:x2c]
    return out


def iter_images(folder: str):
    for fname in os.listdir(folder):
        if fname.lower().endswith(VALID_EXTS):
            yield fname


def main():
    parser = argparse.ArgumentParser(
        description="Apply a rectangular mask to all images in a folder (everything black except a rectangle)."
    )
    parser.add_argument("--input", required=True, help="Input folder with images.")
    parser.add_argument("--output", default=None, help="Output folder (default: <input>_masked).")
    parser.add_argument("--x1", type=int, required=True, help="Rectangle x1 (left).")
    parser.add_argument("--y1", type=int, required=True, help="Rectangle y1 (top).")
    parser.add_argument("--x2", type=int, required=True, help="Rectangle x2 (right).")
    parser.add_argument("--y2", type=int, required=True, help="Rectangle y2 (bottom).")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original images (DANGEROUS). If set, --output is ignored.",
    )

    args = parser.parse_args()

    input_dir = args.input
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    if args.in_place:
        output_dir = input_dir
    else:
        output_dir = args.output or (input_dir.rstrip("\\/") + "_masked")
        os.makedirs(output_dir, exist_ok=True)

    total = 0
    ok = 0

    for fname in iter_images(input_dir):
        total += 1
        in_path = os.path.join(input_dir, fname)
        img = cv2.imread(in_path)
        if img is None:
            print(f"[WARN] Could not read: {in_path}")
            continue

        masked = apply_rect_mask(img, args.x1, args.y1, args.x2, args.y2)

        out_path = os.path.join(output_dir, fname)
        if cv2.imwrite(out_path, masked):
            ok += 1
            if args.in_place:
                print(f"[OK] Overwritten: {out_path}")
            else:
                print(f"[OK] Saved: {out_path}")
        else:
            print(f"[WARN] Could not write: {out_path}")

    print(f"\nDone. Processed: {total}, Saved: {ok}. Output: {output_dir}")


if __name__ == "__main__":
    main()