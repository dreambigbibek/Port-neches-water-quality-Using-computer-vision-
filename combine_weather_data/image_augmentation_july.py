import os
from pathlib import Path
from PIL import Image, ImageOps

INPUT_DIR = r"/home/bmt.lamar.edu/bgautam3/deep neural network/combine_weather_data/Image_turbidity_map_saltwater_july/"
TRANSLATE_X_FRAC = 0.05               # 5% of width to the right
TRANSLATE_Y_FRAC = 0.05               # 5% of height down
ROTATION_DEG = 5                      # +5 degrees
# -----------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def center_crop(img, target_w, target_h):
    w, h = img.size
    left = max(0, (w - target_w) // 2)
    top = max(0, (h - target_h) // 2)
    right = left + target_w
    bottom = top + target_h
    return img.crop((left, top, right, bottom))

def translate(img, tx, ty, fillcolor=None):
    # Affine transform matrix for translation (a, b, c, d, e, f)
    # x' = a*x + b*y + c ; y' = d*x + e*y + f
    # For pure translation: a=1, b=0, c=tx, d=0, e=1, f=ty
    w, h = img.size
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, tx, 0, 1, ty),
        resample=Image.BICUBIC,
        fillcolor=fillcolor
    )

def process_image(path: Path):
    try:
        with Image.open(path) as im0:
            # Normalize orientation
            im0 = ImageOps.exif_transpose(im0)
            base = path.stem
            ext = path.suffix

            # 1) Flip L-R
            aug1 = ImageOps.mirror(im0)
            aug1.save(path.with_name(f"{base}_augmented1{ext}"))

            # 2) Rotate +5°; keep original size (rotate with expand then center-crop back)
            W, H = im0.size
            rot = im0.rotate(ROTATION_DEG, resample=Image.BICUBIC, expand=True, fillcolor=None)
            rot_cropped = center_crop(rot, W, H)
            rot_cropped.save(path.with_name(f"{base}_augmented2{ext}"))

            # 3) Translate by a fraction of width/height
            tx = int(W * TRANSLATE_X_FRAC)
            ty = int(H * TRANSLATE_Y_FRAC)
            # Use edge fillcolor if image has no alpha; None keeps transparent where supported
            fill = None if im0.mode in ("RGBA", "LA") else im0.getpixel((0,0))
            trans = translate(im0, tx, ty, fillcolor=fill)
            trans.save(path.with_name(f"{base}_augmented3{ext}"))

            print(f"Done: {path.name}")
    except Exception as e:
        print(f"Skip {path.name}: {e}")

def main():
    in_dir = Path(INPUT_DIR)
    for p in sorted(in_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            # Avoid re-augmenting already-augmented files
            if any(s in p.stem for s in ["_augmented1", "_augmented2", "_augmented3"]):
                continue
            process_image(p)

if __name__ == "__main__":
    main()
