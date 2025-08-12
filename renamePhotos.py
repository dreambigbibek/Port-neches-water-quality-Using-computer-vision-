import os
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
import shutil

# === CONFIGURATION ===
source_folder = "/home/bmt.lamar.edu/bgautam3/Saltwater Barrier Images/4. June 30- July 24/";
destination_folder = "/home/bmt.lamar.edu/bgautam3/deep neural network/renamedPhotos2";

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

def get_exif_data(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None

        exif = {
            TAGS.get(tag): value
            for tag, value in exif_data.items()
            if tag in TAGS
        }
        return exif
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def get_date_taken(exif):
    date_str = exif.get('DateTimeOriginal') or exif.get('DateTime')
    if date_str:
        try:
            return datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
        except:
            return None
    return None

def rename_and_copy_images():
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)

        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # Skip non-image files

        exif = get_exif_data(file_path)
        if not exif:
            print(f"No EXIF data found for {filename}. Skipping.")
            continue

        date_taken = get_date_taken(exif)
        if not date_taken:
            print(f"No DateTimeOriginal found for {filename}. Skipping.")
            continue

        # Format: 2023-07-08_14-30-15.jpg
        new_filename = date_taken.strftime('%Y-%m-%d_%H-%M-%S') + os.path.splitext(filename)[1]
        new_path = os.path.join(destination_folder, new_filename)

        # If filename exists, append a counter
        counter = 1
        while os.path.exists(new_path):
            new_filename = date_taken.strftime('%Y-%m-%d_%H-%M-%S') + f"_{counter}" + os.path.splitext(filename)[1]
            new_path = os.path.join(destination_folder, new_filename)
            counter += 1

        shutil.copy2(file_path, new_path)
        print(f"Copied and renamed: {filename} → {new_filename}")

rename_and_copy_images()
