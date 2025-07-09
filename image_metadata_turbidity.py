import pandas as pd
import os
from datetime import datetime
from PIL import Image
import piexif

# PARAMETERS — customize these:
turbidity_csv_path = "turbidity_data.csv"
image_folder = "path_to_images"
output_csv_path = "image_with_turbidity.csv"

# 1. Load turbidity data
turbidity_df = pd.read_csv(turbidity_csv_path)
turbidity_df['timestamp'] = pd.to_datetime(turbidity_df['timestamp'])
turbidity_df = turbidity_df.sort_values('timestamp')

# 2. Extract image timestamps (assuming filenames like IMG_20240615_093000.jpg)
image_data = []
for file in os.listdir(image_folder):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    name_part = file.split('.')[0]
    dt_str = name_part.split('_')[1]  # adjust this based on your actual filename format
    img_time = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    image_data.append({'filename': file, 'timestamp': img_time})

images_df = pd.DataFrame(image_data).sort_values('timestamp')

# 3. Merge nearest turbidity
merged_df = pd.merge_asof(images_df, turbidity_df, on='timestamp', direction='nearest', tolerance=pd.Timedelta('10min'))

# 4. Save final matched result
merged_df.to_csv(output_csv_path, index=False)
