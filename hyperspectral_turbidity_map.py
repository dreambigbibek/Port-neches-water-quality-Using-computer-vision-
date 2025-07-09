import pandas as pd
import os

# Load your turbidity mapping CSV
turbidity_csv_path = "/home/bmt.lamar.edu/bgautam3/deep neural network/SaltBarrier_image_turbidity.csv"  # <-- update if CSV is in different location
turbidity_df = pd.read_csv(turbidity_csv_path)


# Base folder path (common prefix)
base_folder = "/home/bmt.lamar.edu/bgautam3"

# Band folders
band_folders = ['SB_Band_27', 'SB_Band_29', 'SB_Band_31']

# Store all data
data = []

for band in band_folders:
    folder_path = os.path.join(base_folder, band)
    for file in os.listdir(folder_path):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Match turbidity
        turbidity_row = turbidity_df[turbidity_df['ImageFile'] == file]
        print(turbidity_row)

        if turbidity_row.empty:
            print(f"Warning: No turbidity found for {file}")
            continue

        turbidity_value = turbidity_row.iloc[0]['Turbidity']
        full_path = os.path.join(folder_path, file)

        data.append({
            'path': full_path,
            'filename': file,
            'turbidity': turbidity_value
        })

# Create dataframe
final_df = pd.DataFrame(data)

# Save final mapping
final_df.to_csv("SB_final_HS_image_turbidity_mapping.csv", index=False)

print("Done: CSV created.")
