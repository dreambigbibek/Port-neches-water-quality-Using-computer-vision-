import pandas as pd

# Load the turbidity data
turbidity_df = pd.read_csv("turbidity_datetime.csv")  # Replace with your actual turbidity CSV file

# Load the rainfall (increment/accumulation) data
rain_df = pd.read_csv("/home/bmt.lamar.edu/bgautam3/deep neural network/RainfallData/onerain_St._4320_LNVA_Saltwater_Barrier_Neches_River_48367_Rain_Accumulation_0.csv")  # Replace with your actual rainfall CSV file



# Convert to datetime, forcing errors to NaT
turbidity_df['Datetime'] = pd.to_datetime(turbidity_df['Datetime'], errors='coerce')
rain_df['Reading'] = pd.to_datetime(rain_df['Reading'], errors='coerce')
print(turbidity_df.__sizeof__)
print(rain_df.__sizeof__)

# Drop rows where conversion failed
turbidity_df = turbidity_df.dropna(subset=['Datetime'])
rain_df = rain_df.dropna(subset=['Reading'])

matched_rows = []

# Compare times and keep only matched rows (within 7 minutes)
for _, turb_row in turbidity_df.iterrows():
    turb_time = turb_row['Datetime']
    time_diffs = abs((rain_df['Reading'] - turb_time).dt.total_seconds().abs())

    # Filter close matches
    close_rows = rain_df[time_diffs <= 900]  # 15 minutes in seconds

    if not close_rows.empty:
        closest_idx = time_diffs[time_diffs <= 900].idxmin()
        closest = rain_df.loc[closest_idx]

        # Merge only matched rows
        matched_rows.append({
            'Datetime': turb_time,
            'Turbidity': turb_row['Turbidity'],
            'Reading': closest['Reading'],
            'Value': closest['Value'],
            'Unit': closest['Unit'],
            'Data Quality': closest['Data Quality']
        })

# Create DataFrame of matches
matched_df = pd.DataFrame(matched_rows)



# Save to CSV
matched_df.to_csv("RainfallData/merged_accumulation_turbidity_with_rain_if_within_7min.csv", index=False)

print("✅ Done: merged if time difference < 7 minutes. Saved to 'merged_accumulation_turbidity_with_rain_if_within_7min.csv'")
