import pandas as pd
from datetime import datetime

# Loading CSV file
df = pd.read_csv("/home/bmt.lamar.edu/bgautam3/deep neural network/image_turbidity_mapping_july.csv")  

# Converting 'ImageFile' to datetime
def parse_datetime(filename):
    # Strip extension
    base = filename.replace('.JPG', '')
    # Convert to datetime object
    dt = datetime.strptime(base, "%Y-%m-%d_%H-%M-%S")
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# Apply conversion
df['Datetime'] = df['ImageFile'].apply(parse_datetime)

# Reorder columns: Datetime first, then Turbidity
df = df[['Datetime', 'Turbidity']]

# Save to new CSV
df.to_csv('turbidity_datetime.csv', index=False)

print("✅ Done. Saved as 'turbidity_datetime.csv'")
