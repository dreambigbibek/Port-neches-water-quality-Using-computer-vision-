import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- PARAMETERS ---
IMAGE_DIR = "/home/bmt.lamar.edu/bgautam3/croppedImageDL2"
CSV_FILE = "final_target_turbidity.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# --- LOAD CSV AND VERIFY IMAGE MATCHING ---
df = pd.read_csv("/home/bmt.lamar.edu/bgautam3/deep neural network/SaltBarrier_image_turbidity.csv")
df.columns = ['filename', 'turbidity']  # Use proper column names from the file

# Filter only existing image files
df['filepath'] = df['filename'].apply(lambda f: os.path.join(IMAGE_DIR, f))
df = df[df['filepath'].apply(os.path.exists)].reset_index(drop=True)

print(f"✅ Total matched image-label pairs: {len(df)}")

# --- LOAD IMAGES AND LABELS ---
def load_images_and_labels(df):
    images = []
    labels = []
    for _, row in df.iterrows():
        img = load_img(row['filepath'], target_size=IMAGE_SIZE)
        img = img_to_array(img) / 255.0
        images.append(img)
        labels.append(row['turbidity'])
    return np.array(images), np.array(labels)

X, y = load_images_and_labels(df)

# --- TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- VGG16 REGRESSION MODEL ---
base_model = VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
x = base_model.get_layer('fc2').output
x = Dense(1, activation='linear', name='regression_output')(x)
model = Model(inputs=base_model.input, outputs=x)

# --- FREEZE LAYERS ---
for layer in base_model.layers:
    if layer.name in ['fc1', 'fc2']:
        layer.trainable = True
    else:
        layer.trainable = False

# --- COMPILE & TRAIN ---
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, y_train, validation_split=0.3, epochs=EPOCHS, batch_size=BATCH_SIZE)

# --- EVALUATE ---
y_pred = model.predict(X_test).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nRMSE on Test Set: {rmse:.2f}")
print(f"R^2 on Test Set: {r2:.2f}")

# --- PLOT: Actual vs Predicted ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Turbidity')
plt.ylabel('Predicted Turbidity')
plt.title('Actual vs Predicted Turbidity')
plt.grid(True)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
plt.show()
