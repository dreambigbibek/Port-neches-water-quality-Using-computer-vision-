import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# PARAMETERS
CSV_FILE = "/home/bmt.lamar.edu/bgautam3/deep neural network/SB_final_HS_image_turbidity_mapping_july.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

# Load CSV
df = pd.read_csv(CSV_FILE)
image_paths = df['path'].values
turbidity = df['turbidity'].values

# Split data
train_paths, val_paths, train_turbidity, val_turbidity = train_test_split(
    image_paths, turbidity, test_size=0.2, random_state=42)

# Image preprocessing
def preprocess_images(paths):
    data = []
    for path in paths:
        img = load_img(path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img) / 255.0
        data.append(img_array)
    return np.array(data)

X_train = preprocess_images(train_paths)
X_val = preprocess_images(val_paths)
y_train = train_turbidity
y_val = val_turbidity

#RestNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#base_model.trainable = False 
for layer in base_model.layers[-5:]:  ## take the last -5 layers for fine-tuning
    layer.trainable = True  
 

# Add regression head
x = base_model.output
x = GlobalAveragePooling2D()(x)         # flattening spatial map
x = Dense(256, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(1)(x)                         # regression output

model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

# Predict and evaluate
y_train_pred = model.predict(X_train).flatten()
y_val_pred = model.predict(X_val).flatten()

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation R²: {val_r2:.4f}")

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_val_pred, alpha=0.7, edgecolor='k')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
plt.xlabel("True Turbidity")
plt.ylabel("Predicted Turbidity")
plt.title("Predicted vs True Turbidity")
plt.grid(True)
plt.tight_layout()
plt.savefig("SB_HS_predicted_vs_true_turbidity_resnet.png", dpi=300)
plt.show()
