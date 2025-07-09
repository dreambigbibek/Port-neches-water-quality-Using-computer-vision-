import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# PARAMETERS
CSV_FILE = "/home/bmt.lamar.edu/bgautam3/deep neural network/SB_final_HS_image_turbidity_mapping.csv"  # use your generated CSV
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

# 1. Load CSV
df = pd.read_csv(CSV_FILE)

# 2. Prepare X and y
image_paths = df['path'].values
turbidity = df['turbidity'].values

# 3. Train-test split
train_paths, val_paths, train_turbidity, val_turbidity = train_test_split(
    image_paths, turbidity, test_size=0.2, random_state=42)

# 4. Image loading function
def preprocess_images(paths):
    data = []
    for path in paths:
        img = load_img(path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # normalize
        data.append(img_array)
    return np.array(data)

# Load actual data (slow but simple version)
X_train = preprocess_images(train_paths)
X_val = preprocess_images(val_paths)
y_train = train_turbidity
y_val = val_turbidity

# # 5. Build model
# # 1. Load VGG16 with top (1000-class output)
# vgg_base = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

# # 2. Freeze all VGG layers
# for layer in vgg_base.layers:
#     layer.trainable = False

# # 3. Take 1000-class output
# #x = vgg_base.output  # shape: (None, 1000)
# x = vgg_base.get_layer('fc2').output

# # 4. Add 3 Dense layers (NO dropout)
# x = Dense(512, activation='relu')(x)
# x = Dense(256, activation='relu')(x)
# x = Dense(64, activation='relu')(x)

# # Final output node: turbidity
# x = Dense(1)(x)

# # 5. Build and compile the model
# model = Model(inputs=vgg_base.input, outputs=x)
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# # 6. Train the model
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     verbose=1
# )

# Load pretrained VGG16 without top layers
base_model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

# Freeze only convolutional layers
for layer in base_model.layers:
    if 'conv' in layer.name:
        layer.trainable = False
    else:
        layer.trainable = True

# Custom regression head
#x = base_model.output
x = base_model.get_layer('fc2').output
#x = Flatten()(x)
x = Dense(512, activation='relu')(x)
# x = Dropout(0.5)(x)  # regularization (optional)
x = Dense(1)(x)  # regression output (no activation)

model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 6. Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Predict on validation set
y_pred = model.predict(X_val).flatten()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")

# Calculate R² score
r2 = r2_score(y_val, y_pred)
print(f"Validation R²: {r2:.4f}")

import matplotlib.pyplot as plt
# Scatter plot: True vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.7, edgecolor='k')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')  # identity line
plt.xlabel("True Turbidity")
plt.ylabel("Predicted Turbidity")
plt.title("Predicted vs True Turbidity")
plt.grid(True)
plt.tight_layout()
plt.show()
# Save to file
plt.savefig("SB_HS_predicted_vs_true_turbidity.png", dpi=300)
plt.close()
