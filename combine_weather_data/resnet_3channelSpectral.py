import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ===== Paths =====
CSV_B27 = "/home/bmt.lamar.edu/bgautam3/deep neural network/combine_weather_data/SB_RawHS__turbidity_mapping_band27.csv"
CSV_B29 = "/home/bmt.lamar.edu/bgautam3/deep neural network/combine_weather_data/SB_RawHS__turbidity_mapping_band29.csv"
CSV_B31 = "/home/bmt.lamar.edu/bgautam3/deep neural network/combine_weather_data/SB_RawHS__turbidity_mapping_band31.csv"

# ===== Hyperparams =====
IMAGE_SIZE = (224,224)
BATCH_SIZE = 16
EPOCHS = 20
VAL_SPLIT = 0.2
SEED = 42

# ===== Load & merge =====
b27 = pd.read_csv(CSV_B27).rename(columns={"path": "spec27", "turbidity": "turb27"})
b29 = pd.read_csv(CSV_B29).rename(columns={"path": "spec29", "turbidity": "turb29"})
b31 = pd.read_csv(CSV_B31).rename(columns={"path": "spec31", "turbidity": "turb31"})

df = (
    b27.merge(b29[["filename", "spec29"]], on="filename")
       .merge(b31[["filename", "spec31"]], on="filename")
       .dropna(subset=["spec27","spec29","spec31","turb27"])
)

# df.to_csv("temp_merged_spectral_data.csv", index=False)
# print(df.head())

# ===== Loader (handles JPEG/PNG) =====
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    # img = tf.cast(img, tf.float32) / 255.0
    return img # (H,W,1)     #img.numpy() 

print(tf.shape(load_img("/home/bmt.lamar.edu/bgautam3/SaltBarrierJuly/HS_SaltBarrier_output_july_noColormap/Band_27/2025-05-15_08-10-56.JPG")))

# Build X (3-band stack) & y 
X = []
H, W = IMAGE_SIZE

from tensorflow.keras.preprocessing.image import load_img, img_to_array

for _, row in df.iterrows():
    g27 = img_to_array(load_img(row["spec27"], color_mode="grayscale", target_size=(H, W)))  # (224,224,1)
    g29 = img_to_array(load_img(row["spec29"], color_mode="grayscale", target_size=(H, W)))  # (224,224,1)
    g31 = img_to_array(load_img(row["spec31"], color_mode="grayscale", target_size=(H, W)))  # (224,224,1)

    # Stack into channels-last (H, W, 3)
    merged = np.concatenate([g27, g29, g31], axis=-1).astype(np.float32)  # (224,224,3)
    X.append(merged)

X = np.asarray(X, dtype=np.float32)
y = df["turb27"].to_numpy(dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ===== Split =====
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SPLIT, random_state=SEED)

# ===== Model =====
inp = Input(shape=(224, 224, 3))
backbone = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
for layer in backbone.layers:
    layer.trainable = False
for layer in backbone.layers[-7:]:  # fine-tune a bit
    layer.trainable = True

x = backbone(inp, training=False) # backbone without top layers
gap = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(gap)
out = Dense(1)(x)
model = Model(inp, out)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])   #, jit_compile=False
model.summary()

# ===== Train =====
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# ===== Predict =====
y_train_pred = model.predict(X_train, verbose=0).ravel()   #ravel flatten the array into 1D
y_val_pred   = model.predict(X_val,   verbose=0).ravel()

# ===== Metrics (manual RMSE to support older sklearn) =====
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

rmse_tr = rmse(y_train, y_train_pred)
r2_tr   = r2_score(y_train, y_train_pred)
rmse_v  = rmse(y_val,   y_val_pred)
r2_v    = r2_score(y_val,   y_val_pred)

print(f"Training  RMSE: {rmse_tr:.4f}  R²: {r2_tr:.4f}")
print(f"Validation RMSE: {rmse_v:.4f}  R²: {r2_v:.4f}")

# ===== Plot Pred vs True =====
plt.figure(figsize=(5.5,5.5))
plt.scatter(y_val, y_val_pred, s=14, alpha=0.6)
lo = float(np.floor(min(y_val.min(),  y_val_pred.min())))
hi = float(np.ceil (max(y_val.max(),  y_val_pred.max())))
plt.plot([lo, hi], [lo, hi], lw=2)
plt.xlabel("True turbidity"); plt.ylabel("Predicted turbidity")
plt.title(f"Pred vs True  (RMSE={rmse_v:.2f}, R²={r2_v:.2f})")
plt.axis('equal'); plt.xlim(lo, hi); plt.ylim(lo, hi)
plt.grid(True, ls="--", alpha=0.35)
plt.tight_layout()
plt.savefig("pred_vs_true_3Channel.png", dpi=300, bbox_inches="tight")
plt.close()
