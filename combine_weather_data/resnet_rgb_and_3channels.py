import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate, Multiply,Subtract
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras import ops

# Paths 
CSV_B27 = "/home/bmt.lamar.edu/bgautam3/deep neural network/combine_weather_data/SB_RawHS__turbidity_mapping_band27.csv"
CSV_B29 = "/home/bmt.lamar.edu/bgautam3/deep neural network/combine_weather_data/SB_RawHS__turbidity_mapping_band29.csv"
CSV_B31 = "/home/bmt.lamar.edu/bgautam3/deep neural network/combine_weather_data/SB_RawHS__turbidity_mapping_band31.csv"
CSV_RGB = "/home/bmt.lamar.edu/bgautam3/deep neural network/combine_weather_data/SB_final_HS_image_turbidity_mapping_july.csv"  

# Hyperparams 
IMAGE_SIZE = (224,224)
BATCH_SIZE = 16
EPOCHS = 20
VAL_SPLIT = 0.2
SEED = 42

#Utils
def exists(p):
    return isinstance(p, str) and os.path.isfile(p)

def load_gray(path):  # returns (H,W,1) float32 in [0,1]
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    # img = tf.cast(img, tf.float32) / 255.0
    return img     #.numpy()

def load_rgb(path):   # returns (H,W,3) float32 in [0,1]
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    # img = tf.cast(img, tf.float32) / 255.0
    return img    #.numpy()    #.numpy()

#Load & merge
b27 = pd.read_csv(CSV_B27).rename(columns={"path": "spec27", "turbidity": "turb27"})
b29 = pd.read_csv(CSV_B29).rename(columns={"path": "spec29"})
b31 = pd.read_csv(CSV_B31).rename(columns={"path": "spec31"})
rgb = pd.read_csv(CSV_RGB).rename(columns={"path": "rgb_path", "turbidity": "turb_rgb"})

# minimal cols expected: filename + path(s) + turbidity in at least one file
df = (
    b27.merge(b29[["filename", "spec29"]], on="filename")
       .merge(b31[["filename", "spec31"]], on="filename")
       .merge(rgb[["filename", "rgb_path", "turb_rgb"]], on="filename", how="inner")
)

# # Prefer turbidity from b27 if present; else fallback to rgb's label
# if "turb27" in df.columns:
#     df["turbidity"] = df["turb27"].where(df["turb27"].notna(), df["turb_rgb"])
# else:
#     df["turbidity"] = df["turb_rgb"]

# Drop rows with missing critical fields
df = df.dropna(subset=["spec27", "spec29", "spec31", "rgb_path", "turb27"]).copy()

# Keep only rows with actual files
df = df[
    df["spec27"].apply(exists) &
    df["spec29"].apply(exists) &
    df["spec31"].apply(exists) &
    df["rgb_path"].apply(exists)
].reset_index(drop=True)

print("Total samples after filtering:", len(df))

# Build arrays 
X_spec = []
X_rgb  = []
for _, row in df.iterrows():
    g27 = load_gray(row["spec27"])  # (H,W,1)
    g29 = load_gray(row["spec29"])
    g31 = load_gray(row["spec31"])
    spec_stack = np.concatenate([g27, g29, g31], axis=-1)  # (H,W,3)

    rgb_img = load_rgb(row["rgb_path"])  # (H,W,3)

    X_spec.append(spec_stack)
    X_rgb.append(rgb_img)

X_spec = np.asarray(X_spec, dtype=np.float32)
X_rgb  = np.asarray(X_rgb,  dtype=np.float32)
y = df["turb27"].to_numpy(dtype=np.float32)

print("X_spec shape:", X_spec.shape)  # (N, 301, 301, 3)
print("X_rgb  shape:", X_rgb.shape)   # (N, 301, 301, 3)
print("y shape     :", y.shape)

# Split (same indices for both inputs) 
Xspec_tr, Xspec_val, Xrgb_tr, Xrgb_val, y_tr, y_val = train_test_split(
    X_spec, X_rgb, y, test_size=VAL_SPLIT, random_state=SEED
)

# RESNET FEATURE EXTRACTOR 

backbone = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
for layer in backbone.layers:
    layer.trainable = False
for layer in backbone.layers[-7:]:  # fine-tune a bit
    layer.trainable = True

last_conv = backbone.get_layer("conv5_block3_out").output  # after res5c_branch2c
gap = GlobalAveragePooling2D(name="gap_from_res5c")(last_conv)
feature_extractor = Model(inputs=backbone.input, outputs=gap, name="resnet50_gap_extractor")

rgb_in  = Input(shape=(224,224,3), name="rgb_input")
spec_in  = Input(shape=(224,224,3), name="spectral_input")

feat_rgb = feature_extractor(rgb_in)
feat_spec = feature_extractor(spec_in)

# Merge
merged = Concatenate(name="concat_feats")([feat_spec, feat_rgb])

# # Separate backbones
# rgb_backbone  = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3), name="rgb_resnet")
# spec_backbone = ResNet50(weights=None,        include_top=False, input_shape=(224,224,3), name="spec_resnet")

# # Freeze most of RGB; fine-tune the very top
# for l in rgb_backbone.layers: l.trainable = False
# for l in rgb_backbone.layers[-7:]: l.trainable = True

# # Spectral: start conservative—freeze early blocks, learn higher ones
# for l in spec_backbone.layers: l.trainable = True
# for l in spec_backbone.layers[:100]:  # adjust cut as needed
#     l.trainable = False

# # Inputs
# spec_in = Input(shape=(224,224,3), name="spectral_input")
# rgb_in  = Input(shape=(224,224,3), name="rgb_input")


# # Features
# f_spec = GlobalAveragePooling2D(name="gap_spec")(spec_backbone(spec_in, training=True))
# f_rgb  = GlobalAveragePooling2D(name="gap_rgb")(rgb_backbone(rgb_in,   training=False))

#     # Merge
# merged = Concatenate(name="concat_feats")([f_spec, f_rgb])

# Regression head
x = Dense(256, activation="relu", name="fc1")(merged)
x = Dense(64, activation="relu", name="fc2")(x)
out = Dense(1, name="turbidity")(x)

model = Model(inputs=[spec_in, rgb_in], outputs=out, name="spec_rgb_fusion_regressor")
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# Train 
history = model.fit(
    {"spectral_input": Xspec_tr, "rgb_input": Xrgb_tr},  
    y_tr,
    validation_data=({"spectral_input": Xspec_val, "rgb_input": Xrgb_val}, y_val),  
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Predict
y_tr_pred  = model.predict({"spectral_input": Xspec_tr,  "rgb_input": Xrgb_tr }, verbose=0).ravel()
y_val_pred = model.predict({"spectral_input": Xspec_val, "rgb_input": Xrgb_val}, verbose=0).ravel()

# Metrics
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

rmse_tr = rmse(y_tr, y_tr_pred)
r2_tr   = r2_score(y_tr, y_tr_pred)
rmse_v  = rmse(y_val, y_val_pred)
r2_v    = r2_score(y_val, y_val_pred)

print(f"Training  RMSE: {rmse_tr:.4f}  R²: {r2_tr:.4f}")
print(f"Validation RMSE: {rmse_v:.4f}  R²: {r2_v:.4f}")

# Plot Pred vs True (val)
plt.figure(figsize=(5.5,5.5))
plt.scatter(y_val, y_val_pred, s=14, alpha=0.6)
lo = float(np.floor(min(y_val.min(),  y_val_pred.min())))
hi = float(np.ceil (max(y_val.max(),  y_val_pred.max())))
plt.plot([lo, hi], [lo, hi], lw=2)
plt.xlabel("True turbidity"); plt.ylabel("Predicted turbidity")
plt.title(f"Fusion Pred vs True  (RMSE={rmse_v:.2f}, R²={r2_v:.2f})")
plt.axis('equal'); plt.xlim(lo, hi); plt.ylim(lo, hi)
plt.grid(True, ls="--", alpha=0.35)
plt.tight_layout()
plt.savefig("pred_vs_true_fusion.png", dpi=300, bbox_inches="tight")
plt.close()