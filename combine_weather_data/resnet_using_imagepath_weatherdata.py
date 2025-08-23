import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate, Average
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Pick GPU 1 if GPU 0 is crowded
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # or leave unset if 0 is free

# Allow TF to grow GPU memory instead of preallocating all
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

# Optionally disable XLA JIT to avoid the strict autotuner path
tf.config.optimizer.set_jit(False)  # comment out if you want JIT

# ======================
# CONFIG — EDIT THESE
# ======================
CSV_B27 = "/home/bmt.lamar.edu/bgautam3/deep neural network/combine_weather_data/SB_HS_turbidity_mapping_band27.csv"   # cols: path, filename, turbidity
CSV_B29 = "/home/bmt.lamar.edu/bgautam3/deep neural network/combine_weather_data/SB_HS_turbidity_mapping_band29.csv"   # cols: path, filename, turbidity
CSV_B31 = "/home/bmt.lamar.edu/bgautam3/deep neural network/combine_weather_data/SB_HS_turbidity_mapping_band31.csv"   # cols: path, filename, turbidity

# RGB input: choose ONE
RGB_FROM_DIR = True                 # True -> build rgb_path as RGB_DIR/filename
RGB_DIR = "/home/bmt.lamar.edu/bgautam3/deep neural network/combine_weather_data/Image_turbidity_map_saltwater_july/"
RGB_CSV = None                      # if using a CSV instead: cols: filename, rgb_path

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8 
EPOCHS = 20
VAL_SPLIT = 0.2
STRICT_TURB_CHECK = True            # set False to skip the equality check

# ======================
# LOAD & MERGE BANDS
# ======================
def need(cols, df_):
    missing = set(cols) - set(df_.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

b27 = pd.read_csv(CSV_B27)
b29 = pd.read_csv(CSV_B29)
b31 = pd.read_csv(CSV_B31)
for d in (b27, b29, b31):
    need(["path", "filename", "turbidity"], d)

b27 = b27.rename(columns={"path": "spec27", "turbidity": "turb27"})
b29 = b29.rename(columns={"path": "spec29", "turbidity": "turb29"})
b31 = b31.rename(columns={"path": "spec31", "turbidity": "turb31"})

# inner join by filename
df = b27.merge(b29[["filename", "spec29", "turb29"]], on="filename", how="inner") \
        .merge(b31[["filename", "spec31", "turb31"]], on="filename", how="inner")

# ✅ Turbidity: take from Band_27;
if STRICT_TURB_CHECK:
    if not (np.allclose(df["turb27"], df["turb29"]) and np.allclose(df["turb27"], df["turb31"])):
        raise ValueError("Turbidity differs across band CSVs for some filenames.")
df["turbidity"] = df["turb27"].astype(float)


# ADD RGB PATHS

if RGB_FROM_DIR:
    df["rgb_path"] = df["filename"].apply(lambda fn: os.path.join(RGB_DIR, fn))
else:
    if RGB_CSV is None:
        raise ValueError("Provide RGB_CSV or set RGB_FROM_DIR=True with RGB_DIR.")
    rgbdf = pd.read_csv(RGB_CSV)  # filename, rgb_path
    need(["filename", "rgb_path"], rgbdf)
    df = df.merge(rgbdf[["filename", "rgb_path"]], on="filename", how="inner")


# FILTER: to ensure all the files exist (3 spectral and 1 rgb)

def exists(p): 
    return isinstance(p, str) and os.path.isfile(p)

before = len(df)
df = df[df["rgb_path"].apply(exists) &
        df["spec27"].apply(exists) &
        df["spec29"].apply(exists) &
        df["spec31"].apply(exists)].reset_index(drop=True)
print(f"Kept {len(df)}/{before} rows after file-existence filtering.")
if len(df) == 0:
    raise RuntimeError("No rows left after filtering. Check paths.")

print(df.head(50))
#df.to_csv("merged_rgb_spec_turbidity_paths.csv", index=False)


# TRAIN/VAL SPLIT

idx = np.arange(len(df))
train_idx, val_idx = train_test_split(idx, test_size=VAL_SPLIT, random_state=42)


# IMAGE IO

def load_img_arr(p):
    img = load_img(p, target_size=IMAGE_SIZE)
    return img_to_array(img) / 255.0

def stack(paths):
    return np.stack([load_img_arr(p) for p in paths], axis=0)

Xrgb_train = stack(df.loc[train_idx, "rgb_path"].values)
Xrgb_val   = stack(df.loc[val_idx,   "rgb_path"].values)

Xs27_train = stack(df.loc[train_idx, "spec27"].values)
Xs27_val   = stack(df.loc[val_idx,   "spec27"].values)

Xs29_train = stack(df.loc[train_idx, "spec29"].values)
Xs29_val   = stack(df.loc[val_idx,   "spec29"].values)

Xs31_train = stack(df.loc[train_idx, "spec31"].values)
Xs31_val   = stack(df.loc[val_idx,   "spec31"].values)

y_train = df.loc[train_idx, "turbidity"].astype("float32").values
y_val   = df.loc[val_idx,   "turbidity"].astype("float32").values


# RESNET FEATURE EXTRACTOR @ conv5_block3_out -> GAP

backbone = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
for layer in backbone.layers:
    layer.trainable = False
for layer in backbone.layers[-2:]:  # fine-tune a bit
    layer.trainable = True

last_conv = backbone.get_layer("conv5_block3_out").output  # after res5c_branch2c
gap = GlobalAveragePooling2D(name="gap_from_res5c")(last_conv)
feature_extractor = Model(inputs=backbone.input, outputs=gap, name="resnet50_gap_extractor")


# FUSION: RGB + average(spectral)

rgb_in  = Input(shape=(224,224,3), name="rgb_input")
b27_in  = Input(shape=(224,224,3), name="band27_input")
b29_in  = Input(shape=(224,224,3), name="band29_input")
b31_in  = Input(shape=(224,224,3), name="band31_input")

rgb_feat = feature_extractor(rgb_in)
b27_feat = feature_extractor(b27_in)
b29_feat = feature_extractor(b29_in)
b31_feat = feature_extractor(b31_in)

spec_feat = Concatenate(name="spec_avg")([b27_feat, b29_feat, b31_feat])  # 2048-d
# spec_feat = Concatenate(name="spec_avg")([b27_feat, b29_feat, b31_feat])
# fused= spec_feat
fused = Concatenate(name="fused_feats")([rgb_feat, spec_feat])        # 4096-d

# x = Dense(1080, activation="relu", name="reg_dense_1080")(fused)
# x = Dense(256,  activation="relu", name="reg_dense_256")(x)
x = Dense(64,  activation="relu", name="reg_dense_64")(fused)
out = Dense(1, name="turbidity")(x)

model = Model(inputs=[ rgb_in,b27_in, b29_in, b31_in], outputs=out, name="rgb_spec_fusion_regressor")
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
model.summary()


# TRAIN

history = model.fit(
   [Xrgb_train, Xs27_train, Xs29_train, Xs31_train], y_train,
    # [ Xs27_train, Xs29_train, Xs31_train], y_train,
    validation_data=([Xrgb_val, Xs27_val, Xs29_val, Xs31_val], y_val),
    # validation_data=([ Xs27_val, Xs29_val, Xs31_val], y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)


# EVAL

y_train_pred = model.predict([Xrgb_train, Xs27_train, Xs29_train, Xs31_train]).flatten()
# y_train_pred = model.predict([ Xs27_train, Xs29_train, Xs31_train]).flatten()
y_val_pred   = model.predict([Xrgb_val,   Xs27_val,   Xs29_val,   Xs31_val]).flatten()
# y_val_pred   = model.predict([   Xs27_val,   Xs29_val,   Xs31_val]).flatten()

rmse = lambda y, yp: float(np.sqrt(mean_squared_error(y, yp)))
print(f"Training RMSE: {rmse(y_train, y_train_pred):.4f}  R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"Validation RMSE: {rmse(y_val,   y_val_pred):.4f}  R²: {r2_score(y_val,   y_val_pred):.4f}")


# PLOT

plt.figure(figsize=(8,6))
plt.scatter(y_val, y_val_pred, alpha=0.7, edgecolor="k")
mn, mx = float(np.min(y_val)), float(np.max(y_val))
plt.plot([mn, mx], [mn, mx])
plt.xlabel("True Turbidity")
plt.ylabel("Predicted Turbidity")
plt.title("RGB + (Band27,29,31 avg) — Predicted vs True")
plt.grid(True)
plt.tight_layout()
plt.savefig("fusion_resnet_pred_vs_true_turbidity.png", dpi=300)
plt.show()
