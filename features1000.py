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
from scipy.io import savemat  # For saving .mat file

# --- PARAMETERS ---
IMAGE_DIR = "/home/bmt.lamar.edu/bgautam3/croppedImage2"
CSV_FILE = "target_turbidity.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# --- LOAD TURBIDITY ONLY ---
df = pd.read_csv(CSV_FILE,header=None)
#df.dropna(inplace=True)
print(df.describe())

# --- GET IMAGE FILENAMES ---
image_filenames = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

df["filename"] = image_filenames # Ensure the filenames are sorted


# --- LOAD IMAGES AND LABELS ---
def load_images_and_labels(df):
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(IMAGE_DIR, row["filename"])
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=IMAGE_SIZE)
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(row[0])  # Assuming the first column is the label
    return np.array(images), np.array(labels)

X, y = load_images_and_labels(df)

# --- TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LOAD BASE MODEL WITH ORIGINAL DENSE LAYERS ---
vgg_model = VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
vgg_model.trainable = True # Ensure the model is trainable(fine tuning)
base_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('predictions').output)
print(X.shape)

# --- CREATE FEATURE EXTRACTOR FOR 1000-D OUTPUT ---
feature1000_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('predictions').output)


# --- EXTRACT FEATURES ---
features = feature1000_model.predict(X, batch_size=1, verbose=1)

# # --- MODIFY FINAL OUTPUT LAYER FOR REGRESSION ---
# x = vgg_model.get_layer('fc2').output  # Get output of last Dense before softmax
# x = Dense(1, activation='linear', name='regression_output')(x)  # Your custom output
# model = Model(inputs=vgg_model.input, outputs=x)

# # --- FREEZE ALL BUT DENSE LAYERS (fc1, fc2, regression_output) ---
# for layer in vgg_model.layers:
#     if layer.name in ['fc1', 'fc2']:
#         layer.trainable = True
#     else:
#         layer.trainable = False

# # --- EXTRACT 1000-D OUTPUT FROM ORIGINAL VGG16 (BEFORE SOFTMAX) ---
# feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('predictions').output)
# logits_1000 = feature_extractor.predict(X)  # Shape: (num_samples, 1000)

# # --- SAVE TO .MAT FILE FOR MATLAB USE ---
savemat("vgg16_1000_logits.mat", {"vgg16_logits": features})
print(len(X));

# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR

# # Example 1: Linear Regression
# lr = LinearRegression()
# lr.fit(logits_1000, y_test)
# y_pred_lr = lr.predict(logits_1000)

# rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
# r2_lr = r2_score(y_test, y_pred_lr)

# print("\nLinear Regression using 1000-d VGG16 logits:")
# print(f"RMSE: {rmse_lr:.2f}")
# print(f"R^2: {r2_lr:.2f}")

# # Example 2: Random Forest Regression (optional, for comparison)
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(logits_1000, y_test)
# y_pred_rf = rf.predict(logits_1000)

# rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
# r2_rf = r2_score(y_test, y_pred_rf)

# print("\nRandom Forest Regression using 1000-d VGG16 logits:")
# print(f"RMSE: {rmse_rf:.2f}")
# print(f"R^2: {r2_rf:.2f}")
