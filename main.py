import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

# Step 1: Paths and Parameters
data_dir = "D:\\data"  # Absolute path to your data directory
train_frames_dir = os.path.join(data_dir, "train_frames_2")  # Training frames directory
test_frames_dir = os.path.join(data_dir, "test_frames_2")    # Test frames directory
save_dir = os.path.join(data_dir, "prepared_data")         # Directory to save processed datasets

# Ensure save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")

print("Starting script...")

# Step 2: Load Pre-trained Model for Feature Extraction
print("Loading EfficientNetB0 for feature extraction...")
cnn_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
input_size = (224, 224)  # Explicitly set the expected input size

def extract_features(image_path, model):
    """
    Extract features from a single image using the provided model.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}. Skipping...")
            return None
        
        # Resize the image and preprocess it for the model
        image = cv2.resize(image, input_size)
        image = preprocess_input(np.expand_dims(image, axis=0))
        
        # Extract features using the pre-trained model
        features = model.predict(image).flatten()
        return features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Step 3: Prepare Dataset
def prepare_dataset(frame_dir, cnn_model, label_encoder):
    """
    Extract features and labels from the provided frame directory.
    """
    print(f"Extracting features from directory: {frame_dir}")
    X, y = [], []
    for label in tqdm(os.listdir(frame_dir), desc="Processing labels"):
        label_path = os.path.join(frame_dir, label)
        if os.path.isdir(label_path):
            for frame_file in tqdm(os.listdir(label_path), desc=f"Processing frames for {label}", leave=False):
                frame_path = os.path.join(label_path, frame_file)
                features = extract_features(frame_path, cnn_model)
                if features is not None:
                    X.append(features)
                    y.append(label)
    print(f"Number of samples extracted from {frame_dir}: {len(y)}")
    y = label_encoder.transform(y)
    return np.array(X), np.array(y)

# Step 4: Label Encoding
train_df = pd.read_csv("train.csv")  # Placeholder for your train metadata
label_encoder = LabelEncoder()
label_encoder.fit(train_df["label"].astype(str).unique())

# Extract features for train and test datasets
print("Preparing train dataset...")
X_train, y_train = prepare_dataset(train_frames_dir, cnn_model, label_encoder)

print("Preparing test dataset...")
X_test, y_test = prepare_dataset(test_frames_dir, cnn_model, label_encoder)

# Save the prepared datasets for reuse
np.save(os.path.join(save_dir, "X_train.npy"), X_train)
np.save(os.path.join(save_dir, "y_train.npy"), y_train)
np.save(os.path.join(save_dir, "X_test.npy"), X_test)
np.save(os.path.join(save_dir, "y_test.npy"), y_test)
print(f"Datasets prepared and saved in directory: {save_dir}")

# One-hot encode labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Step 5: Build and Train the Model
print("Building and training the model...")
model = Sequential([
    Dense(128, activation="relu", input_dim=X_train.shape[1]),
    Dense(64, activation="relu"),
    Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train_encoded, validation_data=(X_test, y_test_encoded), epochs=10, batch_size=32, verbose=1)

# Step 6: Evaluate the Model
print("Evaluating the model...")
y_pred = model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
y_test_labels = label_encoder.inverse_transform(np.argmax(y_test_encoded, axis=1))
print(classification_report(y_test_labels, y_pred_labels))

# Step 7: Save the Model and Label Encoder
print("Saving the model and label encoder...")
model.save(os.path.join(save_dir, "anomaly_detection_model.h5"))
with open(os.path.join(save_dir, "label_encoder.npy"), "wb") as f:
    np.save(f, label_encoder.classes_)

print("Model and label encoder saved successfully!")
