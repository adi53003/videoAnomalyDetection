import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Paths
prepared_data_dir = "D:\\data\\prepared_data"
x_train_path = os.path.join(prepared_data_dir, "X_train.npy")
y_train_path = os.path.join(prepared_data_dir, "y_train.npy")
x_test_path = os.path.join(prepared_data_dir, "X_test.npy")
y_test_path = os.path.join(prepared_data_dir, "y_test.npy")
label_encoder_path = os.path.join(prepared_data_dir, "label_encoder.npy")
model_save_path = os.path.join(prepared_data_dir, "high_accuracy_model.h5")

# Step 1: Load datasets
print("Loading prepared datasets...")
X_train = np.load(x_train_path)
y_train = np.load(y_train_path)
X_test = np.load(x_test_path)
y_test = np.load(y_test_path)

# Step 2: Ensure label encoder exists or recreate it
try:
    print("Loading label encoder...")
    label_classes = np.load(label_encoder_path, allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_classes
except (FileNotFoundError, ValueError) as e:
    print(f"Error with label encoder: {e}. Recreating a new one...")
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    np.save(label_encoder_path, label_encoder.classes_)
    print(f"Label encoder saved at: {label_encoder_path}")

# One-hot encode labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Step 3: Build the Enhanced Model
print("Building the enhanced model...")
model = Sequential([
    Dense(1024, activation="relu", input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 4: Train the Model
print("Training the model...")
history = model.fit(
    X_train, y_train_encoded,
    validation_data=(X_test, y_test_encoded),
    epochs=30,
    batch_size=64,
    verbose=1
)

# Step 5: Save the Model
print("Saving the trained model...")
model.save(model_save_path)
print(f"Model saved at: {model_save_path}")

# Step 6: Evaluate the Model
print("Evaluating the model...")
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Classification Report
print("Classification Report:")
report = classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_)
print(report)
