import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox, QFrame, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QFont, QCursor, QPixmap, QPainter, QColor
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

# Paths
model_path = "D:/data/prepared_data/high_accuracy_model.h5"  # Update the path if needed
label_encoder_path = "D:/data/prepared_data/label_encoder.npy"

# Load the model
print("Loading the model...")
model = load_model(model_path)

# Load the label encoder
print("Loading label encoder...")
label_classes = np.load(label_encoder_path, allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

# Dynamically find the label for "Normal"
normal_class = None
for idx, label in enumerate(label_encoder.classes_):
    if label.lower() == 'normal':  # Case-insensitive match
        normal_class = idx
        break

if normal_class is None:
    print("Error: 'Normal' label not found in label encoder.")
    sys.exit()

print(f"'Normal' class label is: {normal_class}")

# EfficientNetB0 model as feature extractor
base_model = EfficientNetB0(weights="imagenet", include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor_model = Model(inputs=base_model.input, outputs=x)


class VideoProcessor(QThread):
    """
    A QThread to process the video and classify it as 'Normal' or 'Not Normal'.
    """
    progress_updated = Signal(float)  # Signal to update progress bar
    classification_done = Signal(str)  # Signal to display the final result

    def __init__(self, video_path, sensitivity_mode, parent=None):
        super(VideoProcessor, self).__init__(parent)
        self.video_path = video_path
        self.sensitivity_mode = sensitivity_mode  # Sensitivity mode passed when the thread is created
        print(f"VideoProcessor initialized with sensitivity mode: {self.sensitivity_mode}")  # Debugging statement

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        predictions = []

        print(f"Processing video: {self.video_path}, FPS: {fps}, Total Frames: {total_frames}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process one frame per second
            if frame_count % int(fps) == 0:
                # Resize and preprocess the frame for the model
                frame_resized = cv2.resize(frame, (224, 224))
                frame_preprocessed = np.expand_dims(frame_resized, axis=0)
                frame_preprocessed = preprocess_input(frame_preprocessed)

                # Extract features using the feature extractor model
                features = feature_extractor_model.predict(frame_preprocessed)

                # Make prediction using the model
                y_pred = model.predict(features)
                y_pred_class = np.argmax(y_pred, axis=1)[0]

                # Classify as Normal (0) or Not Normal (1)
                if y_pred_class == normal_class:
                    predictions.append(0)  # Normal
                else:
                    predictions.append(1)  # Not Normal

                print(f"Frame {frame_count}: Prediction = {'Normal' if y_pred_class == normal_class else 'Not Normal'}")

            frame_count += 1

            # Update progress
            progress = (frame_count / total_frames) * 100
            self.progress_updated.emit(progress)  # Emit signal for progress update

        cap.release()

        # Adjust voting logic based on sensitivity mode
        weighted_predictions = np.array(predictions)
        if self.sensitivity_mode == "Low":
            not_normal_weight = 1
            video_class = "Not Normal" if np.sum(weighted_predictions * not_normal_weight) / (len(weighted_predictions) * not_normal_weight) > 0.5 else "Normal"
        else:
            not_normal_weight = 2
            video_class = "Not Normal" if np.sum(weighted_predictions * not_normal_weight) / (len(weighted_predictions) * not_normal_weight) > 0.01 else "Normal"

        self.classification_done.emit(video_class)  # Emit signal for classification result


class MainWindow(QMainWindow):
    """
    Main application window with a semi-transparent background image.
    """

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Video Classification")
        self.setStyleSheet("background-color: transparent; color: #ecf0f1;")
        self.sensitivity_mode = "High"  # Default mode is High Sensitivity

        # Set up background label
        self.background_label = QLabel(self)

        # Load and process the background image
        image_path = "image.jpg"  # Replace with your image file
        background_pixmap = QPixmap(image_path)

        # Apply stronger semi-transparency using QPainter
        painter = QPainter(background_pixmap)
        painter.fillRect(background_pixmap.rect(), QColor(0, 0, 0, 150))  # RGBA: 180 = stronger semi-transparency
        painter.end()

        self.background_label.setPixmap(background_pixmap)
        self.background_label.setScaledContents(True)
        self.background_label.setGeometry(0, 0, self.width(), self.height())
        self.background_label.lower()  # Ensure it stays below all other widgets

        #video player
        self.video_frame = QFrame(self)
        self.video_frame.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;  /* Dark grey background */
                border: 3px solid #3498db;  /* Blue border */
                border-radius: 12px;       /* Rounded corners */
            }
        """)
        self.video_frame.setFixedSize(480, 350)  # Slightly larger than the video widget for padding

        # Add shadow effect to the QFrame
        shadow_effect = QGraphicsDropShadowEffect(self)
        shadow_effect.setBlurRadius(20)               # Increase for softer edges
        shadow_effect.setOffset(0, 0)                 # Shadow offset (x, y)
        shadow_effect.setColor(QColor(0, 0, 0, 200))  # Black shadow with transparency
        self.video_frame.setGraphicsEffect(shadow_effect)

        self.video_widget = QVideoWidget(self.video_frame)
        self.video_widget.setGeometry(6, 6, 468, 338)  # Leave space for the border

        self.video_player = QMediaPlayer(self)
        self.video_player.setVideoOutput(self.video_widget)

        # Widgets
        self.label = QLabel("Select a video to classify", self)
        self.label.setFont(QFont("Helvetica", 24, QFont.Bold))
        self.label.setAlignment(Qt.AlignCenter)

        self.select_button = QPushButton("Select Video", self)
        self.select_button.setFont(QFont("Helvetica", 16))
        self.select_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 15px;
                padding: 15px;
                max-width: 300px;
            }
            QPushButton:hover {
                background-color: #1abc9c;
            }
        """)
        self.select_button.clicked.connect(self.open_file_dialog)

        self.sensitivity_label = QLabel(f"Mode: {self.sensitivity_mode} Sensitivity", self)
        self.sensitivity_label.setFont(QFont("Helvetica", 16))
        self.sensitivity_label.setAlignment(Qt.AlignCenter)

        self.toggle_button = QPushButton("Switch", self)
        self.toggle_button.setFont(QFont("Helvetica", 16))
        self.toggle_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 15px;
                padding: 10px 15px;
                max-width: 150px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_sensitivity)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #34495e;
                border-radius: 15px;
                text-align: center;
                font: bold 14px;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #1abc9c;
                border-radius: 15px;
            }
        """)
        self.progress_bar.setValue(0)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addSpacing(10)
        # Replace this line in the layout setup
        layout.addWidget(self.video_frame, alignment=Qt.AlignCenter)
        layout.addSpacing(5)
        button_layout2 = QHBoxLayout()
        button_layout2.addWidget(self.select_button, alignment=Qt.AlignCenter)
        layout.addLayout(button_layout2)
        layout.addSpacing(30)
        layout.addWidget(self.sensitivity_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.toggle_button, alignment=Qt.AlignCenter)
        layout.addLayout(button_layout)
        layout.addSpacing(30)
        layout.addWidget(self.progress_bar)
        layout.setAlignment(Qt.AlignCenter)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Set window to fullscreen
        self.showFullScreen()

    def resizeEvent(self, event):
        """
        Update the background label size when the window is resized.
        """
        self.background_label.setGeometry(0, 0, self.width(), self.height())
        super(MainWindow, self).resizeEvent(event)

    def open_file_dialog(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4);;All Files (*)")
        if video_path:
            self.video_player.setSource(video_path)
            self.video_player.play()
            self.start_video_processing(video_path)

    def start_video_processing(self, video_path):
        # Print the sensitivity mode to ensure it's being passed correctly
        print(f"Starting video processing with sensitivity mode: {self.sensitivity_mode}")
        
        # Recreate the VideoProcessor instance with the updated sensitivity mode
        self.processor = VideoProcessor(video_path, self.sensitivity_mode)
        self.processor.progress_updated.connect(self.update_progress_bar)
        self.processor.classification_done.connect(self.show_result)
        self.processor.start()

    def update_progress_bar(self, value):
        self.progress_bar.setValue(int(value))

    def show_result(self, result):
        # Create a custom QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("Video Classification")
        
        if result == "Normal":
            msg.setText("The video is classified as: Normal")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #2ecc71; /* Green */
                    color: white;
                    min-width: 500px;
                    min-height: 350px;
                    padding: 20px;
                }
                QMessageBox QLabel {
                    background: transparent;
                    color: white;
                    font-size: 22px;
                }
                QMessageBox QPushButton {
                    background-color: #1abc9c;
                    color: white;
                    border-radius: 15px;
                    padding: 20px;
                }
                QMessageBox QPushButton:hover {
                    background-color: #16a085;
                }
            """)
        else:
            msg.setText("The video is classified as: Not Normal")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #e74c3c; /* Red */
                    color: white;
                    min-width: 500px;
                    min-height: 350px;
                    padding: 20px;
                }
                QMessageBox QLabel {
                    background: transparent;
                    color: white;
                    font-size: 22px;
                }
                QMessageBox QPushButton {
                    background-color: #c0392b;
                    color: white;
                    border-radius: 15px;
                    padding: 20px;
                }
                QMessageBox QPushButton:hover {
                    background-color: #e74c3c;
                }
            """)
        
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        
        # Adjust the font for the button
        for button in msg.buttons():
            button.setFont(QFont("Helvetica", 16, QFont.Bold))

        msg.resize(500, 350)  # Make the message box much bigger
        msg.exec()

    def toggle_sensitivity(self):
        """
        Toggle sensitivity mode between High and Low.
        """
        if self.sensitivity_mode == "High":
            self.sensitivity_mode = "Low"
            self.sensitivity_label.setText(f"Mode: {self.sensitivity_mode} Sensitivity")
            print("Low sensitivity mode enabled")
        else:
            self.sensitivity_mode = "High"
            self.sensitivity_label.setText(f"Mode: {self.sensitivity_mode} Sensitivity")
            print("High sensitivity mode enabled")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
