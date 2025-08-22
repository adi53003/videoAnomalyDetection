# 🎥 Video Anomaly Classification with GUI

## 📌 Overview
This project is a **Video Anomaly Classification System** that classifies videos as **Normal** or **Not Normal** using a deep learning model integrated with a stylish **PySide6 GUI application**.  

The system uses **EfficientNetB0** as a feature extractor and a **custom Sequential neural network** for classification.  
The GUI allows users to upload a video, play it, and view classification results with a progress bar and sensitivity controls.

---

## 📂 Dataset
- Source: **Kaggle Dataset (1700 videos)**
- **10 categories** of activities (e.g., Fighting, Accident, Walking, Running, Explosion, etc.)
- Each category mapped into:
  - **Normal**
  - **Not Normal**

---

## 🔄 Approach

### 1. **Frame Extraction**
- Extracted frames from videos (1 frame per second).

### 2. **Frame → Feature Vector**
- Passed each frame through **EfficientNetB0 (pre-trained CNN)**.
- Extracted a **feature vector** representing the important visual content.

### 3. **Sequential Model**
- Feature vector fed into a **Sequential Neural Network** with 5 Dense layers.
- Final layer: **10 nodes (categories)** with **Softmax activation**.

### 4. **Softmax Probabilities**
- Softmax converts raw outputs into a probability distribution.
- Example: `[0.01, 0.03, 0.90, ...]` → Predicted class is the one with highest probability.

### 5. **Normal vs Not Normal**
- Each predicted category is mapped into either:
  - **Normal**
  - **Not Normal**
- Final decision based on majority voting across frames.

---

## 🧠 Model Architecture

### **Feature Extractor**
- EfficientNetB0 (without top layer)
- Output: 1280-dimensional feature vector per frame

### **Sequential Model (Example)**
```python
model = Sequential([
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
```
## 🚀 How to Run
### 1. Clone Repo
```python
git clone https://github.com/your-username/video-anomaly-classification.git
cd video-anomaly-classification
```

### 2. Install Dependencies
```python
pip install -r requirements.txt
```
### 3. Run Application
```python
python main.py
