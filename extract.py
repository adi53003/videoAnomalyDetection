import os
import cv2
import pandas as pd
from tqdm import tqdm

# Paths
train_csv = "train2.csv"  # Train CSV file
test_csv = "test2.csv"  # Test CSV file
train_frame_dir = "train_frames_2"  # Directory to store train frames
test_frame_dir = "test_frames_2"  # Directory to store test frames
os.makedirs(train_frame_dir, exist_ok=True)
os.makedirs(test_frame_dir, exist_ok=True)

# Load train and test data
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Clean the `video_name` column to remove "data/" prefix
train_df["video_name"] = train_df["video_name"].str.replace("data/", "", regex=False)
test_df["video_name"] = test_df["video_name"].str.replace("data/", "", regex=False)

# Function to extract frames
def extract_frames(video_path, output_dir, video_name, label, fps):
    """
    Extract frames from a video at the specified frames per second (fps).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Ensure the label directory exists
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    count = 0
    frame_count = 0
    success, frame = cap.read()
    
    with tqdm(total=total_frames, desc=f"Extracting frames from {video_name}") as pbar:
        while success:
            if count % frame_interval == 0:
                frame_name = f"{video_name}_frame{frame_count}.jpg"
                frame_path = os.path.join(label_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            success, frame = cap.read()
            count += 1
            pbar.update(1)
    
    cap.release()

# Frame extraction for train and test sets
def process_videos(data_df, output_dir, split_name):
    """
    Process videos to extract frames and store them in the corresponding split directory.
    """
    print(f"Processing {split_name} videos...")
    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc=f"Processing {split_name}"):
        video_path = row["video_name"]  # Video path directly from the cleaned column
        label = str(row["label"])
        video_name = os.path.splitext(os.path.basename(video_path))[0]  # Remove file extension

        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_path} not found!")
            continue

        # Set the FPS: 1 FPS for most labels, 1/3 FPS (every 3 seconds) for "normal"
        fps = 1 if label != "normal" else 1 / 3
        extract_frames(video_path, output_dir, video_name, label, fps)

# Process train and test videos
process_videos(train_df, train_frame_dir, "train")
process_videos(test_df, test_frame_dir, "test")

print("Frame extraction completed successfully!")
