import os
import cv2
from ultralytics import YOLO

# Fix for GoPro/High-Res Video Timeouts
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "20000"

# ================= CONFIGURATION =================
model_path = "models/yolov8_best.pt"
video_path = "data/test_video.mp4"
# =================================================

print(f"Loading Model from {model_path}...")
model = YOLO(model_path)

# Run inference with stream=True to prevent memory crashes
print(f"Processing video: {video_path}")
results = model.predict(
    source=video_path,
    line_width=2,
    save=True,        # Saves the output video
    conf=0.5,
    imgsz=640,
    stream=True
)

for i, r in enumerate(results):
    if i % 100 == 0:
        print(f"Processed Frame {i}...", end='\r')

print("\nVideo processing complete.")
