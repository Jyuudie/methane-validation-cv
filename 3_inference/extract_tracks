import cv2
import csv
import os
from ultralytics import YOLO

# ================= CONFIGURATION =================
video_path = "data/raw_milking_session.mp4"
model_path = "models/yolov8_best.pt"
output_csv = "output/cow_tracks.csv"
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"
# =================================================

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Tracking cows in {total_frames} frames...")

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'track_id', 'class', 'conf', 'x', 'y', 'w', 'h'])

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run ByteTrack with persistence to maintain ID across frames
        results = model.track(frame, conf=0.25, persist=True, tracker="bytetrack.yaml", verbose=False)

        for box in results[0].boxes:
            if box.id is not None:
                track_id = int(box.id[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x, y, w, h = box.xywh[0].tolist()
                writer.writerow([frame_count, track_id, cls, conf, x, y, w, h])

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Tracking frame {frame_count}/{total_frames}...", end='\r')

print(f"\nDone! Track data saved to {output_csv}")
