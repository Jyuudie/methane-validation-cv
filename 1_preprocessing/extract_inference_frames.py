import cv2
import os

# ================= CONFIGURATION =================
# Path to the new raw video you want to label
VIDEO_PATH = "data/raw_uploads/new_cow_video.mp4"

# Output folder for the images
OUTPUT_FOLDER = "data/inference_frames"

# Frame Skip: Save every Nth frame to avoid duplicate data
# (e.g., Set to 10 to label only 3 images per second instead of 30)
FRAME_SKIP = 5 
# =================================================

def extract_frames():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}")
        return

    frame_count = 0
    saved_count = 0
    print(f"Extracting frames from {VIDEO_PATH}...")

    while True:
        success, frame = cap.read()
        if not success:
            break 
        
        # Only save if we hit the skip interval
        if frame_count % FRAME_SKIP == 0:
            # Name matches CVAT convention (frame_000000.jpg)
            filename = f"frame_{frame_count:06d}.jpg"
            save_path = os.path.join(OUTPUT_FOLDER, filename)
            
            cv2.imwrite(save_path, frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"Success! Extracted {saved_count} images to '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    extract_frames()
