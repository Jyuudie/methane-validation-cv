import cv2
import sys

# ================= CONFIGURATION =================
VIDEO_PATH = "data/raw_video.mp4"
# =================================================

def inspect_metadata():
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"--- Video Metadata ---")
    print(f"File: {VIDEO_PATH}")
    print(f"FPS: {fps:.2f}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")

    cap.release()

if __name__ == "__main__":
    inspect_metadata()
