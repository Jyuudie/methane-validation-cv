from ultralytics import RTDETR
import cv2

# ================= CONFIGURATION =================
# Path to your Transformer Model
MODEL_PATH = "models/rtdetr_run2/weights/best.pt"
VIDEO_PATH = "data/test_video.mp4"
# =================================================

def run_transformer():
    # Load the RT-DETR model (Vision Transformer)
    model = RTDETR(MODEL_PATH)
    
    # Run inference
    results = model.predict(
        source=VIDEO_PATH,
        save=True,
        conf=0.25,
        iou=0.45,
        stream=True # Use stream to manage memory
    )
    
    print("Running RT-DETR Inference...")
    for r in results:
        pass # Process stream
    print("Done! Transformer results saved.")

if __name__ == "__main__":
    run_transformer()
