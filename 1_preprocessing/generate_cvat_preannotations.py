from ultralytics import YOLO
import os

# ================= CONFIGURATION =================
# Path to your best current model
MODEL_PATH = "models/yolov8_best.pt"

# Folder of new, raw images you want to auto-label
SOURCE_IMAGES = "data/raw_uploads"

# Where to save the text files for CVAT import
PROJECT_NAME = "auto_label_results"
# =================================================

def generate_pre_labels():
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    print(f"Generating pre-annotations for images in {SOURCE_IMAGES}...")
    
    # Run inference with save_txt=True
    # This creates the .txt files that CVAT can import as "Pre-annotations"
    model.predict(
        source=SOURCE_IMAGES,
        save_txt=True,       # Critical for CVAT
        save_conf=False,     # CVAT doesn't need the confidence score line
        conf=0.25,           # Lower threshold to catch difficult cows
        project=PROJECT_NAME,
        name="cvat_export"
    )

    print(f"Done! Upload the .txt files in '{PROJECT_NAME}/cvat_export/labels' to CVAT.")

if __name__ == "__main__":
    generate_pre_labels()
