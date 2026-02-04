import cv2
import os
from ultralytics import YOLO

# ================= CONFIGURATION =================
# Use relative paths for portability
MODEL_PATH = "models/yolov8_best.pt"
IMAGE_PATH = "data/tuning_samples/messy_frame_09.jpg"
# =================================================

def run_tuner():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(IMAGE_PATH):
        print("Error: Model or Image not found. Check paths.")
        return

    model = YOLO(MODEL_PATH)
    original_img = cv2.imread(IMAGE_PATH)
    show_labels = True 

    def nothing(x): pass

    cv2.namedWindow('NMS Tuner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('NMS Tuner', 1200, 800)

    # Defaults: Conf=25%, IoU=70%
    cv2.createTrackbar('Confidence %', 'NMS Tuner', 25, 100, nothing)
    cv2.createTrackbar('NMS (IoU) %', 'NMS Tuner', 70, 100, nothing)

    print("Controls: Adjust sliders to tune. Press 'l' to toggle labels. 'q' to quit.")

    while True:
        # 1. Read Sliders
        conf_val = max(0.01, cv2.getTrackbarPos('Confidence %', 'NMS Tuner') / 100.0)
        iou_val = cv2.getTrackbarPos('NMS (IoU) %', 'NMS Tuner') / 100.0
        
        # 2. Inference
        results = model.predict(source=original_img, conf=conf_val, iou=iou_val, verbose=False)

        # 3. Visualization
        annotated_frame = results[0].plot(line_width=1, font_size=1, labels=show_labels, conf=show_labels)

        # 4. Dashboard Overlay
        cv2.rectangle(annotated_frame, (0, 0), (450, 80), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Conf: {conf_val:.2f} | IoU: {iou_val:.2f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Detections: {len(results[0].boxes)}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow('NMS Tuner', annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('l'): show_labels = not show_labels

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tuner()
