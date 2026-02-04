from ultralytics import YOLO, RTDETR
import torch

def main():
    # Check if GPU is available
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device.upper()}")

    # Path to data.yaml file
    data_path = r"C:\dataset\data.yaml"

    # ==========================================
    # MODEL 1: YOLOv8 
    # ==========================================
    print("\n--- Starting YOLOv8 Training ---")

    # Load a pre-trained model
    model_yolo = YOLO('yolov8n.pt') 

    # Train
    results_yolo = model_yolo.train(
        data=data_path,
        epochs=100,           
        patience=15,          
        imgsz=640,
        batch=16,             
        device=device,
        project='cow_project',
        name='yolov8_run',
        dropout=0.2,
        workers=8             
    )

    # ==========================================
    # MODEL 2: RT-DETR 
    # ==========================================
    print("\n--- Starting RT-DETR Training ---")

    model_detr = RTDETR('rtdetr-l.pt') 

    results_detr = model_detr.train(
        data=data_path,
        epochs=100,
        patience=15,
        imgsz=640,
        batch=4,
        device=device,
        project='cow_project',
        name='rtdetr_run'
    )

    print("All training finished!")

if __name__ == '__main__':
    main()
