from ultralytics import YOLO
import os
import torch
import yaml

def train():
    # Get the absolute path to the script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths using relative references
    dataset_dir = os.path.join(base_dir, "dataset")
    data_path = os.path.join(dataset_dir, "data.yaml")
    with open(data_path, encoding="utf-8") as data_file:
        dataset_config = yaml.safe_load(data_file)
    dataset_config["path"] = dataset_dir

    # Load YOLO model from scratch (no old weights)
    model_path = os.getenv("TRAIN_MODEL", "yolo11x.pt")
    model = YOLO(model_path)

    # Train the model
    model.train(
        data=dataset_config,
        epochs=120,
        imgsz=640,
        batch=8,
        device=os.getenv("TRAIN_DEVICE", "0" if torch.cuda.is_available() else "cpu"),
        lr0=0.0005,
        half=torch.cuda.is_available()
    )

if __name__ == "__main__":
    train()
