from ultralytics import YOLO


def yolo_train(data_yaml):
    # Load a model
    model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
    model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")  # build from YAML and transfer weights

    # Train the model
    model.train(
        data=data_yaml,
        epochs=150,
        imgsz=640,
        freeze=10,        # ★ 冻结 backbone
        lr0=1e-4,         # ★ 小学习率
        patience=30,
        workers=0
    )

def yolo_train_resume(path):
    model = YOLO(path)  # load a trained model "runs/segment/train2/weights/last.pt"
    results = model.train(resume=True)  # resume training from last.pt

if __name__ == "__main__":
    yolo_train("coco8-seg.yaml")

    