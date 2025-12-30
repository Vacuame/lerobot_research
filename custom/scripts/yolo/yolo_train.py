from ultralytics import YOLO


def yolo_train(data_path):
    # Load a model
    model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
    model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=data_path, epochs=100, imgsz=640,workers=0,) # workers太高了我的电脑会崩溃

def yolo_train_resume(path):
    model = YOLO(path)  # load a trained model "runs/segment/train2/weights/last.pt"
    results = model.train(resume=True)  # resume training from last.pt

if __name__ == "__main__":
    yolo_train("datasets/R2P2/data.yaml")

    