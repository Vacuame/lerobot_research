from ultralytics import YOLO


def yolo_train(data_yaml):
    # Load a model
    model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
    model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")  # build from YAML and transfer weights

    # Train the model
    # results = model.train(data=data_path, epochs=100, imgsz=640) # workers太高了我的电脑会崩溃
    # Train the model
    model.train(
        data=data_yaml,
        epochs=200,
        imgsz=640,
        freeze=10,        # ★ 冻结 backbone
        lr0=1e-4,         # ★ 小学习率
        patience=10
        # name="train1"  # 结果保存路径为 runs/segment/train8
    )

def yolo_train_resume(path):
    model = YOLO(path)  # load a trained model "runs/segment/train2/weights/last.pt"
    results = model.train(resume=True)  # resume training from last.pt

if __name__ == "__main__":
    yolo_train("datasets/block2/data.yaml")

    