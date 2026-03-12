from ultralytics import YOLO


def yolo_train():
    # Load a model
    model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

    # 这个用于极小数据量，freeze设为10，学习率很小且固定
    # model.train(      
    #     data=data_yaml,
    #     epochs=150,
    #     imgsz=640,
    #     freeze=10,        # ★ 冻结 backbone
    #     lr0=1e-4,         # ★ 小学习率
    #     patience=30,
    #     workers=0
    # )
    model.train( # 新尝试的yaml设置训练
        data="datasets/block4/data.yaml",
        cfg="custom/scripts/yolo/train.yaml",
        name="grab_block"
    )

def yolo_train_resume(path):
    model = YOLO(path)  # load a trained model "runs/segment/train2/weights/last.pt"
    model.train(resume=True)  # resume training from last.pt

if __name__ == "__main__":
    yolo_train()

    