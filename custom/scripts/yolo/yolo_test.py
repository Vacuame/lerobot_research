from ultralytics import YOLO
import cv2
import numpy as np

def print_label_names(model_path="yolov8n-seg.pt"):
    model = YOLO(model_path)
    names = model.names
    print(names)

def draw_results_on_frame(model,frame,results):
    h, w = frame.shape[:2]
    
    for r in results:
        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for i, mask in enumerate(masks):
            # resize mask 到原图尺寸
            mask = cv2.resize(mask, (w, h))
            mask = (mask > 0.5).astype(np.uint8) * 255  # 二值化掩码

            # 创建彩色掩码（绿色）
            color = np.zeros_like(frame)
            color[:, :, 2] = mask  # 绿色通道

            # 将掩码叠加到原图上（半透明）
            frame = cv2.addWeighted(frame, 1.0, color, 0.4, 0)

            # 绘制边界框和标签
            x1, y1, x2, y2 = boxes[i].astype(int)
            label = f"id:{r.boxes.id[i] if r.boxes.id is not None else ''}, {model.names[int(classes[i])]}, {r.boxes.conf[i]:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2
            )
    return frame

def yolo_seg_camera(
    model_path="yolov8n-seg.pt",
    cam_id=0
):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False)
        frame = draw_results_on_frame(model,frame,results)

        cv2.imshow("YOLOv8 Seg Camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
            break

    cap.release()
    cv2.destroyAllWindows()

def yolo_seg_picture(
    model_path="yolov8n-seg.pt",
    picture_path="custom/tests/yolo_test/test.jpg"
):
    # 加载模型
    model = YOLO(model_path)
    model.track()

    # 读取图片
    frame = cv2.imread(picture_path)
    if frame is None:
        print(f"Error: Could not read image from {picture_path}")
        return

    # YOLO 推理（直接喂 numpy）
    results = model(frame, verbose=False)

    frame = draw_results_on_frame(model,frame,results)

    # 显示结果
    cv2.imshow("YOLOv8 Segmentation Result", frame)
    cv2.waitKey(0)  # 按任意键关闭窗口
    cv2.destroyAllWindows()

from yolo_stable_id import StableObjectManager
def yolo_seg_camera_stable(
    model_path="yolov8n-seg.pt",
    cam_id=0
):
    model = YOLO(model_path)
    tracker = StableObjectManager(
        iou_thresh=0.6,
        max_missing=5
    )

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            verbose=False
        )

        r = results[0]
        if r.masks is None:
            cv2.imshow("YOLOv8 Seg Camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # 1️⃣ 取出 masks（转为 bool numpy）
        masks = r.masks.data.cpu().numpy() > 0.5

        # 2️⃣ 更新 stable_id
        stable_ids = tracker.update(list(masks))

        # 3️⃣ 画结果
        for mask, sid in zip(masks, stable_ids):
            color = (0, 255, 0)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue

            cx, cy = xs.mean().astype(int), ys.mean().astype(int)
            frame[mask] = frame[mask] * 0.5 + np.array(color) * 0.5
            cv2.putText(
                frame,
                f"stable_id={sid}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.imshow("YOLOv8 Seg Camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    yolo_seg_camera_stable("yolov8n-seg.pt")

    #yolo_seg_picture(picture_path="custom/scripts/yolo/image/sheep.jpg")

    #print_label_names()


