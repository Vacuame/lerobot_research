import cv2
import time

def open_camera(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_MSMF)

    # 强制分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 强制使用 YUYV（不压缩，吃带宽）
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

    # 尝试设 30 FPS（不一定生效，但无所谓）
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {cam_id}")

    return cap


caps = [open_camera(0), open_camera(1)]

frame_count = [0, 0]
start_time = time.time()

while True:
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            frame_count[i] += 1
            cv2.imshow(f"Camera {i}", frame)

    # 每 2 秒统计一次 FPS
    elapsed = time.time() - start_time
    if elapsed >= 2:
        fps0 = frame_count[0] / elapsed
        fps1 = frame_count[1] / elapsed
        print(f"FPS -> Cam0: {fps0:.1f}, Cam1: {fps1:.1f}")

        frame_count = [0, 0]
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
