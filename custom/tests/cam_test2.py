import cv2
import os

def capture_from_camera(index, width=640, height=480, output_name="camera"):
    print(f"尝试打开摄像头 {index}...")
    
    # 使用 Media Foundation 后端（Windows 更稳定）
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"❌ 摄像头 {index} 无法打开！")
        return False

    # 设置 MJPEG 格式（大幅降低带宽）
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    # 设置分辨率和帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 读几帧让摄像头稳定（部分摄像头需要）
    for _ in range(5):
        ret, frame = cap.read()
    
    ret, frame = cap.read()
    if not ret:
        print(f"❌ 无法从摄像头 {index} 读取画面")
        cap.release()
        return False

    # 保存图像
    filename = f"{output_name}_{index}.jpg"
    cv2.imwrite(filename, frame)
    print(f"✅ 摄像头 {index} 成功拍照，已保存为: {filename}")

    cap.release()
    return True

def main():
    print("=== 双摄像头拍照测试 (320x240) ===")
    
    success0 = capture_from_camera(0, width=320, height=240, output_name="cam")
    success1 = capture_from_camera(1, width=320, height=240, output_name="cam")

    if success0 or success1:
        print("\n📸 拍照完成！")
        # 可选：自动打开图片（Windows）
        # os.startfile("cam_0.jpg") if success0 else None
        # os.startfile("cam_1.jpg") if success1 else None
    else:
        print("\n❌ 两个摄像头都无法使用！")

if __name__ == "__main__":
    main()