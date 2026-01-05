import cv2
import os
import tkinter as tk
from tkinter import filedialog

def extract_frames_every_n(video_path, step=5):
    # 获取视频文件名（不含扩展名）和所在目录
    video_dir = os.path.dirname(video_path)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    # 创建输出文件夹：视频名_拆分
    output_folder = os.path.join(video_dir, f"{video_basename}_split")
    os.makedirs(output_folder, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件！")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_count = 0
    current_frame = 0

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        # 每隔 n 帧保存一次（从第0帧开始，即第0,5,10,...帧）
        img_name = f"{video_basename}_{saved_count}.jpg"
        img_path = os.path.join(output_folder, img_name)
        cv2.imwrite(img_path, frame)
        saved_count += 1
        print(f"已完成 {current_frame} / {total_frames}  {current_frame/ total_frames * 100:.2f}%")

        current_frame += step

    cap.release()
    print(f"完成！共保存 {saved_count} 张图片到文件夹：{output_folder}")

def select_video_and_process():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    video_path = filedialog.askopenfilename(
        title="请选择一个MP4视频文件",
        filetypes=[("MP4 视频", "*.mp4"), ("所有文件", "*.*")]
    )
    
    if video_path:
        if not video_path.lower().endswith('.mp4'):
            print("警告：选择的文件不是 .mp4 格式，但仍尝试处理...")
        extract_frames_every_n(video_path, 60)
    else:
        print("未选择任何文件。")

if __name__ == "__main__":
    select_video_and_process()