import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import ffmpeg
from pathlib import Path
import argparse
import os


# 剪裁选定帧
def cut_video_by_frames_ffmpeg(input_path, output_path, from_frame, to_frame):
    """
    基于帧数裁剪视频
    """
    # 计算起始时间和持续时间（需要知道原视频的FPS）
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    start_time = from_frame / fps
    duration = (to_frame - from_frame) / fps
    (
        ffmpeg
        .input(input_path, ss=start_time)
        .output(output_path, t=duration, c='copy')
        .overwrite_output()
        .global_args('-loglevel', 'error')  # 只显示错误信息
        .run()
    )
    print(f"输出文件: {output_path}\n")

# 编译选定帧，太慢了，不用这个，丢弃
'''
def save_all_frames_as_video(dataloader, output_path, fps=30):
    # 从 dataloader 中获取所有帧并保存为视频
    all_frames = []
    import tqdm
    # 使用 tqdm 显示进度
    episode = tqdm.tqdm(dataloader, total=len(dataloader))
    
    for batch in episode:
        # 假设 batch[0] 是图像数据
        # batch 的形状通常是 [batch_size, channels, height, width]
        batch_data = batch['observation.images.front']  # 获取图像数据
        
        # 将批次数据转换为单个帧列表
        for i in range(batch_data.shape[0]):  # 遍历批次中的每个样本
            frame = batch_data[i]  # 获取第 i 个帧 [channels, height, width]
            
            # 如果是 tensor，转换为 numpy
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            
            # 如果是 CHW 格式，转换为 HWC
            if len(frame.shape) == 3 and frame.shape[0] <= 4:  # CHW 格式
                frame = np.transpose(frame, (1, 2, 0))
            
            # 确保像素值在 0-255 范围内
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:  # 如果是 0-1 范围
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            all_frames.append(frame)
    
    if len(all_frames) == 0:
        print("没有获取到任何帧！")
        return
    
    # 获取帧的尺寸
    height, width = all_frames[0].shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in all_frames:
        # 确保帧尺寸一致
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        # 如果是 RGB，转换为 BGR
        if frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        
        out.write(frame_bgr)
    
    out.release()
    print(f"视频已保存到: {output_path}, 总共 {len(all_frames)} 帧")
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")

    # repo_id='Vacuame/triple_data'
    dataset = LeRobotDataset(repo_id)
    repo_name = repo_id.split('/')[-1]
    output_folder_path = f'C:/dataset_videos/{repo_name}'

    for ep_idx in range(dataset.meta.info['total_episodes']):
        for vid_key in dataset.meta.video_keys:
            video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, vid_key)
            from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
            to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
            print(f'文件路径：{video_path}')
            print(f'帧数范围：[{from_idx}, {to_idx})')
            vid_name = vid_key.split('.')[-1]
            output_path = f'{output_folder_path}/{ep_idx}_{vid_name}.mp4'
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cut_video_by_frames_ffmpeg(video_path,output_path, from_idx, to_idx)
    
    os.startfile(output_folder_path)

    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     num_workers=0,
    #     batch_size=1,
    #     sampler=episode_sampler,
    # )
    #save_all_frames_as_video(dataloader,'C:/Mine/asd23.mp4')