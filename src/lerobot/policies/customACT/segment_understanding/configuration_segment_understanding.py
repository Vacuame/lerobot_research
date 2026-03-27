
from dataclasses import dataclass,field


#TODO 最好还可以定义 use_fk: bool 之类的选项
#TODO num_classes和output_dim改为动态获取，不记在配置里
@dataclass
class SegmentUnderstandingConfig:
    # YOLO 
    yolo_path: str = "runs/segment/grab_block/weights/best.pt"
    tracker_path: str = "custom/scripts/yolo/botsort.yaml"
    camera_name: str = "robot1"
    max_yolo_objects: int = 20  # YOLO 检测的最大物体数
    num_classes: int = 4  # YOLO 类别数（动态赋值）

    # FK
    urdf_path = "custom/config/SO101/so101_new_calib.urdf"
    ee_frame_name = "gripper_frame_link"

    # ==============================
    # R (YOLO detection set) 编码配置
    # ==============================
    cls_embed_dim: int = 64      # 类别 embedding 维度
    r_hidden_dim: int = 128      # numeric encoder 中间层 & 输出维度
    r_token_dim: int = 256       # per-object token 投影维度（pooling 前）
    r_global_dim: int = 256      # R pooling 后的全局表示维度（可 ≠ dim_model）

    # Pooling 方式： "mean" 或 "attention"
    pooling: str = "mean"

    # ==============================
    # FK (机械臂末端位姿) 编码配置
    # ==============================
    fk_input_dim: int = 12        # [x, y, z, 3x3旋转矩阵]
    fk_hidden_dim: int = 128     # FK encoder 中间层维度
    fk_embed_dim: int = 128      # FK 编码后维度（可 ≠ dim_model）

    # ==============================
    # Fusion & Output 配置
    # ==============================
    fusion_hidden_dim: int = 512  # 融合 MLP 的隐藏层维度（可 > dim_model）
    output_dim: int = 512  # 最终输出维度，应 = dim_model