from dataclasses import dataclass, field


@dataclass
class SegmentUnderstandingConfig:
    # YOLO
    yolo_path: str = "runs/segment/grab_block/weights/best.pt"
    tracker_path: str = "custom/scripts/yolo/botsort.yaml"
    camera_name: str = "robot1"
    max_yolo_objects: int = 5
    num_classes: int = 4

    # Wrist-camera end-effector prior in normalized image coordinates.
    ee_anchor: list[float] = field(default_factory=lambda: [0.5, 1.0])
    anchor_distance_weight: float = 0.25
    anchor_area_weight: float = 0.05

    # FK
    urdf_path: str = "custom/config/SO101/so101_new_calib.urdf"
    ee_frame_name: str = "gripper_frame_link"

    # Object token encoder.
    object_numeric_dim: int = 12
    cls_embed_dim: int = 64
    r_hidden_dim: int = 128
    object_hidden_dim: int = 256
    object_refine_layers: int = 1

    # FK encoder. The raw FK input is [xyz, 3x3 rotation], internally reduced to xyz + 6D rotation.
    fk_input_dim: int = 12
    fk_hidden_dim: int = 128

    # Fusion.
    output_dim: int = 512
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.1
