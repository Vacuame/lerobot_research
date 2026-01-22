import numpy as np
import pinocchio as pin
import torch

class SimpleKinematics:
    def __init__(self, urdf_path: str, ee_frame: str):
        self.robot_model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.robot_model.createData()
        self.ee_frame = self.robot_model.getFrameId(ee_frame)

    def forward_kinematics_telop(self, joint_dict: dict[str, float], joint_names: list[str]) -> np.ndarray:
        q = np.array([joint_dict[f"{name}.pos"] for name in joint_names], dtype=np.float64)
        q_rad = np.deg2rad(q)  # 转换为弧度，这一步是关键，之前乱数据就是因为这个
        pin.forwardKinematics(self.robot_model, self.data, q_rad)
        ee_pose = pin.updateFramePlacement(self.robot_model, self.data, self.ee_frame)
        #DEBUG 打印所有body的位姿
        # print("\n=== All Body Poses (w.r.t. base) ===")
        # for i in range(1, self.robot_model.nbodies):  # 0 is universe
        #     name = self.robot_model.names[i]
        #     pose = self.data.oMi[i]
        #     p = pose.translation
        #     print(f"{name:20s} | p = [{p[0]:7.3f}, {p[1]:7.3f}, {p[2]:7.3f}]")
        return ee_pose  # 使用之前需要展平
    
    
    def forward_kinematics_batch(
        self,
        q_batch: torch.Tensor,   # (B, N), rad
    ) -> torch.Tensor:
        """
        Returns:
            ee_pose: (B, 12)
              [x, y, z,
               R00, R01, R02,
               R10, R11, R12,
               R20, R21, R22]
        """
        assert q_batch.ndim == 2
        B, N = q_batch.shape

        # Pinocchio: numpy + CPU
        q_np = q_batch.detach().cpu().numpy()

        ee_out = np.zeros((B, 12), dtype=np.float64)

        for i in range(B):
            q = q_np[i].astype(np.float64)

            # 1. FK（joint）
            pin.forwardKinematics(self.robot_model, self.data, q)

            # 2. frame placement
            pin.updateFramePlacements(self.robot_model, self.data)

            # 3. 取 EE
            oMf = self.data.oMf[self.ee_frame]

            p = oMf.translation          # (3,)
            R = oMf.rotation.reshape(-1) # (9,)

            ee_out[i, :3] = p
            ee_out[i, 3:] = R

        return torch.from_numpy(ee_out).to(
            device=q_batch.device,
            dtype=q_batch.dtype
        )




# 使用例，暂时放在这里
# from lerobot.model.custom_kinematics import RobotKinematics, compute_forward_kinematics_joints_to_ee,SimpleKinematics
#         urdf_path = "custom/config/SO101/so101_new_calib.urdf"
#         ee_frame_name = "gripper_frame_link"
#         joint_names = list(robot.bus.motors.keys())
#         kin = SimpleKinematics(urdf_path, ee_frame_name)
#         ee_pose = kin.forward_kinematics(robot_action_to_send, joint_names)
#         result = ee_pose
#         print(result)