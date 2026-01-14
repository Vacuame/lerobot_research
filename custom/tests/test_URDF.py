
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# 加载 URDF
robot = RobotWrapper.BuildFromURDF("custom/config/SO101/so101_new_calib.urdf", ["custom/config/SO101/"], pin.JointModelFreeFlyer())

# 构造关节向量（必须是 np.ndarray，dtype=float64）
q = np.zeros(robot.nq, dtype=np.float64)  # 或 np.array([0.0]*robot.nq, dtype=np.float64)

# 获取末端帧 ID
ee_frame_id = robot.model.getFrameId("gripper_frame_link")

# 前向运动学
ee_pose = robot.framePlacement(q, ee_frame_id)
print("末端位姿:\n", ee_pose)