import math
import numpy as np
import quaternion
from utils.math_util import get_random_rotation

class RandomHoriRotate:
    """
    Randomly rotate the horizontal plane.
    """
    def __init__(self, random_angle):
        self.random_angle = random_angle

    def __call__(self, feature, target):
        """
        Args:
            feature: (N, 6) [gyro, acce]
            target: (2,) or (3,) [vx, vy] or [vx, vy, vz]
        """
        # 生成随机旋转 (绕 Z 轴)
        angle = np.random.uniform(-self.random_angle, self.random_angle)
        q = quaternion.from_rotation_vector([0, 0, angle])
        
        # 旋转 Feature (IMU)
        # feature 是 (N, 6)，前3是gyro，后3是acce
        gyro = feature[:, :3]
        acce = feature[:, 3:]
        
        # 将 vector 转换为 quaternion 进行旋转: q * v * q_conj
        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros((gyro.shape[0], 1)), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros((acce.shape[0], 1)), acce], axis=1))
        
        gyro_rotated = quaternion.as_float_array(q * gyro_q * q.conj())[:, 1:]
        acce_rotated = quaternion.as_float_array(q * acce_q * q.conj())[:, 1:]
        
        feature_rotated = np.concatenate([gyro_rotated, acce_rotated], axis=1)
        
        # 旋转 Target (Velocity)
        # target 通常是 (2,) 或 (3,)
        targ_dim = target.shape[0]
        if targ_dim == 2:
            # 只有 vx, vy
            v = np.array([target[0], target[1], 0.0])
        else:
            v = target
            
        v_q = quaternion.from_float_array(np.concatenate([[0], v]))
        v_rotated = quaternion.as_float_array(q * v_q * q.conj())[1:]
        
        target_rotated = v_rotated[:targ_dim]
        
        return feature_rotated, target_rotated