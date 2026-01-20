import math
import numpy as np
import quaternion
from utils.math_util import get_random_rotation

class RandomHoriRotate:
    """
    Randomly rotate the horizontal plane.
    Supports both single vector and sequence targets.
    """
    def __init__(self, random_angle):
        self.random_angle = random_angle

    def __call__(self, feature, target):
        """
        Args:
            feature: (N, 6) [gyro, acce]
            target: (2,) or (3,) [vx, vy] or [vx, vy, vz]
                    OR (N, 2) or (N, 3) for sequences
        """
        # Generate random rotation (Z-axis)
        angle = np.random.uniform(-self.random_angle, self.random_angle)
        q = quaternion.from_rotation_vector([0, 0, angle])
        
        # Rotate Feature (IMU) (N, 6)
        gyro = feature[:, :3]
        acce = feature[:, 3:]
        
        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros((gyro.shape[0], 1)), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros((acce.shape[0], 1)), acce], axis=1))
        
        gyro_rotated = quaternion.as_float_array(q * gyro_q * q.conj())[:, 1:]
        acce_rotated = quaternion.as_float_array(q * acce_q * q.conj())[:, 1:]
        
        feature_rotated = np.concatenate([gyro_rotated, acce_rotated], axis=1)
        
        # Rotate Target (Velocity)
        if target.ndim == 1:
            # Single vector case (2,) or (3,)
            v = target
            if v.shape[0] == 2:
                v = np.array([v[0], v[1], 0.0])
            
            v_q = quaternion.from_float_array(np.concatenate([[0], v]))
            v_rotated = quaternion.as_float_array(q * v_q * q.conj())[1:]
            target_rotated = v_rotated[:target.shape[0]]
            
        else:
            # Sequence case (N, 2) or (N, 3)
            v = target
            is_2d = (v.shape[1] == 2)
            if is_2d:
                # Append z=0 column
                v = np.concatenate([v, np.zeros((v.shape[0], 1))], axis=1)
            
            # (N, 3) -> (N, 4) quaternions
            v_q = quaternion.from_float_array(np.concatenate([np.zeros((v.shape[0], 1)), v], axis=1))
            v_rotated = quaternion.as_float_array(q * v_q * q.conj())[:, 1:]
            
            if is_2d:
                target_rotated = v_rotated[:, :2]
            else:
                target_rotated = v_rotated
        
        return feature_rotated, target_rotated
