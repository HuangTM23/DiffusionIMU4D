import numpy as np

def remove_gravity(accel, gravity=9.81):
    """
    从加速度数据中减去重力分量（假设已对齐到重力方向）。
    
    Args:
        accel (np.ndarray): 加速度序列 (N, 3)。
        gravity (float): 重力加速度常数。
        
    Returns:
        np.ndarray: 减去重力后的加速度序列。
    """
    # 假设重力始终在 Z 轴。在 RoNIN 数据集中，
    # 预处理后的数据通常已经将重力方向对齐到 Z 轴。
    accel_no_g = accel.copy()
    accel_no_g[:, 2] -= gravity
    return accel_no_g

def rotate_imu_data(data, rotation_matrices):
    """
    将 IMU 数据从坐标系 A 旋转到坐标系 B。
    
    Args:
        data (np.ndarray): 待旋转数据 (N, 3)。
        rotation_matrices (np.ndarray): 旋转矩阵。
            可以是一个矩阵 (3, 3) 应用于所有点，
            或者是矩阵序列 (N, 3, 3)。
            
    Returns:
        np.ndarray: 旋转后的数据。
    """
    if rotation_matrices.ndim == 2:
        # 单一旋转矩阵应用于所有数据点
        return np.matmul(data, rotation_matrices.T)
    elif rotation_matrices.ndim == 3:
        # 每个数据点对应一个旋转矩阵
        # result[i] = R[i] @ data[i]
        # 使用 einsum 进行批量矩阵向量乘法
        return np.einsum('nij,nj->ni', rotation_matrices, data)
    else:
        raise ValueError("rotation_matrices 维度必须为 2 (3,3) 或 3 (N,3,3)")

def integrate_velocity(accel, dt, initial_velocity=None):
    """
    对加速度进行积分以获取速度（基础物理实现）。
    
    Args:
        accel (np.ndarray): 减去重力后的世界系加速度 (N, 3)。
        dt (float): 时间间隔。
        initial_velocity (np.ndarray): 初始速度 (3,)。
        
    Returns:
        np.ndarray: 速度序列 (N, 3)。
    """
    if initial_velocity is None:
        initial_velocity = np.zeros(3)
    
    velocity = np.zeros_like(accel)
    current_v = initial_velocity
    
    for i in range(len(accel)):
        current_v = current_v + accel[i] * dt
        velocity[i] = current_v
        
    return velocity
