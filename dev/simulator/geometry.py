"""不依赖仿真或数据格式库的通用几何函数。"""

from __future__ import annotations

import numpy as np


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """将 ManiSkill/SAPIEN ``wxyz`` quaternion 转为旋转矩阵。"""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm((w, x, y, z)))
    if norm <= 1e-12:
        raise ValueError("TCP quaternion 退化")
    w, x, y, z = np.asarray((w, x, y, z), dtype=np.float64) / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
