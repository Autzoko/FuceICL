"""Retriever 共用的可解释几何相似度函数。"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as functional


def rbf_score(distance: float, sigma: float) -> float:
    """将非负距离映射为 [0, 1] 的 RBF 相似度。"""
    return math.exp(-0.5 * (distance / sigma) ** 2)


def tensor_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    """计算两个向量的欧氏距离。"""
    return float(
        torch.linalg.vector_norm(first.float().cpu() - second.float().cpu())
    )


def extent_log_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    """计算物体各轴尺度对数比的均方根。"""
    difference = torch.log(first.float().cpu()) - torch.log(second.float().cpu())
    return float(torch.sqrt(torch.mean(difference.square())))


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """将连续 rotation-6D 表示转换为旋转矩阵。"""
    first = functional.normalize(rotation_6d[:3].float().cpu(), dim=0)
    second = rotation_6d[3:6].float().cpu()
    second = functional.normalize(second - torch.dot(first, second) * first, dim=0)
    third = torch.linalg.cross(first, second, dim=0)
    return torch.stack((first, second, third), dim=1)


def rotation_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    """计算两个 rotation-6D 姿态的 SO(3) 测地线距离。"""
    return rotation_matrix_distance(
        rotation_6d_to_matrix(first),
        rotation_6d_to_matrix(second),
    )


def rotation_matrix_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    """计算两个 3×3 旋转矩阵的 SO(3) 测地线距离。"""
    relative = first.transpose(0, 1) @ second
    cosine = torch.clamp((torch.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(torch.acos(cosine))


def mean_score(values: Sequence[float]) -> float:
    """计算非空分数序列的算术平均。"""
    if not values:
        raise ValueError("values 不能为空")
    return sum(values) / len(values)
