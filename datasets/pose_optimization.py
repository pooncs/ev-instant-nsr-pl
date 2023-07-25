"""Lie group and pose methods for pose optimization taken from Nerfstudio"""

import torch
import functools
from typing import Tuple
from typing_extensions import assert_never
import numpy as np
_EPS = np.finfo(float).eps * 4.0

"""
Lie group methods
"""
# We make an exception on snake case conventions because SO3 != so3.
def exp_map_SO3xR3(tangent_vector: torch.Tensor) -> torch.Tensor:
    """Compute the exponential map of the direct product group `SO(3) x R^3`.

    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.

    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector. [b 6]
    Returns:
        [R|t] transformation matrices. [b 3 4]
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    if tangent_vector.ndim == 1: tangent_vector = tangent_vector.unsqueeze(0)
    log_rot = tangent_vector[:, 3:]
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.zeros(tangent_vector.shape[0], 3, 4, dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    # Compute the translation
    ret[:, :3, 3] = tangent_vector[:, :3]
    return ret


def exp_map_SE3(tangent_vector: torch.Tensor) -> torch.Tensor:
    """Compute the exponential map `se(3) -> SE(3)`.

    This can be used for learning pose deltas on `SE(3)`.

    Args:
        tangent_vector: A tangent vector from `se(3)`. [b 6]

    Returns:
        [R|t] transformation matrices. [b 3 4]
    """

    tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
    tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

    theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
    theta2 = theta**2
    theta3 = theta**3

    near_zero = theta < 1e-2
    non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
    theta_nz = torch.where(near_zero, non_zero, theta)
    theta2_nz = torch.where(near_zero, non_zero, theta2)
    theta3_nz = torch.where(near_zero, non_zero, theta3)

    # Compute the rotation
    sine = theta.sin()
    cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
    sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz)
    ret = torch.zeros(tangent_vector.shape[0], 3, 4).to(dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = one_minus_cosine_by_theta2 * tangent_vector_ang @ tangent_vector_ang.transpose(1, 2)

    ret[:, 0, 0] += cosine.view(-1)
    ret[:, 1, 1] += cosine.view(-1)
    ret[:, 2, 2] += cosine.view(-1)
    temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
    ret[:, 0, 1] -= temp[:, 2]
    ret[:, 1, 0] += temp[:, 2]
    ret[:, 0, 2] += temp[:, 1]
    ret[:, 2, 0] -= temp[:, 1]
    ret[:, 1, 2] -= temp[:, 0]
    ret[:, 2, 1] += temp[:, 0]

    # Compute the translation
    sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 - theta2 / 24, one_minus_cosine_by_theta2)
    theta_minus_sine_by_theta3_t = torch.where(near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz)

    ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
    ret[:, :, 3:] += one_minus_cosine_by_theta2 * torch.cross(tangent_vector_ang, tangent_vector_lin, dim=1)
    ret[:, :, 3:] += theta_minus_sine_by_theta3_t * (
        tangent_vector_ang @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
    )
    return ret


"""
Common 3D pose methods
"""

def to4x4(pose: torch.Tensor) -> torch.Tensor:
    """Convert 3x4 pose matrices to a 4x4 with the addition of a homogeneous coordinate.

    Args:
        pose: Camera pose without homogenous coordinate. [b 3 4]

    Returns:
        Camera poses with additional homogenous coordinate added. [b 4 4]
    """
    constants = torch.zeros_like(pose[..., :1, :], device=pose.device)
    constants[..., :, 3] = 1
    return torch.cat([pose, constants], dim=-2)


def inverse(pose: torch.Tensor) -> torch.Tensor:
    """Invert provided pose matrix.

    Args:
        pose: Camera pose without homogenous coordinate. [b 3 4]

    Returns:
        Inverse of pose. [b 3 4]
    """
    R = pose[..., :3, :3]
    t = pose[..., :3, 3:]
    R_inverse = R.transpose(-2, -1)
    t_inverse = -R_inverse.matmul(t)
    return torch.cat([R_inverse, t_inverse], dim=-1)


def multiply(pose_a: torch.Tensor, pose_b: torch.Tensor) -> torch.Tensor:
    """Multiply two pose matrices, A @ B.

    Args:
        pose_a: Left pose matrix, usually a transformation applied to the right. [b 3 4]
        pose_b: Right pose matrix, usually a camera pose that will be transformed by pose_a. [b 3 4]

    Returns:
        Camera pose matrix where pose_a was applied to pose_b. [b 3 4]
    """
    R1, t1 = pose_a[..., :3, :3], pose_a[..., :3, 3:]
    R2, t2 = pose_b[..., :3, :3], pose_b[..., :3, 3:]
    R = R1.matmul(R2)
    t = t1 + R1.matmul(t2)
    return torch.cat([R, t], dim=-1)


def normalize(poses: torch.Tensor) -> torch.Tensor:
    """Normalize the XYZs of poses to fit within a unit cube ([-1, 1]). Note: This operation is not in-place.

    Args:
        poses: A collection of poses to be normalized. [b 3 4]

    Returns;
        Normalized collection of poses. [b 3 4]
    """
    pose_copy = torch.clone(poses)
    pose_copy[..., :3, 3] /= torch.max(torch.abs(poses[..., :3, 3]))

    return pose_copy


def normalize_with_norm(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize tensor along axis and return normalized value with norms.

    Args:
        x: tensor to normalize.
        dim: axis along which to normalize.

    Returns:
        Tuple of normalized tensor and corresponding norm.
    """

    norm = torch.maximum(torch.linalg.vector_norm(x, dim=dim, keepdims=True), torch.tensor([_EPS]).to(x))
    return x / norm, norm

"""
Pose and Intrinsics Optimizers
"""
class CameraOptimizer(torch.nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    def __init__(self, mode, num_cameras, device):
        super().__init__()
        self.num_cameras = num_cameras
        self.mode = mode['mode']
        self.position_noise_std = 0.0
        self.orientation_noise_std = 0.0
        self.rank = device

        # Initialize learnable parameters.
        if self.mode == "none":
            pass
        elif self.mode in ("SO3xR3", "SE3"):
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((self.num_cameras, 6), device=self.rank))
        else:
            assert_never(self.mode)

        # Initialize pose noise; useful for debugging.
        if self.position_noise_std != 0.0 or self.orientation_noise_std != 0.0:
            assert self.position_noise_std >= 0.0 and self.orientation_noise_std >= 0.0
            std_vector = torch.tensor(
                [self.position_noise_std] * 3 + [self.orientation_noise_std] * 3, device=self.rank)
            self.pose_noise = exp_map_SE3(torch.normal(torch.zeros((self.num_cameras, 6), device=self.rank), std_vector))
        else:
            self.pose_noise = None

    def forward(
        self,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates. [num_cam, 3 4]
        """
        outputs = []

        # Apply learned transformation delta.
        if self.mode == "none":
            outputs.append(torch.eye(4, device=self.rank)[None, :3, :4].tile(indices.shape[0], 1, 1))
        elif self.mode == "SO3xR3":
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
        elif self.mode == "SE3":
            outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
        else:
            assert_never(self.mode)

        # Apply initial pose noise.
        if self.pose_noise is not None:
            outputs.append(self.pose_noise[indices, :, :])

        # Return: identity if no transforms are needed, otherwise multiply transforms together.
        if len(outputs) == 0:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            return torch.eye(4, device=self.rank)[None, :3, :4].tile(indices.shape[0], 1, 1)
        return functools.reduce(multiply, outputs)