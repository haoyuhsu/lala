"""Camera configs."""

import numpy as np
import pybullet as p
from pyquaternion import Quaternion

def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)

def lookat(center, position):
    lookdir = normalize(center - position)
    tmp = np.float32([0, 0, -1])
    right = np.cross(tmp, lookdir)
    right = normalize(right)
    up = np.cross(lookdir, right)
    up = normalize(up)
    rotm = np.stack((right, up, lookdir), axis=-1)
    q = Quaternion(matrix=rotm)
    q_pybullet = np.zeros(4)
    q_pybullet[0] = q[1]
    q_pybullet[1] = q[2]
    q_pybullet[2] = q[3]
    q_pybullet[3] = q[0]
    return q_pybullet


class RealSenseD415():
    """Default configuration with 3 RealSense RGB-D cameras."""

    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (480, 640)
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

    # Set default camera poses.
    # front_position = (1., 0, 0.75)
    # front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
    # front_rotation = p.getQuaternionFromEuler(front_rotation)
    # left_position = (0, 0.5, 0.75)
    # left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
    # left_rotation = p.getQuaternionFromEuler(left_rotation)
    # right_position = (0, -0.5, 0.75)
    # right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
    # right_rotation = p.getQuaternionFromEuler(right_rotation)

    # Default camera configs.
    # CONFIG = [{
    #     'image_size': image_size,
    #     'intrinsics': intrinsics,
    #     'position': front_position,
    #     'rotation': front_rotation,
    #     'zrange': (0.01, 10.),
    #     'noise': False
    # }, {
    #     'image_size': image_size,
    #     'intrinsics': intrinsics,
    #     'position': left_position,
    #     'rotation': left_rotation,
    #     'zrange': (0.01, 10.),
    #     'noise': False
    # }, {
    #     'image_size': image_size,
    #     'intrinsics': intrinsics,
    #     'position': right_position,
    #     'rotation': right_rotation,
    #     'zrange': (0.01, 10.),
    #     'noise': False
    # }]

    def get_nerf_camera_config(image_size, intrinsics, n_input_views):
        NERF_CONFIG = []
        center = np.array([0.25, 0, 0])
        height_low = 0.75
        height_high = 1.15
        r = 0.75

        # heights = np.linspace(height_low, height_high, n_input_views)
        heights = height_low * np.ones(n_input_views)  # Fixed height.
        for i in range(n_input_views):
            theta = (2 * np.pi / n_input_views) * i
            x = r * np.cos(theta) + center[0]
            y = r * np.sin(theta) + center[1]
            z = heights[i]
            position = np.array([x, y, z])
            rotation = lookat(center, position)
            
            NERF_CONFIG.append({
                'image_size': image_size,
                'intrinsics': intrinsics,
                'position': position,
                'rotation': rotation,
                'zrange': (0.01, 10.),
                'noise': False
            })
        return NERF_CONFIG
    
    CONFIG = get_nerf_camera_config(image_size, intrinsics, 4)


class Oracle():
    """Top-down noiseless image used only by the oracle demonstrator."""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (63e4, 0, 320., 0, 63e4, 240., 0, 0, 1)
    position = (0.5, 0, 1000.)
    rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))

    # Camera config.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': position,
        'rotation': rotation,
        'zrange': (999.7, 1001.),
        'noise': False
    }]


class RS200Gazebo():
    """Gazebo Camera"""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (554.3826904296875, 0.0, 320.0, 0.0, 554.3826904296875, 240.0, 0.0, 0.0, 1.0)
    position = (0.5, 0, 1.0)
    rotation = p.getQuaternionFromEuler((0, np.pi, np.pi / 2))

    # Camera config.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': position,
        'rotation': rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }]


class KinectFranka():
    """Kinect Franka Camera"""

    # Near-orthographic projection.
    image_size = (424,512)
    intrinsics = (365.57489013671875, 0.0, 257.5205078125, 0.0, 365.57489013671875, 205.26710510253906, 0.0, 0.0, 1.0)
    position = (1.082, -0.041, 1.027)
    rotation = p.getQuaternionFromEuler((-2.611, 0.010, 1.553))

    # Camera config.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': position,
        'rotation': rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }]