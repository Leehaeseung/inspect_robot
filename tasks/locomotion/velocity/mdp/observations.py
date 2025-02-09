from __future__ import annotations
import cv2
import torch
import numpy as np
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg
from omni.isaac.lab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

import torch
import cv2
import numpy as np
import os
from collections import deque

# 🔥 64개 배치 각각의 프레임을 저장하는 큐 (각 배치에 대해 5개 저장)
frame_queues = [deque(maxlen=5) for _ in range(64)]

def image_gray(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True
) -> torch.Tensor:
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output[data_type]  

    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    if normalize and data_type == "rgb":
        images = images.float() / 255.0
        mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
        images -= mean_tensor
    elif "distance_to" in data_type or "depth" in data_type:
        images[images == float("inf")] = 0

    if data_type == "rgb":
        img_np = images.cpu().numpy()

        img_np = ((img_np + 1) * 127.5).astype(np.uint8)

        img_gray = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[2]), dtype=np.uint8)
        for i in range(img_np.shape[0]):
            img_gray[i] = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2GRAY)

        enhanced_images = np.zeros_like(img_gray)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        for i in range(img_gray.shape[0]):
            enhanced_images[i] = clahe.apply(img_gray[i])

        edge_images = np.zeros_like(enhanced_images)
        for i in range(enhanced_images.shape[0]):
            edge_images[i] = cv2.Canny(enhanced_images[i], 20, 40)

        # 🔹 큐에 프레임 추가
        for i in range(64):
            frame_queues[i].append(edge_images[i])

        # 🔹 64개의 배치에 대해 5개 프레임을 스택 형태로 변환 (64, 5, height, width)
        stacked_frames = np.zeros((64, 5, edge_images.shape[1], edge_images.shape[2]), dtype=np.uint8)
        for i in range(64):
            frames_list = list(frame_queues[i])
            for j, frame in enumerate(frames_list):
                stacked_frames[i, j] = frame

        # 🔹 터미널에 ASCII 출력 (최신 프레임의 Batch 0만 표시)
        os.system('clear' if os.name == 'posix' else 'cls') 
        print("\n🖥️ Contour Detection (ASCII View)\n" + "="*40)

        ascii_chars = ['.', '#']  
        resized = cv2.resize(stacked_frames[0, -1], (80, 80))  
        ascii_img = '\n'.join(
            ''.join(ascii_chars[1] if pixel > 0 else ascii_chars[0] for pixel in row)
            for row in resized
        )
        print(ascii_img)  

        # 🔹 PyTorch 텐서로 변환 (64, 5, height * width)
        contour_tensor = torch.from_numpy(stacked_frames).float() / 255.0
        contour_tensor = contour_tensor.to(images.device)
        reshaped_tensor = contour_tensor.view(64,-1)  

        return reshaped_tensor

    return images