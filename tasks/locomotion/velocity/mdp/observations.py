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


# from collections import deque
# import numpy as np
# import cv2
# import os
# import torch

# # 64개 환경 각각 최대 28프레임을 저장
# frame_queues = [deque(maxlen=28) for _ in range(64)]

# def image_contour_debug(
#     env: ManagerBasedEnv,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
#     data_type: str = "rgb",
#     convert_perspective_to_orthogonal: bool = False,
#     normalize: bool = True
# ) -> torch.Tensor:
#     # 1) 센서 데이터 (640×320) 획득
#     sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
#     images = sensor.data.output[data_type]  

#     # 2) 원근 깊이 -> 직교화 (옵션)
#     if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
#         images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

#     # 3) 정규화 (RGB) or depth 후처리
#     if normalize and data_type == "rgb":
#         images = images.float() / 255.0
#         mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
#         images -= mean_tensor
#     elif "distance_to" in data_type or "depth" in data_type:
#         images[images == float("inf")] = 0

#     # 4) 만약 RGB 데이터라면 Canny 엣지 처리
#     if data_type == "rgb":
#         # (64, 320, 640, 3) in [-1,1] -> [0,255]
#         img_np = images.cpu().numpy()
#         img_np = ((img_np + 1) * 127.5).astype(np.uint8)  # shape = (64, 320, 640, 3)

#         # 그레이스케일 변환
#         img_gray = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[2]), dtype=np.uint8)
#         for i in range(img_np.shape[0]):
#             img_gray[i] = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2GRAY)

#         # CLAHE
#         enhanced_images = np.zeros_like(img_gray)
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#         for i in range(img_gray.shape[0]):
#             enhanced_images[i] = clahe.apply(img_gray[i])

#         # Canny (640×320 해상도)
#         edge_images = np.zeros_like(enhanced_images)
#         for i in range(enhanced_images.shape[0]):
#             edge_images[i] = cv2.Canny(enhanced_images[i], 25, 35)

#         # 5) 큐에 프레임 추가 (deque 최대 28)
#         for i in range(64):
#             frame_queues[i].append(edge_images[i])  # (320,640)

#         # 6) 5개 프레임 선택 ([0,7,14,21,27])
#         selected_indices = [0, 7, 14, 21, 27]
#         H, W = edge_images.shape[1], edge_images.shape[2]  # (320,640)
#         stacked_frames = np.zeros((64, 5, H, W), dtype=np.uint8)
#         for i in range(64):
#             frames_list = list(frame_queues[i])
#             for j, idx in enumerate(selected_indices):
#                 if len(frames_list) > idx:
#                     stacked_frames[i, j] = frames_list[idx]
#                 else:
#                     stacked_frames[i, j] = frames_list[-1]

#         # 7) 디버그(환경0 최신 프레임)를 ASCII로 표시 (80×40 크기로 시각화)
#         os.system('clear' if os.name == 'posix' else 'cls') 
#         print("\n🖥️ Contour Detection (ASCII View)\n" + "="*40)

#         ascii_chars = ['.', '#']  
#         # 환경0의 5개 프레임 중 마지막(27번째), shape=(320,640)
#         debug_frame = stacked_frames[0, -1]
#         # 디버그 용도로 (80,40)만 씀
#         debug_resized = cv2.resize(debug_frame, (80, 40))
#         ascii_img = '\n'.join(
#             ''.join(ascii_chars[1] if pixel > 0 else ascii_chars[0] for pixel in row)
#             for row in debug_resized
#         )
#         print(ascii_img)

#         # 8) 최종 출력: (80×80) 크기로 리사이즈 -> (64, 5, 80,80)
#         #    이어서 Flatten -> (64, 5*80*80)
#         TARGET_SIZE = (80, 80)
#         resized_frames = np.zeros((64, 5, TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)
#         for i in range(64):
#             for j in range(5):
#                 resized_frames[i, j] = cv2.resize(stacked_frames[i, j], TARGET_SIZE)

#         contour_tensor = torch.from_numpy(resized_frames).float() / 255.0
#         contour_tensor = contour_tensor.to(images.device)
#         reshaped_tensor = contour_tensor.view(64, -1)

#         return reshaped_tensor

#     # 그 외 타입은 그냥 반환
#     return images

import numpy as np
import cv2
import os
import torch
from collections import deque

# 64개 환경마다 최근 28프레임을 저장
frame_queues = [deque(maxlen=28) for _ in range(64)]

def image_line_debug_latest(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    normalize: bool = True,
    max_edges: int = 30,
    print_debug: bool = True  # 🔹 추가: 디버그 출력 제어
) -> torch.Tensor:
    """
    1) 64개 환경에서 최근 28개 프레임을 저장
    2) [0, 7, 14, 21, 27] 번째 5개 프레임을 사용해 선분 검출
    3) 환경0의 첫 번째 프레임(0번째)에 대한 ASCII 디버깅 수행 (조건: print_debug=True)
    4) 각 환경별 (5프레임 x max_edges x 4) Flatten하여 (64, 5 * max_edges * 4) 형태로 반환
    """

    # 1) 센서로부터 64프레임(각 env) 획득 (320×320)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output[data_type]  # (64, 320, 320, 3)

    # RGB 정규화
    if normalize and data_type == "rgb":
        images = images.float() / 255.0
        mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
        images -= mean_tensor

    # [-1,1] 범위를 [0,255]로 변환
    img_np = images.cpu().numpy()
    img_np = ((img_np + 1) * 127.5).astype(np.uint8)  # (64, 320, 320, 3)

    H, W = img_np.shape[1], img_np.shape[2]
    gray_imgs = np.zeros((64, H, W), dtype=np.uint8)
    for i in range(64):
        gray_imgs[i] = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2GRAY)

    edge_imgs = np.zeros_like(gray_imgs)
    for i in range(64):
        # 예: (5, 30)로 설정
        edge_imgs[i] = cv2.Canny(gray_imgs[i], 5, 30)

    # 🔹 큐에 프레임 저장
    for i in range(64):
        frame_queues[i].append(edge_imgs[i])

    # 🔹 5개 프레임 선택 ([0,7,14,21,27])
    selected_indices = [0, 7, 14, 21, 27]
    stacked_frames = np.zeros((64, 5, H, W), dtype=np.uint8)
    for i in range(64):
        frames_list = list(frame_queues[i])
        for j, idx in enumerate(selected_indices):
            if len(frames_list) > idx:
                stacked_frames[i, j] = frames_list[idx]
            else:
                stacked_frames[i, j] = frames_list[-1]

    # 🔹 HoughLinesP
    rho = 1
    theta = np.pi / 30
    threshold = 10
    min_line_length = 30
    max_line_gap = 10

    line_features_all = np.zeros((64, 5, max_edges * 4), dtype=np.float32)
    detected_counts = np.zeros((64, 5), dtype=int)

    for env_idx in range(64):
        for frame_idx in range(5):
            frame = stacked_frames[env_idx, frame_idx]
            lines = cv2.HoughLinesP(
                frame, rho, theta, threshold,
                minLineLength=min_line_length, maxLineGap=max_line_gap
            )

            frame_features = []
            if lines is not None:
                lines = lines.reshape(-1, 4)
                detected_counts[env_idx, frame_idx] = lines.shape[0]

                for (x1, y1, x2, y2) in lines:
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    angle = np.arctan2((y2 - y1), (x2 - x1))
                    frame_features.append([cx, cy, angle, length])

                frame_features = sorted(frame_features, key=lambda f: f[3], reverse=True)
                frame_features = frame_features[:max_edges]

            if len(frame_features) < max_edges:
                frame_features += [[0.0, 0.0, 0.0, 0.0]] * (max_edges - len(frame_features))

            line_features_all[env_idx, frame_idx] = np.array(frame_features, dtype=np.float32).flatten()

    # 🔹 터미널 출력 (조건: print_debug=True)
    if print_debug:
        os.system('clear' if os.name == 'posix' else 'cls')
        print("\n🖥️ RealTime Edge Debug (ASCII) - First Frame\n" + "="*40)
        print(f"[Env=0] Detected lines = {detected_counts[0, 0]}")

    # 디버그용 이미지 (환경0, 첫 번째 프레임)
    debug_img = stacked_frames[0, 0].copy()  # (320,320)
    color_debug = np.stack([debug_img]*3, axis=-1)  # (320,320,3)

    # 라인 정보
    lines_0 = line_features_all[0, 0].reshape(max_edges, 4)
    for (cx, cy, angle, length) in lines_0:
        if length > 0:
            half = length / 2
            dx = np.cos(angle) * half
            dy = np.sin(angle) * half
            x1, y1 = int(cx - dx), int(cy - dy)
            x2, y2 = int(cx + dx), int(cy + dy)
            cv2.line(color_debug, (x1, y1), (x2, y2), (0,255,0), 1)

    # print_debug=True 일 때만 ASCII 출력
    if print_debug:
        debug_resized = cv2.resize(color_debug, (80, 40))
        debug_gray = cv2.cvtColor(debug_resized, cv2.COLOR_BGR2GRAY)
        ascii_chars = ['.', '#']
        ascii_img = '\n'.join(
            ''.join(ascii_chars[1] if px>0 else ascii_chars[0] for px in row)
            for row in debug_gray
        )
        print(ascii_img)

    # 🔹 최종 (64, 5 * max_edges * 4) Flatten 후 반환 (0~1 정규화)
    final_tensor = torch.from_numpy(line_features_all).float().view(64, -1)
    max_val = torch.max(final_tensor)
    if max_val > 0:
        final_tensor /= max_val

    final_tensor = final_tensor.to(images.device)



    return final_tensor


def joint_vel_debug(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), print_debug: bool = True):
    """The joint velocities of the asset. If print_debug=True, prints them in terminal."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    left_vel = asset.data.joint_vel[0, 0]
    right_vel = asset.data.joint_vel[0, 4]
    left_err = abs(env.action_manager.action[0,0] - left_vel)
    right_err = abs(env.action_manager.action[0,1] - right_vel)

    if print_debug:
        print(f"left_joint_vel = {left_vel:.1f} , right_joint_vel = {right_vel:.1f} , "
              f"{'왼쪽' if left_vel < right_vel else '오른쪽'} "
              f"{abs(left_vel - right_vel):.1f}  (Velocity Difference), "
              f"Minus is go ahead, vel unit: rad/s, torque unit: Nm")
        print(f"left_error = {left_err:.1f} , right_error = {right_err:.1f}")
        print(f"torque: {asset.data.applied_torque[0, [0,1,4,5]]}")
        print(f"acc= {asset.data.joint_acc[0, [0,1,4,5]]}")
        print(f"root_vel= {asset.data.root_lin_vel_w[0, 0]}")

    return asset.data.joint_vel[:, [0,1,4,5]]


def last_action_debug(env: ManagerBasedEnv, action_name: str | None = None, print_debug: bool = True) -> torch.Tensor:
    """
    The last input action to the environment.
    If print_debug=True, prints them in terminal.
    """
    if action_name is None:
        if print_debug:
            os.system('clear' if os.name == 'posix' else 'cls')
            left_input = env.action_manager.action[0,0]
            right_input = env.action_manager.action[0,1]
            diff = abs(left_input - right_input)
            side = '왼쪽' if left_input < right_input else '오른쪽'
            print(f"left_vel_input = {left_input:.1f} , right_vel_input = {right_input:.1f} rad/s, "
                  f"{side} = {diff:.1f} rad/s (Velocity Difference), "
                  f"Minus is go ahead, vel unit: rad/s, torque unit: Nm")

        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions


import torch
import cv2
import numpy as np

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

        # 🔹 RGB 데이터 범위 조정 (0~255로 변환)
        img_np = ((img_np + 1) * 127.5).astype(np.uint8)

        # 🔹 RGB → Grayscale 변환
        img_gray = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[2]), dtype=np.uint8)
        for i in range(img_np.shape[0]):
            img_gray[i] = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2GRAY)

        # 🔹 CLAHE 적용
        enhanced_images = np.zeros_like(img_gray)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        for i in range(img_gray.shape[0]):
            enhanced_images[i] = clahe.apply(img_gray[i])

        # 🔹 Canny Edge Detection
        edge_images = np.zeros_like(enhanced_images)
        for i in range(enhanced_images.shape[0]):
            edge_images[i] = cv2.Canny(enhanced_images[i], 15, 35)

        # 🔹 **채널을 3개로 확장 (기존 형식 유지)**
        edge_images = np.repeat(edge_images[:, :, :, np.newaxis], 3, axis=-1)  # (batch, height, width, 3)

        # 🔹 Tensor 변환 (기존 코드 유지)
        contour_tensor = torch.from_numpy(edge_images).float() / 255.0
        contour_tensor = contour_tensor.to(images.device)
        
        return contour_tensor


class image_features_gray(ManagerTermBase):
    """Extracted image features from a pre-trained frozen encoder.

    This method calls the :meth:`image` function to retrieve images, and then performs
    inference on those images.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from torchvision import models
        from transformers import AutoModel

        def create_theia_model(model_name):
            return {
                "model": (
                    lambda: AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True)
                    .eval()
                    .to("cuda:0")
                ),
                "preprocess": lambda img: (img - torch.amin(img, dim=(1, 2), keepdim=True)) / (
                    torch.amax(img, dim=(1, 2), keepdim=True) - torch.amin(img, dim=(1, 2), keepdim=True)
                ),
                "inference": lambda model, images: model.forward_feature(
                    images, do_rescale=False, interpolate_pos_encoding=True
                ),
            }

        def create_resnet_model(resnet_name):
            return {
                "model": lambda: getattr(models, resnet_name)(pretrained=True).eval().to("cuda:0"),
                "preprocess": lambda img: (
                    img.permute(0, 3, 1, 2)  # Convert [batch, height, width, 3] -> [batch, 3, height, width]
                    - torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
                ) / torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1),
                "inference": lambda model, images: model(images),
            }

        # List of Theia models
        theia_models = [
            "theia-tiny-patch16-224-cddsv",
            "theia-tiny-patch16-224-cdiv",
            "theia-small-patch16-224-cdiv",
            "theia-base-patch16-224-cdiv",
            "theia-small-patch16-224-cddsv",
            "theia-base-patch16-224-cddsv",
        ]

        # List of ResNet models
        resnet_models = ["resnet18", "resnet34", "resnet50", "resnet101"]

        self.default_model_zoo_cfg = {}

        # Add Theia models to the zoo
        for model_name in theia_models:
            self.default_model_zoo_cfg[model_name] = create_theia_model(model_name)

        # Add ResNet models to the zoo
        for resnet_name in resnet_models:
            self.default_model_zoo_cfg[resnet_name] = create_resnet_model(resnet_name)

        self.model_zoo_cfg = self.default_model_zoo_cfg
        self.model_zoo = {}

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "ResNet18",
        model_device: str | None = "cuda:0",
        reset_model: bool = False,
    ) -> torch.Tensor:
        """Extracted image features from a pre-trained frozen encoder.

        Args:
            env: The environment.
            sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").
            data_type: THe sensor configuration datatype. Defaults to "rgb".
            convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
                This is used only when the data type is "distance_to_camera". Defaults to False.
            model_zoo_cfg: Map from model name to model configuration dictionary. Each model
                configuration dictionary should include the following entries:
                - "model": A callable that returns the model when invoked without arguments.
                - "preprocess": A callable that processes the images and returns the preprocessed results.
                - "inference": A callable that, when given the model and preprocessed images,
                    returns the extracted features.
            model_name: The name of the model to use for inference. Defaults to "ResNet18".
            model_device: The device to store and infer models on. This can be used help offload
                computation from the main environment GPU. Defaults to "cuda:0".
            reset_model: Initialize the model even if it already exists. Defaults to False.

        Returns:
            torch.Tensor: the image features, on the same device as the image
        """
        if model_zoo_cfg is not None:  # use other than default
            self.model_zoo_cfg.update(model_zoo_cfg)

        if model_name not in self.model_zoo or reset_model:
            # The following allows to only load a desired subset of a model zoo into GPU memory
            # as it becomes needed, in a "lazy" evaluation.
            print(f"[INFO]: Adding {model_name} to the model zoo")
            self.model_zoo[model_name] = self.model_zoo_cfg[model_name]["model"]()
        if model_device is not None:
            first_param = next(self.model_zoo[model_name].parameters(), None)
            if first_param is not None and first_param.device != torch.device(model_device):
                self.model_zoo[model_name] = self.model_zoo[model_name].to(model_device)


        images = image_gray(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=True,  # want this for training stability
        )

        image_device = images.device

        if model_device is not None:
            images = images.to(model_device)

        proc_images = self.model_zoo_cfg[model_name]["preprocess"](images)
        features = self.model_zoo_cfg[model_name]["inference"](self.model_zoo[model_name], proc_images)

        return features.to(image_device).clone()
