# # Copyright 2025 DeepMind Technologies Limited
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# """VLA bridge for Go1 SAR low-level pilot (OpenVLA-7B placeholder).

# VRAM note: When loading OpenVLA-7B on L4 (24GB), consider device="cuda" for VLA
# while keeping VLM image processing on CPU if OOM occurs.
# """

# from typing import TYPE_CHECKING

# import jax.numpy as jp

# if TYPE_CHECKING:
#   import numpy as np


# import torch
# from transformers import AutoModelForVision2Seq, AutoProcessor

# class OpenVLABridge:
#     def __init__(self, device="cuda", dtype="bf16"):
#         self.device = device
#         # Load in 4-bit to save ~12GB VRAM
#         self.model = AutoModelForVision2Seq.from_pretrained(
#             "openvla/openvla-7b",
#             torch_dtype=torch.bfloat16,
#             low_cpu_mem_usage=True,
#             load_in_4bit=True,
#             trust_remote_code=True
#         )
#         self.processor = AutoProcessor.from_pretrained(
#             "openvla/openvla-7b", 
#             trust_remote_code=True
#         )

#     def get_vla_action(self, fpv_frame, strategic_instruction):
#         # Format the prompt exactly as OpenVLA was trained
#         prompt = f"Inhabitants: What action should the robot take to {strategic_instruction}?"
        
#         # Preprocess image and text
#         inputs = self.processor(prompt, fpv_frame, return_tensors="pt").to(self.device, torch.bfloat16)
        
#         # Inference
#         with torch.no_grad():
#             action = self.model.predict_action(**inputs, unnorm_key="bridge_orig")
            
#         # action is typically [7,] or [3,] depending on your task mapping
#         # Map OpenVLA outputs to your [vel_x, vel_y, yaw_rate]
#         return jp.array(action[:3], dtype=jp.float32)

# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""VLA bridge for Go1 SAR low-level pilot (OpenVLA-7B).

VRAM note: Loading OpenVLA-7B on L4 (24GB) in 4-bit consumes ~5-7GB VRAM,
leaving enough room for the MuJoCo simulation and JAX buffers.
"""

from typing import TYPE_CHECKING, Union

import numpy as np
import torch
import jax.numpy as jp
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

if TYPE_CHECKING:
    pass


def _frame_to_pil(frame: Union["np.ndarray", "Image.Image"]) -> "Image.Image":
    """Convert numpy array or PIL Image to PIL Image for OpenVLA processor."""
    if hasattr(frame, "convert"):
        return frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(
            f"Expected frame shape (H, W, 3) or (H, W, 4), got {getattr(arr, 'shape', 'unknown')}"
        )
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class OpenVLABridge:
    def __init__(self, device="cuda", dtype="bf16"):
        self.device = device

        # Use BitsAndBytesConfig when available (silences load_in_4bit deprecation)
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if BitsAndBytesConfig is not None:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["load_in_4bit"] = True

        # We use AutoModelForVision2Seq because OpenVLA's custom configuration
        # is compatible with this AutoClass as long as trust_remote_code=True.
        self.model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            **load_kwargs,
        )

        # Belt-and-suspenders: transformers 4.48+ may access _supports_sdpa
        if not hasattr(self.model, "_supports_sdpa"):
            self.model._supports_sdpa = False

        self.processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b",
            trust_remote_code=True,
        )

        print("🚀 OpenVLA-7B Pilot Initialized on L4 GPU.")

    def get_vla_action(self, fpv_frame, strategic_instruction: str):
        """
        Takes the FPV image and the high-level VLM strategy to output low-level velocities.

        Accepts fpv_frame as numpy array (H,W,3) or PIL Image; MuJoCo render returns numpy.
        """
        img = _frame_to_pil(fpv_frame)
        prompt = f"Inhabitants: What action should the robot take to {strategic_instruction}?"
        inputs = self.processor(prompt, img, return_tensors="pt").to(
            self.device, torch.bfloat16
        )

        with torch.inference_mode():
            action = self.model.predict_action(**inputs, unnorm_key="bridge_orig")

        # OpenVLA returns 7DoF [x, y, z, roll, pitch, yaw, gripper] -> map to [vel_x, vel_y, yaw]
        # Values from unnorm_key='bridge_orig' are in real-world units; apply velocity gain
        action = np.asarray(action, dtype=np.float32)
        n = len(action)
        vla_v_x = float(action[0]) if n > 0 else 0.0
        vla_v_y = float(action[1]) if n > 1 else 0.0
        vla_yaw = float(action[5]) if n > 5 else 0.0
        linear_gain, angular_gain = 5.0, 2.0
        vla_v_x *= linear_gain
        vla_v_y *= linear_gain
        vla_yaw *= angular_gain

        return jp.array([vla_v_x, vla_v_y, vla_yaw], dtype=jp.float32)