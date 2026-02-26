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

from typing import TYPE_CHECKING
import torch
import jax.numpy as jp
from transformers import AutoModelForVision2Seq, AutoProcessor

if TYPE_CHECKING:
    import numpy as np

class OpenVLABridge:
    def __init__(self, device="cuda", dtype="bf16"):
        self.device = device
        
        # We use AutoModelForVision2Seq because OpenVLA's custom configuration
        # is compatible with this AutoClass as long as trust_remote_code=True.
        self.model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        self.processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b", 
            trust_remote_code=True
        )
        
        # Warm up the model to ensure the first step isn't delayed by kernel loading
        print("🚀 OpenVLA-7B Pilot Initialized on L4 GPU.")

    def get_vla_action(self, fpv_frame, strategic_instruction):
        """
        Takes the FPV image and the high-level VLM strategy to output low-level velocities.
        """
        # Format the prompt exactly as OpenVLA was trained
        prompt = f"Inhabitants: What action should the robot take to {strategic_instruction}?"
        
        # Preprocess image and text
        inputs = self.processor(prompt, fpv_frame, return_tensors="pt").to(self.device, torch.bfloat16)
        
        # Inference using the OpenVLA-specific action head
        with torch.no_grad():
            action = self.model.predict_action(**inputs, unnorm_key="bridge_orig")
            
        # OpenVLA returns a 7DoF action [x, y, z, roll, pitch, yaw, gripper]
        # We map these to the Go1 [vel_x, vel_y, yaw_rate]
        # We apply a slight scaling factor to ensure the robot doesn't lunge too fast
        vla_v_x = action[0] 
        vla_v_y = action[1]
        vla_yaw = action[5]

        return jp.array([vla_v_x, vla_v_y, vla_yaw], dtype=jp.float32)