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
"""VLA bridge for Go1 SAR low-level pilot (Gemini 1.5 Robotics via Vertex AI).

Placeholder module: Vertex AI initialization and dummy get_vla_action.
Replace with actual Gemini 1.5 Robotics VLA integration.
Uses centralized vertex_config for project/location.
"""

import jax.numpy as jp
import numpy as np

from mujoco_playground._src.locomotion.go1 import vertex_config


class VertexVLABridge:
    """Placeholder VLA bridge using Gemini 1.5 Robotics via Vertex AI."""

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        model: str = "gemini-1.5-robotics",
    ):
        self.project = project or vertex_config.get_vertex_project()
        self.location = location or vertex_config.get_vertex_location()
        self.model_name = model
        self._client = vertex_config.get_vertex_client()

        if self._client is not None:
            print("Vertex AI client initialized for Gemini 1.5 Robotics (VLA).")
        else:
            print(
                "WARNING: Vertex AI client unavailable (set GOOGLE_CLOUD_PROJECT, "
                "pip install google-genai). Using dummy VLA actions."
            )

    def get_vla_action(
        self,
        fpv_frame: np.ndarray,
        strategic_instruction: str,
    ) -> tuple[jp.ndarray, np.ndarray]:
        """Get low-level velocity command from FPV image and strategic instruction.

        Placeholder: returns zero action. Replace with actual Gemini 1.5
        Robotics VLA inference via Vertex AI.

        Returns:
            Tuple of (scaled_action, raw_action) matching OpenVLABridge interface.
            scaled_action: [vel_x, vel_y, yaw] clipped to [-2, 2]
            raw_action: raw model output before scaling
        """
        # Placeholder for final Vertex Robotics Pilot call shape.
        _ = fpv_frame
        print(f"VLA received instruction: {strategic_instruction}")
        raw_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        scaled_action = jp.array([0.0, 0.0, 0.0], dtype=jp.float32)
        return scaled_action, raw_action
