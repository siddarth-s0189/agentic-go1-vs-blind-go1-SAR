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
"""VLM bridge for Go1 SAR navigation using GPT-4o vision."""

import base64
import json
import os
from typing import Dict

import cv2
import numpy as np
from openai import OpenAI


class GPT4pVLM:
  """Vision Language Model bridge for quadruped SAR navigation."""

  DEFAULT_MODEL = "gpt-4o-2024-08-06"

  def __init__(self) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
      raise ValueError("OPENAI_API_KEY environment variable is required")
    self._client = OpenAI(api_key=api_key)
    self._model = self.DEFAULT_MODEL

  def encode_image(self, image_array: np.ndarray) -> str:
    """Convert RGB numpy array to base64 JPEG string.

    Args:
      image_array: Numpy array of shape (H, W, 3), dtype uint8, RGB format.
        MuJoCo renderer returns RGB.

    Returns:
      Base64-encoded JPEG string.
    """
    # MuJoCo render() returns RGB; cv2.imencode expects BGR
    if len(image_array.shape) != 3 or image_array.shape[-1] != 3:
      raise ValueError(
          f"Expected RGB image (H, W, 3), got shape {image_array.shape}"
      )
    bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    _, jpeg_bytes = cv2.imencode(".jpg", bgr)
    return base64.b64encode(jpeg_bytes.tobytes()).decode("utf-8")

  def get_action(self, image: np.ndarray) -> Dict[str, float]:
    """Get navigation command from GPT-4o vision based on the current frame.

    Args:
      image: RGB image array from the robot's forward-facing camera.

    Returns:
      Dict with vel_x, vel_y, yaw_rate. On API failure, returns safe stop
      (all zeros).
    """
    safe_stop = {"vel_x": 0.0, "vel_y": 0.0, "yaw_rate": 0.0}

    try:
      base64_str = self.encode_image(image)
      response = self._client.chat.completions.create(
          model=self._model,
          messages=[
              {
                  "role": "system",
                  "content": (
                      "You are a quadruped robot navigator in a Search and "
                      "Rescue debris field. Your camera faces forward. Avoid "
                      "obstacles. You are constrained to a corridor width of "
                      "1.5m. Output a JSON object with keys: 'vel_x' (-1.0 "
                      "to 1.0), 'vel_y' (-1.0 to 1.0), 'yaw_rate' (-1.0 to "
                      "1.0). If you see an obstacle, navigate around it but "
                      "STAY within the corridor boundaries."
                  ),
              },
              {
                  "role": "user",
                  "content": [
                      {
                          "type": "text",
                          "text": "Analyze this frame and provide navigation vectors.",
                      },
                      {
                          "type": "image_url",
                          "image_url": {
                              "url": f"data:image/jpeg;base64,{base64_str}",
                          },
                      },
                  ],
              },
          ],
          response_format={
              "type": "json_schema",
              "json_schema": {
                  "name": "navigation_response",
                  "strict": True,
                  "schema": {
                      "type": "object",
                      "properties": {
                          "vel_x": {
                              "type": "number",
                              "description": "Forward velocity -1.0 to 1.0",
                          },
                          "vel_y": {
                              "type": "number",
                              "description": "Lateral velocity -1.0 to 1.0",
                          },
                          "yaw_rate": {
                              "type": "number",
                              "description": "Yaw rate -1.0 to 1.0",
                          },
                      },
                      "required": ["vel_x", "vel_y", "yaw_rate"],
                      "additionalProperties": False,
                  },
              },
          },
      )
      content = response.choices[0].message.content
      parsed = json.loads(content)
      return {
          "vel_x": float(np.clip(parsed["vel_x"], -1.0, 1.0)),
          "vel_y": float(np.clip(parsed["vel_y"], -1.0, 1.0)),
          "yaw_rate": float(np.clip(parsed["yaw_rate"], -1.0, 1.0)),
      }
    except Exception:
      return safe_stop
