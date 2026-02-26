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
import pathlib
from typing import Dict

import cv2
import numpy as np
from openai import OpenAI


# ---------------------------------------------------------------------------
# Procedural environment generator
# ---------------------------------------------------------------------------

def generate_stage5_xml(seed: int = 42, n_rubble: int = 120) -> str:
  """Generate and write Stage 5 XML with a realistic debris field.

  Fills the FOV rectangle (~5m x 3m) with rubble: pebbles (60%), debris (30%),
  and large obstacles (10%). Uses no-overlap placement and a 0.6m safety zone
  around the robot start. All pieces use matte concrete appearance.

  Args:
    seed: Random seed for reproducible layouts.
    n_rubble: Target number of rubble pieces.

  Returns:
    Absolute path string of the written XML file.
  """
  rng = np.random.RandomState(seed)

  # Size distribution: 60% pebbles (0.03–0.06), 30% debris (0.07–0.12),
  # 10% obstacles (0.13–0.15). Place larger pieces first for better packing.
  n_pebbles = int(n_rubble * 0.60)
  n_debris = int(n_rubble * 0.30)
  n_obstacles = n_rubble - n_pebbles - n_debris

  specs: list[tuple[float, str]] = []
  for _ in range(n_pebbles):
    sz = float(rng.uniform(0.03, 0.06))
    geom_type = "sphere" if rng.random() < 0.30 else "box"
    specs.append((sz, geom_type))
  for _ in range(n_debris):
    sz = float(rng.uniform(0.07, 0.12))
    geom_type = "sphere" if rng.random() < 0.30 else "box"
    specs.append((sz, geom_type))
  for _ in range(n_obstacles):
    sz = float(rng.uniform(0.13, 0.15))
    geom_type = "sphere" if rng.random() < 0.30 else "box"
    specs.append((sz, geom_type))

  specs.sort(key=lambda s: -s[0])  # Largest first for packing

  # Edge-to-edge: full 5m x 3m FOV bounds; only exclusion is 0.6m around spawn (0,0)
  FOV_X_MIN, FOV_X_MAX = 0.1, 4.9
  FOV_Y_MIN, FOV_Y_MAX = -1.45, 1.45
  SAFETY_RADIUS = 0.6
  MIN_GAP = 0.05
  MAX_PLACE_ATTEMPTS = 200

  placed: list[tuple[float, float, float, str]] = []  # (x, y, size, geom_type)

  for size, geom_type in specs:
    found = False
    for _ in range(MAX_PLACE_ATTEMPTS):
      x = float(rng.uniform(FOV_X_MIN, FOV_X_MAX))
      y = float(rng.uniform(FOV_Y_MIN, FOV_Y_MAX))

      if x * x + y * y < SAFETY_RADIUS * SAFETY_RADIUS:
        continue

      overlap = False
      for (ex, ey, es, _) in placed:
        dist_sq = (x - ex) ** 2 + (y - ey) ** 2
        min_dist = size + es + MIN_GAP
        if dist_sq < min_dist * min_dist:
          overlap = True
          break
      if overlap:
        continue

      placed.append((x, y, size, geom_type))
      found = True
      break

  rubble_lines: list[str] = []
  for idx, (x, y, size, geom_type) in enumerate(placed, start=1):
    z = size  # Base at z=0: center at z=size for box/sphere
    rubble_lines.append(
        f'    <body name="rubble{idx}" pos="{x:.3f} {y:.3f} {z:.3f}">'
        f'<geom name="rubble{idx}_geom" type="{geom_type}" '
        f'size="{size:.3f} {size:.3f} {size:.3f}" '
        f'material="debris_matte" '
        f'condim="3" margin="0.02" gap="0.01" '
        f'contype="1" conaffinity="1" priority="1" group="0"/></body>'
    )

  rubble_xml = "\n".join(rubble_lines)

  xml_content = f"""<mujoco model="go1 SAR stage 5 - Debris Field">
  <include file="go1_mjx_feetonly_sar.xml"/>

  <statistic center="0 0 0.1" extent="0.8" meansize="0.04"/>

  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <rgba force="1 0 0 1"/>
    <global azimuth="120" elevation="-20"/>
    <map force="0.01"/>
    <scale forcewidth="0.3" contactwidth="0.5" contactheight="0.2"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
    <texture type="2d" name="groundplane" file="assets/rocky_texture.png"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance=".8"/>
    <material name="floor_dark" rgba="0.1 0.1 0.1 1"/>
    <material name="concrete" rgba="0.3 0.3 0.3 1"/>
    <material name="debris_matte" rgba="0.25 0.25 0.25 1" specular="0" shininess="0"/>
    <material name="rust_metal" rgba="0.5 0.1 0.05 1" specular="0.8"/>
    <material name="charcoal_wall" rgba="0.15 0.15 0.15 1"/>
    <hfield name="hfield" file="assets/hfield.png" size="10 10 .05 1.0"/>
  </asset>

  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" type="directional" castshadow="true"/>
    <geom name="floor" type="hfield" hfield="hfield" material="groundplane" contype="1" conaffinity="0" priority="1"
      friction="1.0"/>

    <!-- Robot spawns at (0,0) facing +X. Goal at x=5. Boundaries via software. -->
    <site name="goal" pos="5 0 0" rgba="1 0 0 1" size="0.1"/>

    <camera name="birds_eye" pos="2.5 0 3.5" xyaxes="1 0 0 0 1 0" fovy="60"/>

    <!-- ENV 5: Realistic debris field — {len(placed)} pieces, no-overlap, seed={seed} -->
{rubble_xml}
  </worldbody>

  <include file="sensor_feet.xml"/>

  <keyframe>
    <key name="home" qpos="0 0 0.35 1 0 0 0 0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"
      ctrl="0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"/>
  </keyframe>
</mujoco>
"""

  xml_path = pathlib.Path(__file__).parent / "xmls" / "scene_mjx_feetonly_sar_stage5.xml"
  xml_path.write_text(xml_content)
  print(
      f"[Stage5] Generated {len(placed)} rubble pieces (pebbles/debris/obstacles, "
      f"no-overlap, seed={seed}) → {xml_path}"
  )
  return str(xml_path)


# Soloist (proxy) and Strategist (hybrid) system prompts.
SOLOIST_SYSTEM_PROMPT = """You are an expert navigator for a Go1 quadruped. You are receiving a Birds-Eye View (BEV). Use this as your GLOBAL MAP to identify the clearest path to the goal (Red Dot) across the whole field.

### SPATIAL ORIENTATION
- The robot is currently facing the RIGHT side of the frame.
- TOP of the image = Robot's LEFT (NEGATIVE vel_y).
- BOTTOM of the image = Robot's RIGHT (POSITIVE vel_y).
- RIGHT of the image = Robot's FORWARD (POSITIVE vel_x).

### NAVIGATION CONSTRAINTS
- The navigable corridor is Y = -1.5 to 1.5. Stay centered.
- Walk over small debris, but navigate around blocks taller than your knees.

### TEMPORAL AWARENESS
- IMPORTANT: You operate in a high-latency loop. Every command is executed for 1.2 seconds (60 physics steps).
- At vel_x = 0.8, you travel 1 meter before the next frame. 
- Imagine your position 1.2s in the future and ensure your chosen speed won't carry you into an obstacle.

### GOAL
- Reach the red marker at the far end of the field (World +X).

### OUTPUT FORMAT
Respond ONLY with a JSON object:
{
  "vel_x": float,
  "vel_y": float,
  "yaw_rate": float,
  "explanation": "string"
}"""

STRATEGIST_SYSTEM_PROMPT = """You are the High-Level Strategist for a Go1 quadruped. You receive a Birds-Eye View (BEV) map. Your role is to provide a Strategic Instruction to a low-level VLA pilot that sees the First-Person View (FPV) and handles reactive dodging.

### SPATIAL ORIENTATION
- Image TOP = Robot's LEFT (NEGATIVE Y-axis).
- Image BOTTOM = Robot's RIGHT (POSITIVE Y-axis).
- Image RIGHT = Robot's FORWARD (POSITIVE X-axis).

### TACTICAL INSTRUCTION LOGIC
- Analyze the long-term path for the next 5 meters. 
- Identify the optimal 'lane' (Left, Center, or Right) that avoids major rubble clusters.
- Your instruction must guide the low-level pilot on how to prioritize its movements to stay on your chosen global path.

### TEMPORAL HORIZON
- Your instruction must remain valid for 1.2 seconds of travel. 
- Forecast the safest trajectory so the pilot doesn't enter a dead-end.

### OUTPUT FORMAT
Respond ONLY with a JSON object:
{
  "strategic_instruction": "string",
  "target_lane": "left/center/right",
  "explanation": "string"
}"""


class GPT4pVLM:
  """Vision Language Model bridge for quadruped SAR navigation."""

  DEFAULT_MODEL = "gpt-4o"

  def __init__(self, architecture: str = "proxy") -> None:
    if architecture not in ("proxy", "hybrid"):
      raise ValueError(
          f"architecture must be 'proxy' or 'hybrid', got {architecture!r}"
      )
    self._architecture = architecture
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

  def get_action(
      self,
      image: np.ndarray,
      physical_feedback: str | None = None,
  ) -> Dict:
    """Get navigation command from GPT-4o vision based on the frame(s).

    Args:
      image: RGB birds-eye view image array (H, W, 3). Architecture handles
        routing: caller passes birds_eye for both proxy and hybrid.
      physical_feedback: Optional proprioceptive feedback to append to the
        prompt (e.g. "Robot is stuck").

    Returns:
      Dict with vel_x, vel_y, yaw_rate, explanation (proxy) or
      strategic_instruction, target_lane, explanation (hybrid). On API failure,
      returns safe stop.
    """
    safe_stop_proxy = {"vel_x": 0.0, "vel_y": 0.0, "yaw_rate": 0.0, "explanation": ""}
    safe_stop_hybrid = {"strategic_instruction": "", "target_lane": "center", "explanation": ""}

    try:
      base64_str = self.encode_image(image)
      user_text = "Analyze this Birds-Eye View and provide navigation."
      if physical_feedback:
        user_text += f" {physical_feedback}"

      if self._architecture == "proxy":
        system_prompt = SOLOIST_SYSTEM_PROMPT
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "navigation_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "vel_x": {"type": "number", "description": "Forward velocity -1.0 to 1.0"},
                        "vel_y": {"type": "number", "description": "Lateral velocity -1.0 to 1.0"},
                        "yaw_rate": {"type": "number", "description": "Yaw rate -1.0 to 1.0"},
                        "explanation": {"type": "string", "description": "Brief reasoning"},
                    },
                    "required": ["vel_x", "vel_y", "yaw_rate", "explanation"],
                    "additionalProperties": False,
                },
            },
        }
      else:
        system_prompt = STRATEGIST_SYSTEM_PROMPT
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "strategist_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "strategic_instruction": {"type": "string", "description": "Instruction for low-level pilot"},
                        "target_lane": {"type": "string", "description": "left, center, or right"},
                        "explanation": {"type": "string", "description": "Brief reasoning"},
                    },
                    "required": ["strategic_instruction", "target_lane", "explanation"],
                    "additionalProperties": False,
                },
            },
        }

      response = self._client.chat.completions.create(
          model=self._model,
          messages=[
              {"role": "system", "content": system_prompt},
              {
                  "role": "user",
                  "content": [
                      {"type": "text", "text": user_text},
                      {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}},
                  ],
              },
          ],
          response_format=response_format,
      )
      content = response.choices[0].message.content
      parsed = json.loads(content)

      if self._architecture == "proxy":
        return {
            "vel_x": float(np.clip(parsed["vel_x"], -1.0, 1.0)),
            "vel_y": float(np.clip(parsed["vel_y"], -1.0, 1.0)),
            "yaw_rate": float(np.clip(parsed["yaw_rate"], -1.0, 1.0)),
            "explanation": str(parsed.get("explanation", "")),
        }
      else:
        lane = str(parsed.get("target_lane", "center")).lower()
        if lane not in ("left", "center", "right"):
          lane = "center"
        return {
            "strategic_instruction": str(parsed.get("strategic_instruction", "")),
            "target_lane": lane,
            "explanation": str(parsed.get("explanation", "")),
        }
    except Exception as e:
      print(
          f"⚠️ VLM API failed (returning safe stop). Check OPENAI_API_KEY and network: {e!r}"
      )
      return safe_stop_proxy if self._architecture == "proxy" else safe_stop_hybrid

  def draw_hud(
      self,
      image: np.ndarray,
      current_vlm_command: np.ndarray,
      camera: str = "birds_eye",
  ) -> np.ndarray:
    """Draw minimal HUD overlay: title bar only.

    Red boundary lines and velocity arrow have been removed. The glass-box
    walls in the MuJoCo scene replace the 2-D corridor overlay, and VLM
    commands are printed to stdout rather than overlaid on the frame.

    Args:
      image: RGB image array (H, W, 3).
      current_vlm_command: Array [vel_x, vel_y, yaw_rate] (kept for API compat).
      camera: Camera name (kept for API compat).

    Returns:
      RGB image with title bar only.
    """
    img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    white = (255, 255, 255)

    cv2.putText(
        img,
        "GPT-4o AUTONOMOUS NAV",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        white,
        2,
        cv2.LINE_AA,
    )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
