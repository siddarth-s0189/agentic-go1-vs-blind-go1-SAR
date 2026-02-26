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
"""VLM bridge for Go1 SAR navigation using Gemini via Vertex AI.

Uses gemini-robotics-er-1.5-preview for the Strategist.
Shared Vertex config with vertex_vla_bridge via vertex_config.py.
"""

import json
import pathlib
import time
from typing import Dict

import cv2
import numpy as np

from mujoco_playground._src.locomotion.go1 import vertex_config

# ---------------------------------------------------------------------------
# Procedural environment generator (unchanged from legacy)
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


# Strategist system prompt: VLM sees BEV, provides instruction for VLA (FPV).
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
  "explanation": "string"
}"""


# Strategist model ID per user spec
STRATEGIST_MODEL = "gemini-robotics-er-1.5-preview"


class VertexVLM:
  """Strategist VLM: BEV → strategic instruction for VLA (FPV) pilot."""

  def __init__(self) -> None:
    self._client = vertex_config.get_vertex_client()
    if not self._client:
      raise ValueError(
          "Vertex AI client not available. Set GOOGLE_CLOUD_PROJECT (and "
          "GOOGLE_CLOUD_LOCATION if needed). pip install google-genai"
      )
    self._model = STRATEGIST_MODEL
    print(f"Vertex VLM (Strategist) initialized: {self._model}")

  def encode_image(self, image_array: np.ndarray) -> bytes:
    """Convert RGB numpy array to JPEG bytes for Vertex AI.

    Args:
      image_array: Numpy array of shape (H, W, 3), dtype uint8, RGB format.

    Returns:
      JPEG bytes (not base64).
    """
    if len(image_array.shape) != 3 or image_array.shape[-1] != 3:
      raise ValueError(
          f"Expected RGB image (H, W, 3), got shape {image_array.shape}"
      )
    bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    _, jpeg_bytes = cv2.imencode(".jpg", bgr)
    return jpeg_bytes.tobytes()

  def get_action(
      self,
      image: np.ndarray,
  ) -> Dict:
    """Get navigation command from Gemini vision based on the frame.

    Args:
      image: RGB birds-eye view image array (H, W, 3).
    Returns:
      Dict with exactly strategic_instruction and explanation.
    """
    safe_stop = {"strategic_instruction": "", "explanation": ""}

    try:
      image_bytes = self.encode_image(image)
      user_text = "Analyze this Birds-Eye View and provide a strategic instruction."

      from google.genai import types

      parts = [
          types.Part.from_text(text=user_text),
          types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
      ]
      response_schema = {
          "type": "object",
          "properties": {
              "strategic_instruction": {"type": "string", "description": "Instruction for low-level pilot"},
              "explanation": {"type": "string", "description": "Brief reasoning"},
          },
          "required": ["strategic_instruction", "explanation"],
      }

      t0 = time.perf_counter()
      response = self._client.models.generate_content(
          model=self._model,
          contents=types.UserContent(parts=parts),
          config=types.GenerateContentConfig(
              system_instruction=STRATEGIST_SYSTEM_PROMPT,
              response_mime_type="application/json",
              response_schema=response_schema,
              temperature=0.2,
          ),
      )
      elapsed = time.perf_counter() - t0
      content = response.text
      if elapsed > 2.0:
        print(f"  VLM latency: {elapsed:.2f}s")
      if not content:
        raise ValueError("Empty model response")
      parsed = json.loads(content)

      return {
          "strategic_instruction": str(parsed.get("strategic_instruction", "")),
          "explanation": str(parsed.get("explanation", "")),
      }

    except Exception as e:
      print(
          f"⚠️ VLM API failed (returning safe stop). Check Vertex config: {e!r}"
      )
      return safe_stop

  def draw_hud(
      self,
      image: np.ndarray,
      strategic_instruction: str,
      camera: str = "birds_eye",
  ) -> np.ndarray:
    """Draw minimal HUD overlay: title bar only.

    Args:
      image: RGB image array (H, W, 3).
      strategic_instruction: Current strategist message for display compatibility.
      camera: Camera name (kept for API compat).

    Returns:
      RGB image with title bar only.
    """
    img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    white = (255, 255, 255)

    cv2.putText(
        img,
        "Gemini AUTONOMOUS NAV",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        white,
        2,
        cv2.LINE_AA,
    )
    _ = (strategic_instruction, camera)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
