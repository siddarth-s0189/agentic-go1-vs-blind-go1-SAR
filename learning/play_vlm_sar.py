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
"""Run Go1 SAR rollout with GPT-4o vision-guided navigation."""

# Must set before any mujoco/jax imports
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("MUJOCO_GL", "egl")
_xla = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = _xla
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Prevent JAX from pre-allocating 75-90% of VRAM
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Alternatively, use a safe cap (60% of total VRAM)
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.60'
# Disable Triton GEMM to prevent some rare WSL2 memory leaks
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

# Orbax 0.7+ compatibility: bridge legacy Brax (PyTreeCheckpointer) to 2026 API
import orbax.checkpoint as ocp
if not hasattr(ocp, 'PyTreeCheckpointer'):
    ocp.PyTreeCheckpointer = lambda: ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

import jax
# Create a local folder to store the "frozen" compiled code
cache_dir = os.path.join(os.getcwd(), "jax_cache")
os.makedirs(cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", cache_dir)

import contextlib
import functools
import gc
import json
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from etils import epath
import math

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from mujoco_playground._src.locomotion.go1 import vlm_bridge

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "Go1JoystickSARStage5",
    "SAR environment (e.g. Go1JoystickSARStage5).",
)
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path",
    "logs/Go1JoystickRoughTerrain-20260211-164753-supply_box_stage1_rough/"
    "checkpoints/000206438400",
    "Path to trained policy checkpoint.",
)
_EPISODE_LENGTH = flags.DEFINE_integer(
    "episode_length",
    600,
    "Number of control steps per episode.",
)
_VLM_INTERVAL = flags.DEFINE_integer(
    "vlm_interval",
    60,
    "Steps between VLM queries (~0.5s at ctrl_dt=0.02).",
)
_OUTPUT_VIDEO = flags.DEFINE_string(
    "output_video",
    None,
    "Output path for rollout video (e.g. vlm_rollout.mp4).",
)
_VLM_LOG = flags.DEFINE_string(
    "vlm_log",
    None,
    "Output path for VLM responses log (.jsonl). Streams to disk to avoid RAM buildup.",
)
_CAMERA = flags.DEFINE_string(
    "camera",
    "birds_eye",
    "Camera for output video (birds_eye, robot_fpv).",
)
_VISION_MODE = flags.DEFINE_enum(
    "vision_mode",
    "combined",
    ["none", "fpv", "birds_eye", "combined"],
    "VLM input: none=blind (skip VLM), fpv=first-person only, birds_eye=top-down only, "
    "combined=side-by-side birds_eye|fpv. Video always shows side-by-side.",
)
_VLM_SWAP_AXES = flags.DEFINE_bool(
    "vlm_swap_axes",
    False,
    "Swap vel_x/vel_y from VLM. VLM is explicitly prompted vel_x=Forward so "
    "the direct mapping is correct by default. Enable only for debugging.",
)
_RUBBLE_SEED = flags.DEFINE_integer(
    "rubble_seed",
    42,
    "Random seed for Stage 5 Gaussian rubble generation.",
)
_N_RUBBLE = flags.DEFINE_integer(
    "n_rubble",
    120,
    "Number of rubble pieces for Stage 5 (100-150 recommended).",
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")


def main(argv):
  del argv

  # Procedurally generate Stage 5 rubble before the environment loads
  # (overwrites scene_mjx_feetonly_sar_stage5.xml with fresh Gaussian layout)
  if "SARStage5" in _ENV_NAME.value:
    vlm_bridge.generate_stage5_xml(
        seed=_RUBBLE_SEED.value,
        n_rubble=_N_RUBBLE.value,
    )

  # Environment config: override_command=[0,0,0] so we inject VLM commands
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env_cfg["impl"] = "jax"
  env_cfg_overrides = {"override_command": [0.0, 0.0, 0.0]}

  eval_env = registry.load(
      _ENV_NAME.value,
      config=env_cfg,
      config_overrides=env_cfg_overrides,
  )

  # Checkpoint path: resolve to step dir (e.g. .../checkpoints/000206438400)
# --- FIXED PATH LOGIC ---
  ppo_params = locomotion_params.brax_ppo_config(_ENV_NAME.value)
  ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
  
  # 1. If the path already has the config file, we are in the right place.
  if (ckpt_path / "ppo_network_config.json").exists():
      restore_checkpoint_path = ckpt_path
  else:
      # 2. If not, look for integer step folders (like 000206438400)
      step_dirs = [d for d in ckpt_path.iterdir() if d.is_dir() and d.name.isdigit()]
      if step_dirs:
          step_dirs.sort(key=lambda x: int(x.name))
          restore_checkpoint_path = step_dirs[-1]
      else:
          # 3. Fallback to whatever was passed
          restore_checkpoint_path = ckpt_path

  print(f"Loading checkpoint from: {restore_checkpoint_path}")
  # -------------------------

  # Load policy
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory,
  )
  inference_fn = ppo_checkpoint.load_policy(
      restore_checkpoint_path,
      network_factory=network_factory,
      deterministic=True,
  )
  jit_inference_fn = jax.jit(inference_fn)

  # VLM (used for inference when vision_mode != none; always for video HUD)
  vlm = vlm_bridge.GPT4pVLM()

  # Rollout loop (Python for-loop: VLM is blocking)
  rng = jax.random.PRNGKey(_SEED.value)
  state = eval_env.reset(rng)
  # Raw VLM output in world frame (forward=+X, lateral=+Y); updated every VLM query
  raw_vlm_world = np.array([0.0, 0.0, 0.0])

  # Y-bound for saturation guardrail (soft stop on lateral only)
  Y_BOUND = 1.45
  # Hardcoded 90-degree offset: VLM forward → World +X alignment
  THETA_OFFSET = 0.0

  def _quat_to_yaw(quat: np.ndarray) -> float:
    """Extract yaw (Z-rotation) from quaternion (w,x,y,z)."""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

  def _world_to_body_cmd(vx_world: float, vy_world: float, theta: float) -> tuple[float, float]:
    """Rotate world-frame (forward=+X, lateral=+Y) to body-frame velocities."""
    c, s = math.cos(theta), math.sin(theta)
    vx_body = vx_world * c + vy_world * s
    vy_body = -vx_world * s + vy_world * c
    return vx_body, vy_body

  def _apply_guardrails_and_rotate(s, raw_world: np.ndarray) -> jax.Array:
    """Saturation + world-to-body rotation + Automatic Nose-to-Goal Yaw."""
    qpos = np.array(jax.device_get(s.data.qpos))
    robot_x, robot_y = float(qpos[0]), float(qpos[1])
    vx, vy = float(raw_world[0]), float(raw_world[1])

    # 1. Lateral Guardrail (unchanged)
    if abs(robot_y) > Y_BOUND:
        if (robot_y > Y_BOUND and vy > 0) or (robot_y < -Y_BOUND and vy < 0):
            vy = 0.0

    # 2. Calculate Angle to Red Dot (at X=5.0, Y=0.0)
    dx = 5.0 - robot_x
    dy = 0.0 - robot_y
    target_yaw = math.atan2(dy, dx)
    
    # 3. Get Current Yaw and calculate error
    current_yaw = _quat_to_yaw(qpos[3:7])
    yaw_error = target_yaw - current_yaw
    yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi # Normalize

    # 4. Rotate Velocities (Theta is the robot's nose direction)
    theta = current_yaw + THETA_OFFSET
    vx_body, vy_body = _world_to_body_cmd(vx, vy, theta)

    # 5. Create final command: [fwd, lat, yaw]
    # We use (yaw_error * 2.0) to steer the nose, plus any delta from the VLM
    final_yaw_vel = (yaw_error * 0.5) + raw_world[2]
    
    return jp.array([vx_body, vy_body, final_yaw_vel])

  vlm_commands_per_step = [np.array(jax.device_get(_apply_guardrails_and_rotate(state, raw_vlm_world)))]

  if _VISION_MODE.value == "none":
    print("Vision mode: none (Blind) — VLM calls skipped, using zero command + guardrails.")

  # Streaming video: 640x480 per camera, write to disk during rollout (no RAM buildup)
  render_every = 2
  render_h, render_w = 480, 640  # Fixed lower resolution for memory-constrained hardware
  fps = 1.0 / eval_env.dt / render_every
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  def _render_frame(s, cmd):
    """Render birds_eye | front_vlm with HUD overlay at fixed 640x480 per view."""
    raw_birds = eval_env.render(
        s, height=render_h, width=render_w, camera="birds_eye", scene_option=scene_option
    )
    raw_fpv = eval_env.render(
        s, height=render_h, width=render_w, camera="front_vlm", scene_option=scene_option
    )
    frame_birds = np.array(jax.device_get(raw_birds))
    frame_fpv = np.array(jax.device_get(raw_fpv))
    wide_frame = np.hstack([frame_birds, frame_fpv])
    return vlm.draw_hud(wide_frame, cmd, camera="birds_eye")

  def _safe_cmd_idx(step: int) -> int:
    """Safe index into vlm_commands_per_step to prevent IndexError."""
    n = len(vlm_commands_per_step)
    return min(step, n - 1) if n > 0 else 0

  # VideoWriter context: wraps entire rollout for proper cleanup on physics errors
  if _OUTPUT_VIDEO.value:
    output_path = Path(_OUTPUT_VIDEO.value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_ctx = media.VideoWriter(
        str(output_path), shape=(render_h, render_w * 2), fps=fps
    )
  else:
    video_ctx = contextlib.nullcontext()

  vlm_log_file = None
  if _VLM_LOG.value:
    vlm_log_path = Path(_VLM_LOG.value)
    vlm_log_path.parent.mkdir(parents=True, exist_ok=True)
    vlm_log_file = open(vlm_log_path, "w", encoding="utf-8")

  try:
    with video_ctx as writer:
      # Write initial frame
      if writer is not None:
        cmd = vlm_commands_per_step[_safe_cmd_idx(0)]
        img = _render_frame(state, cmd)
        writer.add_image(img)
        del img
        gc.collect()

      for step_idx in range(_EPISODE_LENGTH.value - 1):
        # JAX: clear compilation caches every 100 steps to prevent bloat
        if step_idx > 0 and step_idx % 100 == 0:
          jax.clear_caches()

        # Query VLM every N steps (skip when vision_mode=none)
        if step_idx % _VLM_INTERVAL.value == 0 and _VISION_MODE.value != "none":
          vlm_frame = None
          if _VISION_MODE.value == "fpv":
            frame_fpv = eval_env.render(
                state, camera="front_vlm", height=render_h, width=render_w
            )
            vlm_frame = np.array(jax.device_get(frame_fpv))
          elif _VISION_MODE.value == "birds_eye":
            frame_birds = eval_env.render(
                state, camera="birds_eye", height=render_h, width=render_w
            )
            vlm_frame = np.array(jax.device_get(frame_birds))
          else:  # combined
            frame_birds = eval_env.render(
                state, camera="birds_eye", height=render_h, width=render_w
            )
            frame_fpv = eval_env.render(
                state, camera="front_vlm", height=render_h, width=render_w
            )
            frame_birds = np.array(jax.device_get(frame_birds))
            frame_fpv = np.array(jax.device_get(frame_fpv))
            vlm_frame = np.hstack([frame_birds, frame_fpv])
            del frame_birds, frame_fpv

          result = vlm.get_action(vlm_frame)
          del vlm_frame
          gc.collect()

          if _VLM_SWAP_AXES.value:
            raw_vlm_world = np.array(
                [result["vel_y"], result["vel_x"], result["yaw_rate"]]
            )
          else:
            raw_vlm_world = np.array(
                [result["vel_x"], result["vel_y"], result["yaw_rate"]]
            )

          # Stream VLM response to disk (no RAM accumulation)
          if vlm_log_file is not None:
            log_entry = {
                "step": step_idx,
                "vel_x": result["vel_x"],
                "vel_y": result["vel_y"],
                "yaw_rate": result["yaw_rate"],
                "explanation": result.get("explanation", ""),
            }
            vlm_log_file.write(json.dumps(log_entry) + "\n")
            vlm_log_file.flush()

          print(
              f"Step {step_idx}: VLM → fwd={result['vel_x']:+.3f}  "
              f"lat={result['vel_y']:+.3f}  yaw={result['yaw_rate']:+.3f}"
          )
          if result.get("explanation"):
            print(f"  Explanation: {result['explanation']}")

        current_vlm_command = _apply_guardrails_and_rotate(state, raw_vlm_world)
        vlm_commands_per_step.append(np.array(jax.device_get(current_vlm_command)))

        # Immutable state update (JAX)
        state = state.replace(
            info={**state.info, "command": current_vlm_command}
        )

        # Safely update the observation whether it's a Dict or a flat Array
        if isinstance(state.obs, dict):
          new_val = state.obs["state"].at[45:48].set(current_vlm_command)
          state = state.replace(obs={**state.obs, "state": new_val})
        else:
          new_val = state.obs.at[45:48].set(current_vlm_command)
          state = state.replace(obs=new_val)

        # Policy action and step
        rng, act_key = jax.random.split(rng)
        action = jit_inference_fn(state.obs, act_key)[0]
        state = eval_env.step(state, action)

        # Stream frame to video (avoids RAM buildup)
        if writer is not None and (step_idx + 1) % render_every == 0:
          cmd_idx = _safe_cmd_idx(step_idx + 1)
          cmd = vlm_commands_per_step[cmd_idx]
          img = _render_frame(state, cmd)
          writer.add_image(img)
          del img
          gc.collect()

  finally:
    if vlm_log_file is not None:
      vlm_log_file.close()
    if _OUTPUT_VIDEO.value:
      print(f"Video saved: {Path(_OUTPUT_VIDEO.value)}")

  print("Rollout complete.")


def run():
  app.run(main)


if __name__ == "__main__":
  run()
