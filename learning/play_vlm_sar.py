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

import jax
# Create a local folder to store the "frozen" compiled code
cache_dir = os.path.join(os.getcwd(), "jax_cache")
os.makedirs(cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", cache_dir)

import functools
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from etils import epath
import jax
import jax.numpy as jp
import cv2
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
    "Go1JoystickSARStage1",
    "SAR environment (e.g. Go1JoystickSARStage1).",
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
_CAMERA = flags.DEFINE_string(
    "camera",
    "hero_view",
    "Camera for output video (hero_view, follow_side, birds_eye).",
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")


def main(argv):
  del argv

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

  # VLM
  vlm = vlm_bridge.GPT4pVLM()

  # Rollout loop (Python for-loop: VLM is blocking)
  rng = jax.random.PRNGKey(_SEED.value)
  state = eval_env.reset(rng)
  current_vlm_command = jp.array([0.0, 0.0, 0.0])
  trajectory = [state]
  vlm_commands_per_step = [np.array(jax.device_get(current_vlm_command))]

  for step_idx in range(_EPISODE_LENGTH.value - 1):
    # Query VLM every N steps
    if step_idx % _VLM_INTERVAL.value == 0:
      frame = eval_env.render(
          state,
          camera="front_vlm",
          height=480,
          width=640,
      )
      frame = np.array(jax.device_get(frame))
      result = vlm.get_action(frame)
      current_vlm_command = jp.array([
          result["vel_x"],
          result["vel_y"],
          result["yaw_rate"],
      ])
      print(
          f"Step {step_idx}: VLM vel_x={result['vel_x']:.3f} "
          f"vel_y={result['vel_y']:.3f} yaw_rate={result['yaw_rate']:.3f}"
      )

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
    trajectory.append(state)

  print("Rollout complete.")

  # Save video with HUD overlay
  if _OUTPUT_VIDEO.value:
    output_path = Path(_OUTPUT_VIDEO.value)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    traj_subset = trajectory[::render_every]
    vlm_commands_subset = vlm_commands_per_step[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    raw_frames = eval_env.render(
        traj_subset,
        height=480,
        width=640,
        camera=_CAMERA.value,
        scene_option=scene_option,
    )

    # Apply HUD overlay to each frame (corridor lines, velocity arrow, title)
    camera_name = _CAMERA.value or "hero_view"
    frames_with_hud = []
    for i, frame in enumerate(raw_frames):
      frame_np = np.array(jax.device_get(frame))
      cmd = vlm_commands_subset[i] if i < len(vlm_commands_subset) else vlm_commands_subset[-1]
      frame_with_hud = vlm.draw_hud(frame_np, cmd, camera=camera_name)
      frames_with_hud.append(frame_with_hud)

    media.write_video(str(output_path), frames_with_hud, fps=fps)
    print(f"Video saved: {output_path}")


def run():
  app.run(main)


if __name__ == "__main__":
  run()
