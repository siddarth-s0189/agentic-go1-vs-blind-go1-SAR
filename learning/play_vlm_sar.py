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
"""Run Go1 SAR rollout with hybrid architecture: VLM (BEV Strategist) + VLA (FPV Pilot)."""

# Must set before any mujoco/jax imports
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("MUJOCO_GL", "egl")

# JAX/GPU: prevent VRAM race with PyTorch (VLA)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
# Consolidated XLA flags (avoid duplicate/circular registration on Colab)
_xla = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_triton_gemm=false" not in _xla:
    _xla = (_xla + " --xla_gpu_enable_triton_gemm=false").strip()
os.environ["XLA_FLAGS"] = _xla

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

import cv2
import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from mujoco_playground._src.locomotion.go1.vertex_vlm_bridge import (
    VertexVLM,
    generate_stage5_xml,
)
from mujoco_playground._src.locomotion.go1.vertex_vla_bridge import VertexVLABridge

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
    "Steps between VLM (Strategist) queries (~0.5s at ctrl_dt=0.02).",
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


RENDER_EVERY = 5  # Frames between video renders (decoupled from VLM interval)
VLA_ACTION_REPEAT = 5  # 10 Hz: call VLA every 5 steps (ctrl_dt≈0.02s -> 10 Hz)


def main(argv):
  del argv

  # Procedurally generate Stage 5 rubble before the environment loads
  if "SARStage5" in _ENV_NAME.value:
    generate_stage5_xml(
        seed=_RUBBLE_SEED.value,
        n_rubble=_N_RUBBLE.value,
    )

  # Environment config: override_command=[0,0,0] so we inject VLM/VLA commands
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env_cfg["impl"] = "jax"
  env_cfg_overrides = {"override_command": [0.0, 0.0, 0.0]}

  eval_env = registry.load(
      _ENV_NAME.value,
      config=env_cfg,
      config_overrides=env_cfg_overrides,
  )

  # Checkpoint path: resolve to step dir
  ppo_params = locomotion_params.brax_ppo_config(_ENV_NAME.value)
  ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
  if (ckpt_path / "ppo_network_config.json").exists():
      restore_checkpoint_path = ckpt_path
  else:
      step_dirs = [d for d in ckpt_path.iterdir() if d.is_dir() and d.name.isdigit()]
      if step_dirs:
          step_dirs.sort(key=lambda x: int(x.name))
          restore_checkpoint_path = step_dirs[-1]
      else:
          restore_checkpoint_path = ckpt_path

  print(f"Loading checkpoint from: {restore_checkpoint_path}")

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
  rng = jax.random.PRNGKey(_SEED.value)
  state = eval_env.reset(rng)

  # Override starting orientation to face East (+X), so FPV points at target (5, 0)
  new_qpos = state.data.qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))
  state = state.replace(data=state.data.replace(qpos=new_qpos))

  # Jit-ed step: obs update + policy + env.step in one opaque call
  is_dict_obs = isinstance(state.obs, dict)

  def _step_with_command(state, vlm_cmd: jax.Array, rng):
    """Update obs with command, run policy, step env."""
    if is_dict_obs:
      new_obs_val = state.obs["state"].at[45:48].set(vlm_cmd)
      obs = {**state.obs, "state": new_obs_val}
    else:
      obs = state.obs.at[45:48].set(vlm_cmd)
    state = state.replace(info={**state.info, "command": vlm_cmd}, obs=obs)
    rng, act_key = jax.random.split(rng)
    action = inference_fn(state.obs, act_key)[0]
    return eval_env.step(state, action), rng

  jit_step_with_command = jax.jit(_step_with_command)

  # Hybrid: VLM (Strategist) on BEV + VLA (Pilot) on FPV
  vlm = VertexVLM()
  vla = VertexVLABridge()
  strat_msg = "Proceed forward carefully"

  print("Architecture: hybrid (VLM=BEV Strategist @vlm_interval, VLA=FPV Pilot @10Hz)")

  # Initial handshake: strategist instruction first, then pilot action.
  render_h, render_w = 480, 640
  bev0 = eval_env.render(state, camera="birds_eye", height=render_h, width=render_w)
  bev0 = np.array(jax.device_get(bev0))
  initial_result = vlm.get_action(bev0)
  strat_msg = initial_result.get("strategic_instruction", "") or strat_msg
  del bev0

  frame0 = eval_env.render(state, camera="front_vlm", height=render_h, width=render_w)
  frame0 = np.array(jax.device_get(frame0))
  init_scaled, init_raw = vla.get_vla_action(frame0, strat_msg)
  current_vla_action = np.array(jax.device_get(init_scaled))
  current_raw_action = init_raw.copy()
  vlm_commands_per_step = [current_vla_action.copy()]

  # Streaming video
  fps = 1.0 / (eval_env.dt * RENDER_EVERY)
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  def _render_frame(s, strategic_instruction):
    """Render birds_eye | front_vlm with HUD overlay."""
    raw_birds = eval_env.render(
        s, height=render_h, width=render_w, camera="birds_eye", scene_option=scene_option
    )
    raw_fpv = eval_env.render(
        s, height=render_h, width=render_w, camera="front_vlm", scene_option=scene_option
    )
    frame_birds = np.array(jax.device_get(raw_birds))
    frame_fpv = np.array(jax.device_get(raw_fpv))
    wide_frame = np.hstack([frame_birds, frame_fpv])
    return vlm.draw_hud(wide_frame, strategic_instruction, camera="birds_eye")

  def _safe_cmd_idx(step: int) -> int:
    n = len(vlm_commands_per_step)
    return min(step, n - 1) if n > 0 else 0

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
      if writer is not None:
        img = _render_frame(state, strat_msg)
        img = np.array(jax.device_get(img)).astype(np.uint8)
        writer.add_image(img)
        del img
        gc.collect()

      try:
        for step_idx in range(_EPISODE_LENGTH.value - 1):
          if step_idx > 0 and step_idx % 100 == 0:
            jax.clear_caches()

          # Query VLM (Strategist) every N steps
          if step_idx % _VLM_INTERVAL.value == 0:
            vlm_frame = np.array(jax.device_get(eval_env.render(
                state, camera="birds_eye", height=render_h, width=render_w
            )))
            print(f"Step {step_idx} | VLM frame: {vlm_frame.shape}")

            result = vlm.get_action(vlm_frame)
            strat_msg = result.get("strategic_instruction", "") or "Proceed forward carefully"

            if vlm_log_file is not None:
              log_entry = {
                  "step": step_idx,
                  "strategic_instruction": result.get("strategic_instruction", ""),
                  "explanation": result.get("explanation", ""),
              }
              vlm_log_file.write(json.dumps(log_entry) + "\n")
              vlm_log_file.flush()
            print(
                f"Step {step_idx}: VLM (Strategist) → {strat_msg[:60]}..."
            )
            print(f"  Explanation: {result.get('explanation', '') or '(none)'}")

            del vlm_frame
            gc.collect()

          # VLA (Pilot) action repeat
          if step_idx % VLA_ACTION_REPEAT == 0:
            if step_idx > 0:
              fpv_frame = eval_env.render(
                  state, camera="front_vlm", height=render_h, width=render_w
              )
              fpv_frame = np.array(jax.device_get(fpv_frame))
              scaled_cmd, raw_cmd = vla.get_vla_action(
                  fpv_frame, strat_msg
              )
              current_vla_action = np.array(jax.device_get(scaled_cmd))
              current_raw_action = raw_cmd
            current_vlm_command = current_vla_action.copy()
            print(
                f"STEP {step_idx} | Raw: {current_raw_action[:3]} | Scaled: {current_vla_action[:3]}",
                flush=True,
            )
          else:
            current_vlm_command = current_vla_action.copy()

          vlm_commands_per_step.append(np.array(jax.device_get(current_vlm_command)))
          vlm_cmd_jax = jp.array(jax.device_get(current_vlm_command))
          state, rng = jit_step_with_command(state, vlm_cmd_jax, rng)

          if writer is not None and step_idx % RENDER_EVERY == 0:
            cmd_idx = _safe_cmd_idx(step_idx + 1)
            _ = vlm_commands_per_step[cmd_idx]
            img = _render_frame(state, strat_msg)
            img = np.array(jax.device_get(img)).astype(np.uint8)
            writer.add_image(img)
            del img
            gc.collect()

      except KeyboardInterrupt:
        print("KeyboardInterrupt: cleaning up (env, video writer)...")

  finally:
    if vlm_log_file is not None:
      vlm_log_file.close()
    if hasattr(eval_env, "close"):
      eval_env.close()
    if _OUTPUT_VIDEO.value:
      print(f"Video saved: {Path(_OUTPUT_VIDEO.value)}")

  print("Rollout complete.")


def run():
  app.run(main)


if __name__ == "__main__":
  run()
