#!/bin/bash
# Hybrid rollout: VLM (BEV Strategist) + VLA (FPV Pilot)
# Ensure GOOGLE_CLOUD_PROJECT is set (source .env if present)
[ -f .env ] && set -a && source .env && set +a

COMMON_ARGS="--env_name=Go1JoystickSARStage5 --episode_length=600 --n_rubble=60 --rubble_seed=42 --vlm_interval=60"

python3 learning/play_vlm_sar.py $COMMON_ARGS --output_video="hybrid_rollout.mp4" --vlm_log="hybrid_mission.jsonl"
