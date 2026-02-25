#!/bin/bash
# Master Production: VLA-Baseline vs Proxy vs Hybrid

COMMON_ARGS="--env_name=Go1JoystickSARStage5 --episode_length=600 --n_rubble=60 --rubble_seed=42 --vlm_interval=60"

# 1. VLA-ONLY BASELINE (Reactive Only)
# Uses FPV + Static Instruction. No Strategist.
python3 learning/play_vlm_sar.py $COMMON_ARGS --architecture="hybrid" --skip_vlm --output_video="01_vla_baseline.mp4"

# 2. PROXY (VLM Soloist)
# Uses BEV only. High latency map-based planning.
python3 learning/play_vlm_sar.py $COMMON_ARGS --architecture="proxy" --output_video="02_proxy.mp4" --vlm_log="proxy_mission.jsonl"

# 3. HYBRID (Proposed Architecture)
# Uses BEV Strategy + FPV Execution.
python3 learning/play_vlm_sar.py $COMMON_ARGS --architecture="hybrid" --output_video="03_hybrid.mp4" --vlm_log="hybrid_mission.jsonl"