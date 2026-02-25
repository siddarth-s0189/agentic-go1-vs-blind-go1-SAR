#!/bin/bash
# Master Production: Blind vs Egocentric vs Allocentric vs Combined

COMMON_ARGS="--env_name=Go1JoystickSARStage5 --episode_length=600 --n_rubble=60 --rubble_seed=42 --vlm_interval=60"

# 1. BLIND (The Baseline)
python3 learning/play_vlm_sar.py $COMMON_ARGS --vision_mode="none" --output_video="01_blind.mp4"

# 2. EGOCENTRIC (The FPV Test)
python3 learning/play_vlm_sar.py $COMMON_ARGS --vision_mode="fpv" --output_video="02_egocentric.mp4" --vlm_log="egocentric_mission.jsonl"

# 3. ALLOCENTRIC (The Birds-Eye Test)
python3 learning/play_vlm_sar.py $COMMON_ARGS --vision_mode="birds_eye" --output_video="03_allocentric.mp4" --vlm_log="allocentric_mission.jsonl"

# 4. COMBINED (The Final Demo)
python3 learning/play_vlm_sar.py $COMMON_ARGS --vision_mode="combined" --output_video="04_combined_final.mp4" --vlm_log="final_mission.jsonl"