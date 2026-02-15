#!/bin/bash

# Path to your checkpoints folder
BASE_CKPT="logs/Go1JoystickRoughTerrain-20260211-164753-supply_box"

# Define your 4 stages
STAGES=("stage1" "stage2" "stage3" "stage5")
STEPS=("000206438400" "000206438400" "000206438400" "000206438400")

mkdir -p hackathon_delivery

echo "🎬 PRODUCTION START: 8 Videos incoming..."

# --- FIRST PASS: All Birds Eye Views ---
echo "🛰️ PHASE 1: Rendering Birds Eye views for all stages..."
for i in "${!STAGES[@]}"; do
  STAGE=${STAGES[$i]}
  STEP=${STEPS[$i]}
  CKPT_PATH="${BASE_CKPT}_${STAGE}_rough/checkpoints/${STEP}"
  OUT_FILE="hackathon_delivery/Go1_SAR_${STAGE}_birds_eye.mp4"

  echo "🚀 STAGE: $STAGE | VIEW: birds_eye"
  MUJOCO_GL=egl python3 learning/play_vlm_sar.py \
    --load_checkpoint_path="$CKPT_PATH" \
    --camera="birds_eye" \
    --output_video="$OUT_FILE" \
    --episode_length=600 \
    --vlm_interval=60
done

# --- SECOND PASS: Specific Views (Hero for Stage 2, Follow for others) ---
echo "🤖 PHASE 2: Rendering Hero/Follow views..."
for i in "${!STAGES[@]}"; do
  STAGE=${STAGES[$i]}
  STEP=${STEPS[$i]}
  CKPT_PATH="${BASE_CKPT}_${STAGE}_rough/checkpoints/${STEP}"
  
  # Logic: If stage2 use hero_view, else use follow_side
  if [ "$STAGE" == "stage2" ]; then
    VIEW="hero_view"
  else
    VIEW="follow_side"
  fi

  OUT_FILE="hackathon_delivery/Go1_SAR_${STAGE}_${VIEW}.mp4"

  echo "🚀 STAGE: $STAGE | VIEW: $VIEW"
  MUJOCO_GL=egl python3 learning/play_vlm_sar.py \
    --load_checkpoint_path="$CKPT_PATH" \
    --camera="$VIEW" \
    --output_video="$OUT_FILE" \
    --episode_length=600 \
    --vlm_interval=60
done

echo "🎉 DONE. All 8 videos should be in 'hackathon_delivery'."