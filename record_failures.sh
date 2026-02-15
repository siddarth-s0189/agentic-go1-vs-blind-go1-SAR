# Birds eye camera
for STAGE in 1 2 3 5; do
  MUJOCO_GL=egl python3 learning/train_jax_ppo.py \
    --env_name=Go1JoystickSARStage$STAGE \
    --play_only=True \
    --load_checkpoint_path=logs/Go1JoystickRoughTerrain-20260211-164753-supply_box_stage1_rough/checkpoints \
    --output_video=rollout_env${STAGE}_blind_birds_eye.mp4 \
    --camera=birds_eye
done


# Follow side camera
for STAGE in 1 2 3 5; do
  MUJOCO_GL=egl python3 learning/train_jax_ppo.py \
    --env_name=Go1JoystickSARStage$STAGE \
    --play_only=True \
    --load_checkpoint_path=logs/Go1JoystickRoughTerrain-20260211-164753-supply_box_stage1_rough/checkpoints \
    --output_video=rollout_env${STAGE}_blind_follow.mp4 \
    --camera=follow_side
done


# # Hero view camera
# for STAGE in 1 2 3 4 5; do
#   MUJOCO_GL=egl python3 learning/train_jax_ppo.py \
#     --env_name=Go1JoystickSARStage$STAGE \
#     --play_only=True \
#     --load_checkpoint_path=logs/Go1JoystickRoughTerrain-20260211-164753-supply_box_stage1_rough/checkpoints \
#     --output_video=rollout_env${STAGE}_blind_hero.mp4 \
#     --camera=hero_view
# done

# excluding Stage 4 due to persistent JAX JIT compilation issue