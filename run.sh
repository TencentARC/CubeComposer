#!/bin/bash

# ========== Configuration ==========
# Path to local Wan2.2 model cache (base weights)
BASE_MODEL_PATH="./model-cache"  # update this to your local Wan2.2 model cache path if needed

# Path to training args.json (from your training run or provided checkpoint package)
ARGS_JSON=""  # will be auto-downloaded if not provided, but you can specify your own if needed

# Path to fine-tuned checkpoint
CHECKPOINT_PATH=""  # will be auto-downloaded if not provided, but you can specify your own if needed

# Path to ODV360 dataset root
ODV_ROOT_DIR="/path/to/ODVista360"  # update this to your local ODV360 dataset path

# Output directory for generated videos
TEST_OUTPUT_DIR="./test_outputs"

# Number of samples to test (None = all)
NUM_SAMPLES=20
START_IDX=0

# Inference settings
NUM_INFERENCE_STEPS=15
CFG_SCALE=5.0

python run.py \
  --base_model_path "${BASE_MODEL_PATH}" \
  --args_json "${ARGS_JSON}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --odv_root_dir "${ODV_ROOT_DIR}" \
  --output_dir "${TEST_OUTPUT_DIR}" \
  --num_samples ${NUM_SAMPLES} \
  --start_idx ${START_IDX} \
  --num_inference_steps ${NUM_INFERENCE_STEPS} \
  --cfg_scale ${CFG_SCALE} \
  --save_video_format mp4 \
  --trajectory_file ./assets/trajectory_rotation_fov90_2wp_20samples.json \
  --test_mode 3k

echo "Test completed! Check outputs in: ${TEST_OUTPUT_DIR}"
