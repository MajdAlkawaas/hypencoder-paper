#!/bin/bash
set -e

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
TRAINING_RUN_DIR="./matryoshka-hyperhead-run-6-layers"
ENCODED_CORPUS_PATH="./encoded_validation_sets/encoded_items/msmarco-passage_trec_dl_2020.docs"
OUTPUT_DIR="./encoded_validation_sets/retrieved_items/matryoshka/msmarco_passage_trec_dl_2020/checkpoint-15720"
EVAL_DATASET="msmarco-passage/trec-dl-2020"

# The same dimensions you used for training
MATRYOSHKA_DIMS=(64 128 256 512 768)

# Hardware and performance settings
INFERENCE_DTYPE="fp16"
# Set to true if your fp16 corpus + model fits in VRAM
PUT_EMBEDDINGS_ON_DEVICE="true" 

# ==============================================================================
# --- SCRIPT LOGIC ---
# ==============================================================================
echo "Starting Matryoshka evaluation campaign..."
echo " > Training Run: $TRAINING_RUN_DIR"
echo " > Evaluation Dataset: $EVAL_DATASET"
echo " > Dimensions to test: ${MATRYOSHKA_DIMS[@]}"

# Convert bash array to a comma-separated string for the Python script
DIMS_STR=$(IFS=,; echo "${MATRYOSHKA_DIMS[*]}")

python hypencoder_cb/inference/matryoshka_retrieve.py \
    --training_run_dir="$TRAINING_RUN_DIR" \
    --encoded_item_path="$ENCODED_CORPUS_PATH" \
    --ir_dataset_name="$EVAL_DATASET" \
    --matryoshka_dims="$DIMS_STR" \
    --base_output_dir="$OUTPUT_DIR"


echo "🎉 Matryoshka evaluation campaign complete! 🎉"
echo "Results are in: ${BASE_OUTPUT_DIR}"