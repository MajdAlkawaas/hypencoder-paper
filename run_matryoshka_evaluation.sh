#!/bin/bash
# This script automates the full evaluation campaign for a Matryoshka model.
# It evaluates all checkpoints from a training run against a list of specified datasets.

# Exit immediately if a command exits with a non-zero status.
set -e

# ==============================================================================
# --- CONFIGURATION: EDIT THIS SECTION ---
# ==============================================================================

# 1. Path to the directory containing your completed Matryoshka training run.
TRAINING_RUN_DIR="./trained_models/matryoshka-hyperhead-run-6-layers-atempt06"

# 2. The base directory where all evaluation results will be saved.
BASE_OUTPUT_DIR="./encoded_validation_sets/retrieved_items/matryoshka/matryoshka_attempt06"

# 3. Define the datasets to evaluate on.
#    KEY: ir_datasets name (in quotes). Use the '/judged' version for TREC DL.
#    VALUE: Path to the corresponding pre-encoded corpus (in quotes).
declare -A DATASETS_TO_EVALUATE
DATASETS_TO_EVALUATE=(
    ["beir/fiqa/test"]="./encoded_validation_sets/encoded_items/beir_fiqa_test.docs"
    ["beir/nfcorpus/test"]="./encoded_validation_sets/encoded_items/beir_nfcorpus_test.docs"
    ["beir/trec-covid"]="./encoded_validation_sets/encoded_items/beir_trec_covid.docs"
    ["beir/webis-touche2020/v2"]="./encoded_validation_sets/encoded_items/beir_webis-touche2020_v2.docs"
    ["beir/dbpedia-entity/test"]="./encoded_validation_sets/encoded_items/beir_dbpedia-entity_test.docs"
)




# 4. The Matryoshka dimensions you want to test.
MATRYOSHKA_DIMS=(128 256 512 768)

# 5. Hardware and performance settings.
# INFERENCE_DTYPE="fp16"
# PUT_EMBEDDINGS_ON_DEVICE="true" # Set to 'true' if corpus fits in VRAM, 'false' otherwise
# INFERENCE_BATCH_SIZE=131072

# ==============================================================================
# --- SCRIPT LOGIC: DO NOT EDIT BELOW THIS LINE ---
# ==============================================================================

echo "Starting Matryoshka evaluation campaign for training run: ${TRAINING_RUN_DIR}"
echo ""

# Get the list of dataset names (the keys of the associative array)
DATASET_NAMES=("${!DATASETS_TO_EVALUATE[@]}")

# Convert the Matryoshka dimensions array to a comma-separated string for Python
DIMS_STR=$(IFS=,; echo "${MATRYOSHKA_DIMS[*]}")

# --- Main Evaluation Loop: Iterate through each dataset ---
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    ENCODED_CORPUS_PATH="${DATASETS_TO_EVALUATE[$DATASET_NAME]}"
    
    # Sanitize the dataset name to create a valid directory name
    DATASET_NAME_CLEAN=$(echo "$DATASET_NAME" | tr '/' '_')
    
    # Construct the full output path for this dataset's results
    FULL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DATASET_NAME_CLEAN}"

    echo "========================================================================"
    echo "RUNNING EVALUATION FOR DATASET: $DATASET_NAME"
    echo "========================================================================"
    echo " > Using Encoded Corpus: $ENCODED_CORPUS_PATH"
    echo " > Results will be saved to: $FULL_OUTPUT_DIR"
    echo " > Testing Dimensions: ${MATRYOSHKA_DIMS[@]}"
    echo "------------------------------------------------------------------------"

    # Call the Matryoshka evaluation script for the current dataset.
    # It will internally handle finding all checkpoints and looping through dimensions.
    python -m hypencoder_cb.inference.matryoshka_retrieve \
        --training_run_dir="$TRAINING_RUN_DIR" \
        --encoded_item_path="$ENCODED_CORPUS_PATH" \
        --ir_dataset_name="$DATASET_NAME" \
        --matryoshka_dims="$DIMS_STR" \
        --base_output_dir="$FULL_OUTPUT_DIR" \
        # --dtype="$INFERENCE_DTYPE" \
        # --put_all_embeddings_on_device="$PUT_EMBEDDINGS_ON_DEVICE" \
        # --batch_size="$INFERENCE_BATCH_SIZE"

    echo " > Finished evaluation for dataset: $DATASET_NAME"
    echo ""
done

echo "========================================================================"
echo "          🎉 All Matryoshka evaluations complete! 🎉"
echo "Results for all datasets are stored in the base directory: ${BASE_OUTPUT_DIR}"
echo "========================================================================"