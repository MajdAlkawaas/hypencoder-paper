from datasets import load_dataset
import logging
import sys
import jsonlines
from tqdm import tqdm


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout,
                    )

DATASET_NAME = "jfkback/hypencoder-msmarco-training-dataset"
OUTPUT_FILE  = "from_cached_msmarco_train_local.jsonl"

logging.info(f"Starting download of '{DATASET_NAME}'...")

# Load the dataset from the Hub. This will download and cache it.
# We set streaming=False to ensure the whole thing is downloaded.
dataset = load_dataset(DATASET_NAME, split="train", streaming=False, cache_dir="cached_datasets")

logging.info(f"Dataset downloaded successfully. Found {len(dataset)} samples.")
logging.info(f"Saving dataset to local JSONL file: '{OUTPUT_FILE}'...")

# This is a more robust way to write the file that avoids the pandas bug.
try:
    with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
        for sample in tqdm(dataset, desc="Writing to JSONL"):
            writer.write(sample)
    logging.info("Process complete. You can now use this local file for training.")

except Exception as e:
    logging.error(f"An error occurred during writing: {e}")

# ----------------------------------------------------------------

logging.info("Process complete. You can now use this local file for training.")