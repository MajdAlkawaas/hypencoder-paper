# In hypencoder_cb/inference/evaluate_all_checkpoints.py

import fire
import torch
import os
import glob
from typing import List
from pathlib import Path

# Import necessary components from the existing codebase
from .retrieve import HypencoderRetriever, do_eval_and_pretty_print
from .shared import load_encoded_items_from_disk, retrieve_for_ir_dataset_queries

def evaluate_all_checkpoints(
    training_run_dir: str,
    encoded_item_path: str,
    ir_dataset_name: str,
    base_output_dir: str,
    dtype: str = "float32",
    batch_size: int = 131072,
    top_k: int = 1000,
    put_all_embeddings_on_device: bool = True
):
    """
    Automatically discovers and evaluates all checkpoints from a training run
    against a single, pre-loaded encoded corpus using a batched retrieval process.
    """

    # =================================================================================
    # LOGIC FOR ONE-TIME DESERIALIZATION
    # This block runs only once. It loads the corpus into CPU RAM using the correct
    # dtype schema, ensuring memory efficiency.
    # =================================================================================
    print("---" * 10)
    print(f"STAGE 1: Deserializing corpus ONCE from: {encoded_item_path}")

    preloaded_encoded_items = list(load_encoded_items_from_disk(
        encoded_item_path,
        target_dtype=dtype
    ))

    print(f"Loaded {len(preloaded_encoded_items)} documents into CPU RAM.")
    print("---" * 10)

    # --- Automatic Checkpoint Discovery ---
    print(f"Discovering checkpoints in: {training_run_dir}")
    checkpoint_paths = glob.glob(os.path.join(training_run_dir, "checkpoint-*"))
    checkpoint_paths.sort(key=lambda x: int(x.split('-')[-1]))
    if os.path.exists(os.path.join(training_run_dir, "pytorch_model.bin")):
        checkpoint_paths.append(training_run_dir)

    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in directory: {training_run_dir}")
    print(f"Found {len(checkpoint_paths)} checkpoints to evaluate.")

    # --- Loop through checkpoints ---
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print("\\n" + "="*50)
        print(f"EVALUATING CHECKPOINT {i+1}/{len(checkpoint_paths)}: {checkpoint_path}")

        checkpoint_name = Path(checkpoint_path).name
        output_dir_for_checkpoint = Path(base_output_dir) / checkpoint_name
        output_dir_for_checkpoint.mkdir(parents=True, exist_ok=True)

        # =================================================================================
        # LOGIC FOR BATCHED RETRIEVAL
        # The key is `put_all_embeddings_on_device=False`. This tells the retriever
        # to keep the preloaded_encoded_items in CPU RAM and only move small
        # batches (of size `batch_size`) to the GPU for scoring. This prevents OOM errors.
        # =================================================================================
        retriever = HypencoderRetriever(
            model_name_or_path=checkpoint_path,
            encoded_item_path=None,
            preloaded_encoded_items=preloaded_encoded_items,
            batch_size=batch_size,
            dtype=dtype,
            put_all_embeddings_on_device=put_all_embeddings_on_device
        )

        retrieval_file = output_dir_for_checkpoint / "retrieved_items.jsonl"

        retrieve_for_ir_dataset_queries(
            retriever=retriever,
            ir_dataset_name=ir_dataset_name,
            output_path=retrieval_file,
            top_k=top_k,
            include_content=False
        )

        print(f"Retrieval for {checkpoint_name} complete. Evaluating...")
        do_eval_and_pretty_print(
            retrieval_path=retrieval_file,
            output_dir=output_dir_for_checkpoint / "metrics",
            ir_dataset_name=ir_dataset_name
        )

if __name__ == "__main__":
    fire.Fire(evaluate_all_checkpoints)
