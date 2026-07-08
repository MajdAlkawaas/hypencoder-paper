import glob
import os
from pathlib import Path

import fire
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Import necessary components from the existing codebase
from hypencoder_cb.inference.retrieve import (
    HypencoderRetriever,
    do_eval_and_pretty_print,
)
from hypencoder_cb.inference.shared import (
    Item,
    TextQuery,
    load_encoded_items_from_disk,
    retrieve_for_ir_dataset_queries,
)
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from hypencoder_cb.modeling.q_net import MatryoshkaQNetFactory

# Import the helper function we created for the Matryoshka loss
from hypencoder_cb.utils.torch_utils import dtype_lookup


class MatryoshkaHypencoderRetriever(HypencoderRetriever):
    """
    A specialized retriever that generates a q-net of a specific, truncated
    Matryoshka dimension for scoring.
    """

    def __init__(
        self,
        matryoshka_dim: int,
        model_object: HypencoderDualEncoder,
        tokenizer_object: AutoTokenizer,
        embeddings_tensor: torch.Tensor,
        ids_list: list[str],
        texts_list: list[str],
        batch_size: int,
        dtype: str,
        put_all_embeddings_on_device: bool,  # CHANGE
    ):
        """
        This lightweight constructor does NOT call super().__init__().
        It directly assigns the pre-loaded objects to avoid any redundant work.
        """
        # CHANGE: ADDED THE PRINT
        print(
            f"INFO: Initializing MatryoshkaRetriever for dimension: {matryoshka_dim}"
            " (FAST INIT)"
        )
        self.matryoshka_dim = matryoshka_dim

        # 1. Assign pre-loaded, pre-processed objects directly.
        #    This is the core of the optimization. No re-loading or re-processing
        #    happens here.
        self.model = model_object
        self.tokenizer = tokenizer_object
        self.encoded_item_embeddings = embeddings_tensor
        self.encoded_item_ids = ids_list
        self.encoded_item_texts = texts_list

        # 2. Set up the rest of the configuration from arguments.
        if isinstance(dtype, str):
            self.dtype = dtype_lookup(dtype)
        else:
            self.dtype = dtype

        self.device = self.model.device
        self.batch_size = batch_size
        self.put_on_device = put_all_embeddings_on_device
        self.query_max_length = 32

        # CHANGE: ADDED THIS LINE
        self.embeddings_on_gpu = self.encoded_item_embeddings.device.type == "cuda"

        self.matryoshka_qnet_factory = MatryoshkaQNetFactory(
            original_qnet_converter=self.model.query_encoder.weight_to_model_converter
        )

        print(
            "INFO: MatryoshkaRetriever initialized for dimension: "
            f"{self.matryoshka_dim}"
        )

    # This is a class specific definition of the retrieve method which
    # overrides the definition of the parent class retrieve method
    def retrieve(self, query: "TextQuery", top_k: int) -> list["Item"]:
        """
        Overrides the parent retrieve method to use a truncated q-net.
        """
        tokenized_query = self.tokenizer(
            query.text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.query_max_length,
        ).to(self.device)

        with torch.no_grad():
            # 1. Run the query encoder to get the FULL-SIZE generated parameters
            query_output = self.model.query_encoder(
                input_ids=tokenized_query["input_ids"],
                attention_mask=tokenized_query["attention_mask"],
            )
            full_matrices = query_output.generated_matrices
            full_vectors = query_output.generated_vectors

        matryoshka_qnets = self.matryoshka_qnet_factory.build(
            full_matrices,
            full_vectors,
            [self.matryoshka_dim],
            is_training=False,
        )
        # Get the Q-Net out of the dictionary.
        q_net_at_dim = matryoshka_qnets[self.matryoshka_dim]
        # 4. Use this smaller q-net to score the documents (batched logic from parent)
        #    This part of the logic is reused from the parent `retrieve` method.
        num_batches = (len(self.encoded_item_embeddings) // self.batch_size) + 1
        top_k_indices = torch.full((top_k * num_batches,), -1, device="cpu")
        top_k_scores = torch.full((top_k * num_batches,), -float("inf"), device="cpu")

        # CHANGE : ADDED THIS LINE
        embeddings_are_on_gpu = self.encoded_item_embeddings.device.type == "cuda"

        with torch.no_grad():
            for batch_index, batch_item_embeddings in enumerate(
                torch.split(self.encoded_item_embeddings, self.batch_size)
            ):
                if not embeddings_are_on_gpu:
                    batch_item_embeddings = batch_item_embeddings.to(self.device)

                batch_item_embeddings = batch_item_embeddings.unsqueeze(0)

                # Use our on-the-fly q-net here
                similarity_matrix = q_net_at_dim(batch_item_embeddings).squeeze()

                values, indices = torch.topk(
                    similarity_matrix, min(top_k, similarity_matrix.shape[0]), dim=0
                )
                indices = indices.squeeze(0).cpu()
                values = values.squeeze(0).cpu()

                start_idx = batch_index * top_k
                end_idx = start_idx + len(indices)
                top_k_indices[start_idx:end_idx] = indices + (
                    batch_index * self.batch_size
                )
                top_k_scores[start_idx:end_idx] = values

        # Find the final top_k across all batches
        final_values, final_indices_of_indices = torch.topk(top_k_scores, top_k, dim=0)
        final_indices = top_k_indices[final_indices_of_indices]

        items = []
        for item_idx, score in zip(final_indices.tolist(), final_values.tolist()):
            if item_idx == -1:
                continue  # Skip padding values
            items.append(
                Item(
                    text=self.encoded_item_texts[item_idx],
                    id=self.encoded_item_ids[item_idx],
                    score=score,
                    type=f"matryoshka_retriever_dim_{self.matryoshka_dim}",
                )
            )
        return items


# ... (We will add the main script logic below)


# Step 2: The Main Evaluation Orchestration Script
# Now, let's add the main function to this same file. This will be
# your entry point. It will handle the one-time data loading, the
# surgical model loading, and the nested loops.


# =================================================================================
# --- Main Orchestration Script ---
# =================================================================================
def evaluate_matryoshka(
    model_path: str,
    encoded_item_path: str,
    ir_dataset_name: str,
    matryoshka_dims: list[int],
    base_output_dir: str,
    original_model_name: str = "jfkback/hypencoder.6_layer",
    dtype: str = "float32",
    batch_size: int = 131072,
    top_k: int = 1000,
    put_all_embeddings_on_device: bool = True,
):
    """
    Evaluates a Matryoshka Hypencoder model across all specified dimensions, 
    performing data processing and model loading efficiently.
    """
    # =========================================================================
    # --- ONE-TIME DATA DESERIALIZATION & PROCESSING (OUTSIDE ALL LOOPS) ---
    # =========================================================================
    print("--- STAGE 1: Deserializing and processing corpus ONCE ---")
    preloaded_items_raw = list(
        load_encoded_items_from_disk(encoded_item_path, target_dtype=dtype)
    )

    dtype_torch = dtype_lookup(dtype)
    embeddings_cpu = torch.stack(
        [
            torch.tensor(x.representation, dtype=dtype_torch)
            for x in tqdm(preloaded_items_raw, desc="Stacking Embeddings")
        ]
    )
    preloaded_ids = [x.id for x in preloaded_items_raw]
    preloaded_texts = [x.text for x in preloaded_items_raw]
    del preloaded_items_raw
    print("One-time data processing complete.")

    # CHANGE: ADDED THE BELLOW BLOCK
    # --- NEW LOGIC: EXPLICITLY MANAGE DATA LOCATION ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings_for_retriever = None
    if put_all_embeddings_on_device:
        print(
            f"INFO: Moving {embeddings_cpu.nbytes / 1e9:.2f} GB corpus to GPU once..."
        )
        try:
            embeddings_for_retriever = embeddings_cpu.to(device)
            print("INFO: Corpus successfully moved to GPU.")
        except torch.cuda.OutOfMemoryError:
            print(
                "WARNING: Could not fit the entire corpus in VRAM. "
                "Falling back to CPU-based batching."
            )
            embeddings_for_retriever = embeddings_cpu
    else:
        print(
            "INFO: Keeping corpus in CPU RAM. Batches will be moved to GPU on-the-fly."
        )
        embeddings_for_retriever = embeddings_cpu
    # --- END OF NEW LOGIC ---

    # =========================================================================
    # --- SURGICAL MODEL & TOKENIZER LOADING (ONCE PER CHECKPOINT) ---
    # =========================================================================
    local_model = HypencoderDualEncoder.from_pretrained(model_path)
    original_model = HypencoderDualEncoder.from_pretrained(original_model_name)
    local_model.passage_encoder.transformer = original_model.passage_encoder.transformer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    local_model = local_model.to(device, dtype=dtype_torch)

    # --- INNER LOOP: Iterates through each Matryoshka dimension ---
    for dim in matryoshka_dims:
        print("\n" + "=" * 50)
        print(f"EVALUATING DIMENSION: {dim}")

        output_dir_for_dim = (
            Path(base_output_dir)
            / Path(ir_dataset_name)
            / Path(model_path).name
            / f"dim_{dim}"
        )
        output_dir_for_dim.mkdir(parents=True, exist_ok=True)
        time_dir = output_dir_for_dim / "time"
        time_dir.mkdir(parents=True, exist_ok=True)
        # This initialization is now extremely fast. It does no I/O.
        retriever = MatryoshkaHypencoderRetriever(
            matryoshka_dim=dim,
            model_object=local_model,
            tokenizer_object=tokenizer,
            # embeddings_tensor=preloaded_embeddings, # CHANGE: COMMENTED THIS
            embeddings_tensor=embeddings_for_retriever,  # CHNAGE: ADDED THIS
            ids_list=preloaded_ids,
            texts_list=preloaded_texts,
            batch_size=batch_size,
            dtype=dtype,
            put_all_embeddings_on_device=put_all_embeddings_on_device,
        )

        # Run retrieval and evaluation for this specific configuration
        retrieval_file = output_dir_for_dim / "retrieved_items.jsonl"
        retrieve_for_ir_dataset_queries(
            retriever=retriever,
            ir_dataset_name=ir_dataset_name,
            output_path=retrieval_file,
            top_k=top_k,
            track_time=True,
            track_time_file=time_dir / "retrieval_time.json",
        )
        do_eval_and_pretty_print(
            retrieval_path=retrieval_file,
            output_dir=output_dir_for_dim / "metrics",
            ir_dataset_name=ir_dataset_name,
        )
        print(f"EVALUATION FOR DIMENSION {dim} is DONE")
        print(f"SAVED THE RESULTS TO {output_dir_for_dim}")
        print("PROCEEDING TO NEXT DIM")


if __name__ == "__main__":
    fire.Fire(evaluate_matryoshka)


#
