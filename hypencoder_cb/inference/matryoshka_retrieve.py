import fire
import torch
import os
import glob
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer


# Import necessary components from the existing codebase
from hypencoder_cb.inference.retrieve import HypencoderRetriever, do_eval_and_pretty_print
from hypencoder_cb.inference.shared import load_encoded_items_from_disk, retrieve_for_ir_dataset_queries, Item
from hypencoder_cb.modeling.q_net import RepeatedDenseBlockConverter
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from hypencoder_cb.utils.torch_utils import dtype_lookup

# Import the helper function we created for the Matryoshka loss
from hypencoder_cb.modeling.similarity_and_losses import _truncate_parameters

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
        ids_list: List[str],
        texts_list: List[str],
        batch_size: int,
        dtype: str,
        put_all_embeddings_on_device: bool #CHANGE
    ):
        """
        This lightweight constructor does NOT call super().__init__().
        It directly assigns the pre-loaded objects to avoid any redundant work.
        """
        # CHANGE: ADDED THE PRINT
        print(f"INFO: Initializing MatryoshkaRetriever for dimension: {matryoshka_dim} (FAST INIT)")
        self.matryoshka_dim = matryoshka_dim

        # 1. Assign pre-loaded, pre-processed objects directly.
        #    This is the core of the optimization. No re-loading or re-processing happens here.
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
        self.embeddings_on_gpu = self.encoded_item_embeddings.device.type == 'cuda'

        # 3. Move embeddings to GPU if requested.
        # CHANGE Commented the next two lines
        # if self.put_on_device:
        #     self.encoded_item_embeddings = self.encoded_item_embeddings.to(self.device)
        
        print(f"INFO: MatryoshkaRetriever initialized for dimension: {self.matryoshka_dim}")

    # This is a class specific definition of the retrieve method which
    # overrides the definition of the parent class retrieve method 
    def retrieve(self, query: "TextQuery", top_k: int) -> List["Item"]:
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

        # 2. Truncate the parameters to the target Matryoshka dimension
        dim_in = self.encoded_item_embeddings.shape[-1]
        dim_out = 1
        truncated_matrices, truncated_vectors = _truncate_parameters(
            full_matrices, full_vectors, dim_in, self.matryoshka_dim, dim_out
        )
        
        # 3. Build the temporary, smaller q-net for this dimension
        original_converter = self.model.query_encoder.weight_to_model_converter
        num_hidden_layers = original_converter.num_layers - 2
        
        if num_hidden_layers < 0:
            matryoshka_layer_dims = [dim_in, dim_out]
        else:
            matryoshka_layer_dims = [dim_in] + [self.matryoshka_dim] * (num_hidden_layers + 1) + [dim_out]

        temp_converter = RepeatedDenseBlockConverter(
            vector_dimensions=matryoshka_layer_dims,
            activation_type=original_converter.activation.__class__.__name__.lower().replace("relu", "relu"),
            do_dropout=False, # No dropout during inference
            dropout_prob=original_converter.dropout_prob,
            do_layer_norm=original_converter.do_layer_norm,
            do_residual=original_converter.do_residual,
            do_residual_on_last=original_converter.do_residual_on_last,
            layer_norm_before_residual=original_converter.layer_norm_before_residual,
        )
        q_net_at_dim = temp_converter(truncated_matrices, truncated_vectors, is_training=False)
        
        # 4. Use this smaller q-net to score the documents (batched logic from parent)
        #    This part of the logic is reused from the parent `retrieve` method.
        num_batches = (len(self.encoded_item_embeddings) // self.batch_size) + 1
        top_k_indices = torch.full((top_k * num_batches,), -1, device='cpu')
        top_k_scores = torch.full((top_k * num_batches,), -float("inf"), device='cpu')

        # CHANGE : ADDED THIS LINE
        embeddings_are_on_gpu = self.encoded_item_embeddings.device.type == 'cuda'
        
        with torch.no_grad():
            for batch_index, batch_item_embeddings in enumerate(
                torch.split(self.encoded_item_embeddings, self.batch_size)
            ):
                # CHANGE: commented this if statement
                # if self.put_on_device:
                #     # If all embeddings are already on GPU, we're good.
                #     pass
                # else:
                #     # Otherwise, move the current batch to the GPU.
                #     batch_item_embeddings = batch_item_embeddings.to(self.device)

                # CHANGE: Added the following if statement
                # If embeddings are NOT on the GPU, move the current batch.
                if not embeddings_are_on_gpu:
                    batch_item_embeddings = batch_item_embeddings.to(self.device)
                
                batch_item_embeddings = batch_item_embeddings.unsqueeze(0)
                
                # Use our on-the-fly q-net here
                similarity_matrix = q_net_at_dim(batch_item_embeddings).squeeze()

                # ----------- CHANGE COMMENTED ------------- 
                # Find topk within the batch
                values, indices = torch.topk(similarity_matrix, min(top_k, similarity_matrix.shape[0]), dim=0)
                indices = indices.squeeze(0).cpu()
                values = values.squeeze(0).cpu()

                start_idx = batch_index * top_k
                end_idx = start_idx + len(indices)
                top_k_indices[start_idx:end_idx] = indices + (batch_index * self.batch_size)
                top_k_scores[start_idx:end_idx] = values
                # ----------- CHANGE END ------------- 

                # --------- CHANGE ADDED THIS: START -------------
                # if k_for_batch > 0:
                #     values, indices = torch.topk(similarity_matrix, k_for_batch, dim=0)
                    
                #     # Move results to CPU immediately to free VRAM for the next step
                #     indices = indices.squeeze(0).cpu()
                #     values = values.squeeze(0).cpu()

                #     start_idx = batch_index * top_k
                #     end_idx = start_idx + len(indices)
                    
                #     # Store the results in the CPU tensors
                #     top_k_indices[start_idx:end_idx] = indices + (batch_index * self.batch_size)
                #     top_k_scores[start_idx:end_idx] = values
                # --------- CHANGE END -------------
        
        # Find the final top_k across all batches
        final_values, final_indices_of_indices = torch.topk(top_k_scores, top_k, dim=0)
        final_indices = top_k_indices[final_indices_of_indices]

        items = []
        for item_idx, score in zip(final_indices, final_values):
            if item_idx == -1: continue # Skip padding values
            items.append(
                Item(
                    text=self.encoded_item_texts[item_idx],
                    id=self.encoded_item_ids[item_idx],
                    score=score.item(),
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
def evaluate_matryoshka_checkpoints(
    training_run_dir: str,
    encoded_item_path: str,
    ir_dataset_name: str,
    matryoshka_dims: List[int],
    base_output_dir: str,
    original_model_name: str = "jfkback/hypencoder.6_layer",
    dtype: str = "float32",
    batch_size: int = 131072,
    top_k: int = 1000,
    put_all_embeddings_on_device: bool = True,
):
    """
    Evaluates all checkpoints from a Matryoshka training run across all
    specified dimensions, performing data processing and model loading efficiently.
    """
    # =========================================================================
    # --- ONE-TIME DATA DESERIALIZATION & PROCESSING (OUTSIDE ALL LOOPS) ---
    # =========================================================================
    print("--- STAGE 1: Deserializing and processing corpus ONCE ---")
    preloaded_items_raw = list(load_encoded_items_from_disk(encoded_item_path, target_dtype=dtype))
    
    dtype_torch = dtype_lookup(dtype)
    embeddings_cpu = torch.stack(
        [torch.tensor(x.representation, dtype=dtype_torch) for x in tqdm(preloaded_items_raw, desc="Stacking Embeddings")]
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
        print(f"INFO: Moving entire {embeddings_cpu.nbytes / 1e9:.2f} GB corpus to GPU once...")
        try:
            embeddings_for_retriever = embeddings_cpu.to(device)
            print("INFO: Corpus successfully moved to GPU.")
        except torch.cuda.OutOfMemoryError:
            print("WARNING: Could not fit the entire corpus in VRAM. Falling back to CPU-based batching.")
            embeddings_for_retriever = embeddings_cpu
    else:
        print("INFO: Keeping corpus in CPU RAM. Batches will be moved to GPU on-the-fly.")
        embeddings_for_retriever = embeddings_cpu
    # --- END OF NEW LOGIC ---
    
    # --- Automatic Checkpoint Discovery ---
    checkpoint_paths = glob.glob(os.path.join(training_run_dir, "checkpoint-*"))
    checkpoint_paths.sort(key=lambda x: int(x.split('-')[-1]))
    if os.path.exists(os.path.join(training_run_dir, "pytorch_model.bin")):
        checkpoint_paths.append(training_run_dir)
    print(f"Found {len(checkpoint_paths)} checkpoints to evaluate.")
  
    # --- OUTER LOOP: Iterates through each model checkpoint ---
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print("\n" + "#"*70)
        print(f"# LOADING CHECKPOINT & TOKENIZER {i+1}/{len(checkpoint_paths)} ONCE: {checkpoint_path}")
        
        # =========================================================================
        # --- SURGICAL MODEL & TOKENIZER LOADING (ONCE PER CHECKPOINT) ---
        # =========================================================================
        local_model = HypencoderDualEncoder.from_pretrained(checkpoint_path)
        original_model = HypencoderDualEncoder.from_pretrained(original_model_name)
        local_model.passage_encoder.transformer = original_model.passage_encoder.transformer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_model = local_model.to(device, dtype=dtype_torch)

        # --- INNER LOOP: Iterates through each Matryoshka dimension ---
        for dim in matryoshka_dims:
            print("\n" + "="*50)
            print(f"EVALUATING DIMENSION: {dim}")

            checkpoint_name = Path(checkpoint_path).name
            output_dir_for_dim = Path(base_output_dir) / Path(checkpoint_path).name / f"dim_{dim}"
            output_dir_for_dim.mkdir(parents=True, exist_ok=True)
            
            # This initialization is now extremely fast. It does no I/O.
            retriever = MatryoshkaHypencoderRetriever(
                matryoshka_dim=dim,
                model_object=local_model,
                tokenizer_object=tokenizer,
                # embeddings_tensor=preloaded_embeddings, # CHANGE: COMMENTED THIS
                embeddings_tensor=embeddings_for_retriever, # CHNAGE: ADDED THIS
                ids_list=preloaded_ids,
                texts_list=preloaded_texts,
                batch_size=batch_size,
                dtype=dtype,
                put_all_embeddings_on_device=put_all_embeddings_on_device
            )

            # Run retrieval and evaluation for this specific configuration
            retrieval_file = output_dir_for_dim / "retrieved_items.jsonl"
            retrieve_for_ir_dataset_queries(
                retriever=retriever, 
                ir_dataset_name=ir_dataset_name, 
                output_path=retrieval_file, 
                top_k=top_k
            )
            do_eval_and_pretty_print(
                retrieval_path=retrieval_file, 
                output_dir=output_dir_for_dim / "metrics", 
                ir_dataset_name=ir_dataset_name
            )
            print(f"EVALUATION FOR DIMENSION {dim} is DONE")
            print(f"SAVED THE RESULTS TO {output_dir_for_dim}")
            print(f"PROCEEDING TO NEXT DIM")

if __name__ == "__main__":
    fire.Fire(evaluate_matryoshka_checkpoints)