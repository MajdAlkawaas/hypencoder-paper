import torch
from docarray import DocList
from tqdm import tqdm
import fire
# Import the schema from the original codebase
from hypencoder_cb.inference.shared import EncodedItem

def convert_to_fp16(input_path: str, output_path: str):
    """
    Loads embeddings from a file, converts them to float16, and saves to a new file.
    """
    print(f"Loading float32 embeddings from: {input_path}")
    # Use DocList.pull to stream the data efficiently
    docs_fp32 = DocList[EncodedItem].pull(f"file://{input_path}", show_progress=True)

    print(f"\nLoaded {len(docs_fp32)} documents. Converting to float16...")
    
    # Create a new DocList for the converted embeddings
    docs_fp16 = DocList[EncodedItem]()

    for doc in tqdm(docs_fp32, desc="Converting"):
        # Convert the numpy array representation to float16
        fp16_representation = doc.representation.astype('float16')
        
        # Create a new EncodedItem with the converted representation
        docs_fp16.append(
            EncodedItem(
                id=doc.id,
                text=doc.text,
                representation=fp16_representation
            )
        )

    print(f"\nConversion complete. Saving float16 embeddings to: {output_path}")
    # Use push to save the new DocList
    docs_fp16.push(f"file://{output_path}", show_progress=True)
    print("Done.")

if __name__ == "__main__":
    fire.Fire(convert_to_fp16)