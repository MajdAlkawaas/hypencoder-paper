import torch


def dtype_lookup(dtype: str):

    # Make the lookup case-insensitive and handle both short and long names
    clean_dtype = dtype.lower()
    
    dtype_lookup = {
        # Short names
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        # Long names
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if clean_dtype not in dtype_lookup:
        raise KeyError(f"Unsupported dtype '{dtype}'. Supported values are: {list(dtype_lookup.keys())}")
        
    return dtype_lookup[clean_dtype]
