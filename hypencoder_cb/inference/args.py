import os
from dataclasses import dataclass, field

import fire
from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    name_or_path: str


@dataclass
class DatasetConfig:
    # TODO: handle non ir_dataset_name datasets
    ir_dataset_name: str  # The name in the ir_datasets library
    encoding_path: str  # Path to the pre-computed document embeddings
    query_jsonl: str  # Path to the queries
    qrel_json: str  # Path to the qrels
    query_id_key: str
    query_text_key: str


@dataclass
class RetrievalConfig:
    # --- Experiment Setup ---
    model_type: str  # "standard" or "matryoshka"
    models_to_evaluate: list[ModelConfig] = field(default_factory=list)
    datasets_to_evaluate: list[DatasetConfig] = field(default_factory=list)
    base_output_dir: str = "./evaluation_results"

    # --- Matryoshka-Specific Settings ---
    matryoshka_dims: list[int] | None = None

    # --- Hardware and Performance ---
    dtype: str = "fp16"
    put_all_embeddings_on_device: bool = True
    retrieval_batch_size: int = 131072

    # --- Evaluation Settings ---
    top_k: int = 1000
    do_eval: bool = True
    include_content: bool = True
    metric_names: list[str] | None = None
    ignore_same_id: bool = True
    query_max_length: int = 64
    # TODO: Retrieval kwargs


def relative_file_path_to_abs_path(path: str) -> str:
    return os.path.join(os.path.dirname(__file__), path)


def export_config_to_yaml(
    config_name: str | None = None,
    config_dir: str = "configs",
) -> None:
    config_dir = relative_file_path_to_abs_path(config_dir)
    config = OmegaConf.structured(RetrievalConfig)

    if config_name is None:
        config_name = config

    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

    config_path = os.path.join(config_dir, config_name)

    if os.path.isfile(config_path):
        raise ValueError(
            f"The config file {config_path} already exists. Please choose a"
            " different config name."
        )

    print(f"Exporting config to {config_path}")
    OmegaConf.save(config=config, f=config_path)


if __name__ == "__main__":
    fire.Fire(export_config_to_yaml)
