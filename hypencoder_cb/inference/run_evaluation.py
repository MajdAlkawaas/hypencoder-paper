import logging

import fire
from omegaconf import OmegaConf
from tqdm import tqdm

# Import all necessary components from your codebase
from hypencoder_cb.inference.args import RetrievalConfig
from hypencoder_cb.inference.matryoshka_retrieve import evaluate_matryoshka
from hypencoder_cb.inference.retrieve import do_retrieval

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluation_campaign(config: RetrievalConfig) -> None:
    """
    Runs the evaluation campaign.
    """
    logger.info("Starting Automated Evaluation Campaign")

    for model in config.models_to_evaluate:
        logging.info(
            f"Evaluating model: {model.name_or_path} model type: {config.model_type} "
            f"from path: {model_config.path}"
        )
        for dataset_config in config.datasets_to_evaluate:
            if config.model_type == "standard":
                do_retrieval(
                    model_name_or_path=model.name_or_path,
                    encoded_item_path=dataset_config.encoding_path,
                    output_dir=config.base_output_dir,
                    ir_dataset_name=dataset_config.ir_dataset_name,
                    query_jsonl=dataset_config.query_jsonl,
                    qrel_json=dataset_config.qrel_json,
                    query_id_key=dataset_config.query_id_key,
                    query_text_key=dataset_config.query_text_key,
                    dtype=config.dtype,
                    top_k=config.top_k,
                    batch_size=config.retrieval_batch_size,
                    query_max_length=config.query_max_length,
                    include_content=config.include_content,
                    do_eval=config.do_eval,
                    metric_names=config.metric_names,
                    ignore_same_id=config.ignore_same_id,
                )

            elif config.model_type == "matryoshka":
                if config.matryoshka_dims is not None:
                    evaluate_matryoshka(
                        model_path=model.name_or_path,
                        encoded_item_path=dataset_config.encoding_path,
                        ir_dataset_name=dataset_config.ir_dataset_name,
                        matryoshka_dims=config.matryoshka_dims,
                        base_output_dir=config.base_output_dir,
                    )
                else:
                    raise ValueError("Matryoshka dims not specified")
            else:
                raise ValueError(f"Unknown model type: {config.model_type}")

        logger.info("🎉 Evaluation Campaign Complete! 🎉")


def run_evaluation(config_path: str | None = None) -> None:
    schema = OmegaConf.structured(RetrievalConfig)

    if config_path is not None:
        config = OmegaConf.load(config_path)
        config = OmegaConf.merge(schema, config)
    else:
        config = OmegaConf.structured(schema)

    evaluation_campaign(config)


if __name__ == "__main__":
    fire.Fire(run_evaluation)
