import logging
from omegaconf import OmegaConf
from transformers import AutoModel

from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder, HypencoderDualEncoderConfig
from hypencoder_cb.train.args import HypencoderModelConfig # Reuse the config schema

logger = logging.getLogger(__name__)

def load_hypencoder_model(
    model_name_or_path: str,
    # The following args are for building a model from scratch if path is None
    query_encoder_name: str = None,
    passage_encoder_name: str = None,
    is_shared: bool = True,
    # Add other necessary architectural details if they aren't in the config
) -> HypencoderDualEncoder:
    """
    A robust, centralized function to load a HypencoderDualEncoder.
    It correctly handles resuming from checkpoints with potentially broken
    encoder weights by reloading them from their source.
    
    Args:
        model_name_or_path: Path to the local checkpoint or Hub model name.
    """
    logger.info(f"--- Loading Hypencoder Model from: {model_name_or_path} ---")

    # 1. Load the model from the checkpoint. This loads the hyper-head correctly.
    # The encoders might be randomly initialized if the checkpoint is "broken".
    model = HypencoderDualEncoder.from_pretrained(model_name_or_path)
    config = model.config # Get the config that was loaded with the model

    # 2. Re-load the encoder backbones from their original sources to guarantee correctness.
    # The config stored with the checkpoint tells us where they came from.
    query_encoder_path = config.query_encoder_kwargs.get("model_name_or_path")
    if query_encoder_path:
        logger.info(f"Reloading/Verifying Query Encoder from: {query_encoder_path}")
        query_backbone = AutoModel.from_pretrained(query_encoder_path)
        model.query_encoder.transformer = query_backbone

    passage_encoder_path = config.passage_encoder_kwargs.get("model_name_or_path")
    if not config.shared_encoder and passage_encoder_path:
        logger.info(f"Reloading/Verifying Passage Encoder from: {passage_encoder_path}")
        passage_backbone = AutoModel.from_pretrained(passage_encoder_path)
        model.passage_encoder.transformer = passage_backbone
    elif config.shared_encoder:
        logger.info("Ensuring encoders are shared.")
        model.passage_encoder.transformer = model.query_encoder.transformer

    logger.info("Model successfully loaded and reconstructed.")
    return model