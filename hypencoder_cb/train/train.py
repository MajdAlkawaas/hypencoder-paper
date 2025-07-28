import os
from typing import Optional

import fire
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Trainer, TrainingArguments

from hypencoder_cb.modeling.hypencoder import (
    HypencoderDualEncoder,
    HypencoderDualEncoderConfig,
    TextDualEncoder,
)
from hypencoder_cb.modeling.shared import BaseDualEncoderConfig
from hypencoder_cb.train.args import (
    HypencoderDataConfig,
    HypencoderModelConfig,
    HypencoderTrainerConfig,
    HypencoderTrainingConfig,
)
from hypencoder_cb.train.data_collator import GeneralDualEncoderCollator

DEFAULT_CACHE_DIR = os.environ.get(
    "HYPENCODER_CACHE_DIR", os.path.expanduser("~/.cache/hypencoder")
)


# Add this new function to train.py
import torch.nn as nn

def reinitialize_hyper_head(model: nn.Module):
    """
    Finds the Hypencoder query encoder and re-initializes its hyper-head parameters.
    """
    if hasattr(model, 'query_encoder') and isinstance(model.query_encoder, Hypencoder):
        print("Found Hypencoder query encoder. Re-initializing hyper-head.")
        query_encoder = model.query_encoder

        # Manually re-initialize all hyper-head parameters
        # Define custom layers and their initialization types
        custom_layers = {
            'hyper_base_matrices': 'normal',
            'weight_query_embeddings': 'normal',
            'hyper_base_vectors': 'zeros',
            'bias_query_embeddings': 'zeros',
            'weight_hyper_projection': 'xavier',
            'bias_hyper_projection': 'xavier',
            'key_projections': 'xavier',
            'value_projections': 'xavier'
        }

        with torch.no_grad():
            # Validate weight_shapes and bias_shapes
            weight_shapes_len = len(query_encoder.weight_shapes) if hasattr(query_encoder, 'weight_shapes') else 0
            bias_shapes_len = len(query_encoder.bias_shapes) if hasattr(query_encoder, 'bias_shapes') else 0
            if weight_shapes_len == 0 or bias_shapes_len == 0:
                raise ValueError("weight_shapes or bias_shapes not found or empty")

            # Reinitialize weight-related parameters
            for layer_name, init_type in custom_layers.items():
                if 'hyper_base_vectors' in layer_name or 'bias_' in layer_name:
                    continue  # Handle bias-related layers separately
                param_list = getattr(query_encoder, layer_name, None)
                if param_list is None:
                    print(f"Warning: {layer_name} not found in model")
                    continue
                expected_len = weight_shapes_len if layer_name in ['hyper_base_matrices', 'weight_query_embeddings'] else min(weight_shapes_len, len(param_list))
                if len(param_list) < expected_len:
                    print(f"Warning: {layer_name} has fewer parameters ({len(param_list)}) than expected ({expected_len})")
                for i in range(min(len(param_list), expected_len)):
                    param = param_list[i]
                    if isinstance(param, nn.Linear):
                        nn.init.xavier_uniform_(param.weight)
                        if param.bias is not None:
                            nn.init.zeros_(param.bias)
                    else:
                        if init_type == 'xavier':
                            nn.init.xavier_uniform_(param)
                        elif init_type == 'normal':
                            nn.init.normal_(param, mean=0.0, std=0.02)
                    print(f"Reinitialized {layer_name}[{i}] with {init_type}")

            # Reinitialize bias-related parameters
            for layer_name, init_type in custom_layers.items():
                if 'hyper_base_vectors' not in layer_name and 'bias_' not in layer_name:
                    continue  # Handle weight-related layers above
                param_list = getattr(query_encoder, layer_name, None)
                if param_list is None:
                    print(f"Warning: {layer_name} not found in model")
                    continue
                expected_len = bias_shapes_len if layer_name in ['hyper_base_vectors', 'bias_query_embeddings'] else min(bias_shapes_len, len(param_list))
                if len(param_list) < expected_len:
                    print(f"Warning: {layer_name} has fewer parameters ({len(param_list)}) than expected ({expected_len})")
                for i in range(min(len(param_list), expected_len)):
                    param = param_list[i]
                    if isinstance(param, nn.Linear):
                        nn.init.xavier_uniform_(param.weight)
                        if param.bias is not None:
                            nn.init.zeros_(param.bias)
                    else:
                        if init_type == 'xavier':
                            nn.init.xavier_uniform_(param)
                        elif init_type == 'normal':
                            nn.init.normal_(param, mean=0.0, std=0.02)
                        elif init_type == 'zeros':
                            nn.init.zeros_(param)
                    print(f"Reinitialized {layer_name}[{i}] with {init_type}")

        print("Hyper-head reinitialization complete.")

    else:
        print("Warning: Could not find Hypencoder query encoder to re-initialize.")



def load_model(model_config: HypencoderModelConfig):
    config_cls_lookup = {
        "hypencoder": HypencoderDualEncoderConfig,
        "biencoder": BaseDualEncoderConfig,
    }

    model_cls_lookup = {
        "hypencoder": HypencoderDualEncoder,
        "biencoder": TextDualEncoder,
    }

    config_cls = config_cls_lookup[model_config.model_type]
    model_cls = model_cls_lookup[model_config.model_type]

    config = config_cls(
        query_encoder_kwargs=OmegaConf.to_container(
            model_config.query_encoder_kwargs
        ),
        passage_encoder_kwargs=OmegaConf.to_container(
            model_config.passage_encoder_kwargs
        ),
        loss_type=OmegaConf.to_container(model_config.loss_type),
        loss_kwargs=OmegaConf.to_container(model_config.loss_kwargs),
        shared_encoder=model_config.shared_encoder,
    )

    if model_config.checkpoint_path is not None:
        model = model_cls.from_pretrained(
            model_config.checkpoint_path, config=config
        )
        # --- A CALL ADDED HERE ---
        # This will overwrite the loaded hyper-head weights with random ones.
        reinitialize_hyper_head(model)
        # ---------------------------------

    else:
        model = model_cls(config)

    return model


def load_data(data_config: HypencoderDataConfig):
    cache_dir = os.environ.get("HF_HOME", DEFAULT_CACHE_DIR)

    if (data_config.training_data_jsonl is None) == (
        data_config.training_huggingface_dataset is None
    ):
        raise ValueError(
            "Must specify either training_data_jsonl or"
            " training_huggingface_dataset"
        )

    if (
        data_config.validation_data_jsonl is not None
        and data_config.validation_huggingface_dataset is not None
    ):
        raise ValueError(
            "Cannot specify both validation_data_jsonl and"
            " validation_huggingface_dataset"
        )

    if data_config.training_huggingface_dataset is not None:
        training_data = load_dataset(
            data_config.training_huggingface_dataset,
            split=data_config.training_data_split,
            cache_dir=cache_dir,
        )
    elif data_config.training_data_jsonl is not None:
        training_data = load_dataset(
            "json",
            data_files=data_config.training_data_jsonl,
            split=data_config.training_data_split,
            cache_dir=cache_dir,
        )

    validation_data = None
    if data_config.validation_huggingface_dataset is not None:
        training_data = load_dataset(
            data_config.validation_huggingface_dataset,
            split=data_config.validation_data_split,
            cache_dir=cache_dir,
        )
    elif data_config.validation_data_jsonl is not None:
        training_data = load_dataset(
            "json",
            data_files=data_config.validation_data_jsonl,
            split=data_config.validation_data_split,
            cache_dir=cache_dir,
        )

    return training_data, validation_data


def get_collator(
    data_config: HypencoderDataConfig,
    trainer_config: HypencoderTrainerConfig,
    tokenizer,
):
    return GeneralDualEncoderCollator(
        tokenizer=tokenizer,
        num_negatives_to_sample=data_config.num_items_to_sample,
        positive_filter=data_config.positive_filter_type,
        positive_filter_kwargs=data_config.positive_filter_kwargs,
        positive_sampler="random",
        negative_sampler="random",
        num_positives_to_sample=data_config.num_positives_to_sample,
        label_key=data_config.label_key,
        query_padding_mode="longest",
    )


def load_tokenizer(model_config: HypencoderModelConfig):
    return AutoTokenizer.from_pretrained(
        model_config.tokenizer_pretrained_model_name_or_path
    )


def train_model(cfg: HypencoderTrainingConfig):
    print(cfg)
    resume_from_checkpoint = cfg.trainer_config.resume_from_checkpoint

    training_data, validation_data = load_data(cfg.data_config)
    tokenizer = load_tokenizer(cfg.model_config)
    model = load_model(cfg.model_config)
    collator = get_collator(cfg.data_config, cfg.trainer_config, tokenizer)

    train_arguments_kwargs = None
    hf_trainer_config = cfg.trainer_config.hf_trainer_config

    if OmegaConf.is_config(hf_trainer_config):
        train_arguments_kwargs = OmegaConf.to_container(hf_trainer_config)
    else:
        train_arguments_kwargs = hf_trainer_config.__dict__

    training_args = TrainingArguments(
        **train_arguments_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=validation_data,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("Starting training")
    if resume_from_checkpoint is True:
        if not os.path.exists(training_args.output_dir) or not any(
            [
                "checkpoint" in name
                for name in os.listdir(training_args.output_dir)
            ]
        ):
            resume_from_checkpoint = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def run_training(config_path: Optional[str] = None) -> None:
    schema = OmegaConf.structured(HypencoderTrainingConfig)

    if config_path is not None:
        config = OmegaConf.load(config_path)
        config = OmegaConf.merge(schema, config)
    else:
        config = schema

    train_model(config)


if __name__ == "__main__":
    fire.Fire(run_training)
