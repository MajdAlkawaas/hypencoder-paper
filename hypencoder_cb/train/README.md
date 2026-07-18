
## Overview (Original Training system created for the original Hypencoder paper with the Matryoshka training)
The main training code is in `train.py`. It uses arguments from `args.py` which are passed as a yaml config file. To customize training you can edit the yaml configuration file.

To run training call the training script with a path to a training configuration like so:
```
python hypencoder_cb/train/train.py hypencoder_cb/train/configs/hypencoder.6_layer.yaml
```
for multi-gpu training it is recommended to use either torch-launch or Huggingface's accelerate launch like this:
```
accelerate launch --num_processes={{num_gpus}} --main_process_port={{main_process_port}} hypencoder_cb/train/train.py hypencoder_cb/train/configs/hypencoder.6_layer.yaml
```


## Reproducing Paper Results
The configurations for the models in the paper are in the `configs` directory and have the name `hypencoder.[NUM_LAYERS]_layers.yaml`. Where valid `NUM_LAYERS=[2, 4, 6, 8]`. To run them follow the steps in the overview.


## Training your own Hypencoder
### Data preparation
There is a standard format used for all the training data. It is a JSONL format where each line of the file has the following keys:
```
{
  "query": {
    "id": query ID,
    "content": query text,
  },
  "items": [
    {
      "id": passage ID,
      "content": passage text,
      "score": Optional teacher score,
      "type": Sometimes used to specify type of item,
    },
  ]
}
```

Before anything else you will need to put your data in this format. Depending on what kind of training you want to do you will need to structure/provide the data as follows:
- **Teacher Distillation**: You must (1) provide the key `score` for all items (2) there needs to be a way to find the "positive" item. There are two ways to do this (a) put the positive item as the first item in the "items" list (b) give "positive" item/s a unique type such as "given" or "positive", the exact value can be anything.
- **Contrastive Loss with Hard Negatives**: You must provide (1) multiple items (2) a way to distinguish the positive and negative items. If there is only one positive item a simple way is to make it the first item in the list, otherwise you can give it a unique value under the "type" key. The exact value for "type" doesn't matter as long as all positives have the same value. You do NOT have to provide the "score" key.
- **Contrastive Loss without Hard Negatives**: The easiest way to do this is to just include only a single positive i.e. relevant item to the query in the "items" for that query. The negatives will then be the other queries positives in the batch.


#### Tokenizing Data
With your data correctly formatted it is time to tokenize it. This process tokenizes the queries and items before training to reduce the time to train and recomputation.

To tokenize your data use the following command:
```
python hypencoder_cb/utils/tokenizer_utils.py \
--standard_format_jsonl=/path/to/your/data.jsonl \
--output_file=/path/to/the/tokenized/data.jsonl \
--tokenizer="google-bert/bert-base-uncased" \
--add_special_tokens=True \
--query_max_length=32 \
--item_max_length=512
```

Now you are ready to train. If you have validation data you can use the same command as above just change the paths as needed.


### Training
To train you will need to create a new training config file. See the example.* yaml files in the `configs` directory. Some important things to know:

##### Specifying positives
This is done by specifying `positive_filter_type` in `data_config`. If the positive is the first item in items use the option `first`.
```
data_config:
    positive_filter_type: first
```
If your positive has a specific `type` value (see the data preparation section for more details) use:
```
data_config:
    type: first
    positive_filter_kwargs:
        positive_type: your-positive-type-name-here
```


#### Starting from pre-trained Hypencoder
To use a pre-trained Hypencoder you need to do two things:

1) Copy the model config from the Hypencoder you want to use as the pretrained model. For example, for the 6-layer Hypencoder the config is located in `configs/hypencoder.6_layer.yaml` and the config looks like:
```
model_config:
  tokenizer_pretrained_model_name_or_path: google-bert/bert-base-uncased
  query_encoder_kwargs:
    model_name_or_path: google-bert/bert-base-uncased
    freeze_transformer: false
    embedding_representation: null
    base_encoder_output_dim: 768
    converter_kwargs:
      vector_dimensions: [768, 768, 768, 768, 768, 768, 768, 1]
      activation_type: relu
      do_residual_on_last: false
  passage_encoder_kwargs:
    model_name_or_path: google-bert/bert-base-uncased
    freeze_transformer: false
    pooling_type: cls
  shared_encoder: true
  loss_type:
    - margin_mse
    - cross_entropy
  loss_kwargs:
    - {}
    - {"use_in_batch_negatives": true, "only_use_first_item": true}
```

2) Add the line `checkpoint_path` which should point to either the Huggingface hub model repo or a local path to a DualEncoder model. Note, it is important that the model you use has the same configuration as what you use. Otherwise things might not load properly or there might be a large performance decrease.
```
model_config:
    checkpoint_path: jfkback/hypencoder.6_layer
    ... # Additional parameters from before
```

3) Change the `loss_type` based on your requirements. Unlike the other model parameters it is fine to change the `loss_type` and `loss_kwargs` to something different than what the base model used.



#### Loss Type
Depending on the data you provided and your desired training routine you will need to specify a loss type under:
```
model_config:
  loss_type:
    - margin_mse
    - cross_entropy
  loss_kwargs:
    - {}
    - {"only_use_first_item": true}
```
If your desired model is a Matryoshka Hypencoder you will need to add the `matryoshka_dims` parameter to the `converter_kwargs`. Please refer to the provided example `train/configs/matryoshka_hypencoder_6_layers.yaml`
```
model_config:
  loss_type:
    - matryoshka_dim_margin_mse
  loss_kwargs:
    - {"matryoshka_dims": [128, 256, 512]}
```

Above you can see the two losses in this repo are and the two used to train the main Hypencoder. MarginMSE is a knowledge distillation loss which requires your data to include the "score" key with teacher scores for each query-item pair.

If you do not have scores you can only use CrossEntropy loss. CrossEntropy loss is a contrastive loss where one item is assumed to be positive for a query and all the other items in the batch are considered negatives. The option `only_use_first_item` will compute the similarity matrix as if each query only has the first item. This is useful when the "negatives" for a query may, in fact, be positives such as with knowledge distillation. Generally, if you are just doing contrastive loss set `"only_use_first_item": false`.


#### Running Training
To run training just run:
```
python hypencoder_cb/train/train.py path/to/your/config.yaml
```
or, for multi-GPU training:
```
accelerate launch --num_processes={{num_gpus}} --main_process_port={{main_process_port}} hypencoder_cb/train/train.py path/to/your/config.yaml
```