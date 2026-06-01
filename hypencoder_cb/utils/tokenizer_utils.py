import copy
import functools
from typing import Callable, List, Optional, Union

import fire
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from hypencoder_cb.utils.jsonl_utils import JsonlReader, JsonlWriter
from hypencoder_cb.utils.iterator_utils import batchify


def tokenizer_standard_format_file(
    standard_format_jsonl: str,
    output_file: str,
    tokenizer: Union[str, PreTrainedTokenizerBase],
    add_special_tokens: bool = True,
    query_max_length: int = 32,
    item_max_length: int = 512,
    batch_size: int = 1000,
    query_tokenizer_fn: Optional[
        Callable[[List[str]], List[List[int]]]
    ] = None,
    item_tokenizer_fn: Optional[Callable[[List[str]], List[List[int]]]] = None,
) -> None:
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def default_tokenizer_fn(texts: List[str], **kwargs):
        return tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=True,
            **kwargs,
        )["input_ids"]

    if query_tokenizer_fn is None:
        query_tokenizer_fn = functools.partial(
            default_tokenizer_fn, max_length=query_max_length
        )

    if item_tokenizer_fn is None:
        item_tokenizer_fn = functools.partial(
            default_tokenizer_fn, max_length=item_max_length
        )

    queries = {}
    items = {}

    with JsonlReader(standard_format_jsonl) as reader:
        for line in tqdm(reader):
            query = line["query"]
            line_items = line["items"]

            if "content" not in query:
                raise ValueError("Query does not have a content field.")

            if any("content" not in item for item in line_items):
                raise ValueError("Item does not have a content field.")

            if "id" not in query:
                query_id = hash(query["content"])
            else:
                query_id = query["id"]

            queries[query_id] = query["content"]

            for item in line_items:
                if "id" not in item:
                    item_id = hash(item["content"])
                else:
                    item_id = item["id"]

                items[item_id] = item["content"]

    # Tokenize queries and items
    tokenized_queries = {}
    tokenized_items = {}

    for batch in tqdm(batchify(queries.items(), batch_size)):
        batch_query_ids = [query_id for query_id, _ in batch]
        batch_query_contents = [query_content for _, query_content in batch]

        tokenized_batch_query_contents = query_tokenizer_fn(
            batch_query_contents,
        )

        for query_id, tokenized_query_content in zip(
            batch_query_ids, tokenized_batch_query_contents
        ):
            tokenized_queries[query_id] = tokenized_query_content

    for batch in tqdm(batchify(items.items(), batch_size)):
        batch_item_ids = [item_id for item_id, _ in batch]
        batch_item_contents = [item_content for _, item_content in batch]

        tokenized_batch_item_contents = item_tokenizer_fn(
            batch_item_contents,
        )

        for item_id, tokenized_item_content in zip(
            batch_item_ids, tokenized_batch_item_contents
        ):
            tokenized_items[item_id] = tokenized_item_content

    # Write tokenized standard format
    with (
        JsonlReader(standard_format_jsonl) as reader,
        JsonlWriter(output_file) as writer,
    ):
        for line in tqdm(reader):
            output_line = copy.deepcopy(line)

            query = line["query"]

            if "id" not in query:
                query_id = hash(query["content"])
            else:
                query_id = query["id"]

            tokenized_query = tokenized_queries[query_id]
            output_line["query"]["tokenized_content"] = tokenized_query

            for i, item in enumerate(output_line["items"]):
                if "id" not in item:
                    item_id = hash(item["content"])
                else:
                    item_id = item["id"]

                tokenized_item = tokenized_items[item_id]

                output_line["items"][i]["tokenized_content"] = tokenized_item

            writer.write(output_line)


if __name__ == "__main__":
    fire.Fire(tokenizer_standard_format_file)