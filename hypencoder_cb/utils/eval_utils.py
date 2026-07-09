import json
from collections import defaultdict
from numbers import Number
from pathlib import Path
from typing import Optional
import os 
import ir_measures
import time
from hypencoder_cb.utils.jsonl_utils import JsonlReader

DEFAULT_METRICS = [
    "nDCG@10",
    "nDCG@5",
    "P@10",
    "P@5",
    "R@10",
    "MRR",
    "R@1000",
    "MRR@10",
]


def pretty_print_aggregated_metrics(
    aggregated_metrics_json: str,
    metric_name_ordering: Optional[list[str]] = None,
) -> str:
    with open(aggregated_metrics_json) as f:
        aggregated_metrics = json.load(f)

    if metric_name_ordering is None:
        metric_name_ordering = [
            "nDCG@10",
            "nDCG@5",
            "P@10",
            "P@5",
            "R@10",
            "RR",
            "R@1000",
            "RR@10",
        ]

    output = ""

    for metric_name in metric_name_ordering:
        if metric_name in aggregated_metrics:
            output += f"{metric_name},"
    output += "\n"

    for metric_name in metric_name_ordering:
        if metric_name in aggregated_metrics:
            output += f"{aggregated_metrics[metric_name] * 100:.2f},"
    output += "\n"

    return output


def pretty_print_aggregated_metrics_to_file(
    aggregated_metrics_json: str,
    output_file: Optional[str] = None,
    metric_name_ordering: Optional[list[str]] = None,
) -> None:
    if output_file is None:
        output_file = Path(aggregated_metrics_json).with_suffix(".txt")

    with open(output_file, "w") as f:
        f.write(
            pretty_print_aggregated_metrics(
                aggregated_metrics_json,
                metric_name_ordering=metric_name_ordering,
            )
        )


def calculate_metrics(
    run: dict[str, dict[str, Number]],
    qrels: dict[str, dict[str, Number]],
    metric_names: Optional[list[str]] = None,
) -> tuple[dict[str, Number], dict[str, dict[str, Number]]]:
    if metric_names is None:
        metric_names = DEFAULT_METRICS

    metric_objects = [ir_measures.parse_measure(metric) for metric in metric_names]
    aggregated_metrics = ir_measures.calc_aggregate(metric_objects, qrels, run)

    per_query_metrics = defaultdict(dict)
    for metric in ir_measures.iter_calc(metric_objects, qrels, run):
        per_query_metrics[metric.query_id][str(metric.measure)] = metric.value

    return aggregated_metrics, per_query_metrics


def calculate_metrics_to_file(
    run: dict[str, dict[str, Number]],
    qrels: dict[str, dict[str, Number]],
    output_folder: str,
    metric_names: Optional[list[str]] = None,
) -> None:
    aggregated_metrics, per_query_metrics = calculate_metrics(
        run, qrels, metric_names=metric_names
    )

    output_folder = Path(output_folder)
    # output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n--- DEBUG: Inside calculate_metrics_to_file ---")
    print(f"  > Current Working Directory: {os.getcwd()}")
    print(f"  > Target output_folder (absolute): {output_folder.resolve()}")
    
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"  > Successfully created or found directory: {output_folder.resolve()}")
    except Exception as e:
        print(f"  > FAILED to create directory: {e}")
        # If this fails, we know it's a permissions issue.
        # It's important to exit here if we can't write.
        raise

    aggregated_filename = output_folder / "aggregated_metrics.json"

    # --- Test write, flush, and check existence ---
    try:
        print(f"  > Attempting to write to test file: {output_folder / 'test_write.txt'}")
        with open(output_folder / "test_write.txt", "w") as f:
            f.write("This is a test write.\n")
            # Force the write buffer to be flushed to the OS
            f.flush()
            # Ask the OS to write all its caches to disk (for NFS)
            os.fsync(f.fileno()) 
        print("  > Write and fsync completed without error.")
        
        # Give the file system a moment to sync
        time.sleep(2)
        
        # Now, immediately check if the file exists from Python's perspective
        if os.path.exists(output_folder / "test_write.txt"):
            print("  > SUCCESS: Python can see the test file it just wrote.")
        else:
            print("  > CRITICAL FAILURE: Python CANNOT see the test file it just wrote. This indicates a severe file system or NFS caching issue.")

    except Exception as e:
        print(f"  > FAILED during test write: {e}")
        raise
    
    aggregated_filename = output_folder / "aggregated_metrics.json"
    per_query_filename = output_folder / "per_query_metrics.json"

    aggregated_metrics = {str(k): v for k, v in aggregated_metrics.items()}

    with open(output_folder/"test.txt", "a") as f:
        print(f"Hello I am supposed to be writing a file at the path {output_folder}")
        f.write("Now the file has more content!")
    
    with open(aggregated_filename, "w") as f:
        print("YOYO", aggregated_filename)
        json.dump(aggregated_metrics, f, sort_keys=True, indent=4)

    with open(per_query_filename, "w") as f:
        json.dump(per_query_metrics, f, sort_keys=True, indent=4)

    pretty_aggregated_filename = aggregated_filename.with_suffix(".txt")
    pretty_print_aggregated_metrics_to_file(
        aggregated_filename,
        output_file=pretty_aggregated_filename,
        metric_name_ordering=metric_names,
    )

    print("Saved aggregated metrics to", aggregated_filename)
    print("Saved pretty aggregated metrics to", pretty_aggregated_filename)
    print("Saved per query metrics to", per_query_filename)

    return aggregated_filename, per_query_filename


def load_standard_format_as_run(
    input_jsonl: str,
    score_key: str = "score",
) -> dict[str, dict[str, Number]]:
    """
    Load the standard format as a run.

    Args:
        input_jsonl (str): The input jsonl file.

    Returns:
        Dict[str, Dict[str, Number]]: The run.
    """
    with JsonlReader(input_jsonl) as reader:
        run = {}
        for line in reader:
            query_id = line["query"]["id"]
            run[query_id] = {str(item["id"]): item[score_key] for item in line["items"]}

    return run


def pretty_print_standard_format(
    standard_format_jsonl: str,
    output_file: str,
    score_key: str = "score",
) -> None:
    with JsonlReader(standard_format_jsonl) as reader:
        with open(output_file, "w") as f:
            for line in reader:
                query_id = line["query"]["id"]
                query_text = line["query"]["content"]
                f.write(f"Query: {query_text} ({query_id})\n")
                for i, item in enumerate(
                    sorted(
                        line["items"],
                        key=lambda x: x[score_key],
                        reverse=True,
                    )
                ):
                    item_id = item["id"]
                    item_text = item["content"]
                    item_score = item[score_key]
                    f.write(f"\t{i + 1}. {item_text} ({item_id}) - {item_score}\n")
                f.write("\n")
                f.write("-" * 80)
                f.write("\n")
                f.write("\n")
