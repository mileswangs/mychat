import json
import os
import random
import time

import pandas as pd
import yaml
from mychat.common import get_base_dir, print0
from mychat.core_eval import evaluate_task


def evaluate_model(model, tokenizer, device, max_per_task=1):
    """Evaluate a base model on the core benchmark .
    - max_per_task: crop the data to this many examples per task for testing(-1 = disable)
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]
    eval_meta_data = pd.read_csv(eval_meta_data)

    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print0(
            f"Evaluating task: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ",
            end="",
        )
        # load data for this task
        data_path = os.path.join(data_base_path, task_meta["dataset_uri"])
        with open(data_path, "r") as f:
            data = [json.loads(line.strip()) for line in f]

        # shuffle the data because in many cases it appears ordered
        # but we want the ability to only run a subset of the data for debugging purposes
        shuffle_rng = random.Random(42)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accurary = evaluate_task(model, tokenizer, device, data, task_meta)

        results[label] = accurary
        row = eval_meta_data[eval_meta_data["Eval Task"] == label]
        random_baseline = row["Random baseline"].values[0]
        centered_result = (accurary - 0.01 * random_baseline) / (1 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(f"accuracy: {accurary:.4f}% | centered : {centered_result:.4f}%, time: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {"results": results, "centered_results": centered_results, "core_metric": core_metric}
    return out
