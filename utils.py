""" Utilities functions """

import json
import os

from datasets import load_dataset

FACTUALITY_METRICS = ["AXCEL", "HHEM", "Minicheck", "FACTSGJudge"]
SIMILARITY_METRICS = ["Rouge"]
ALL_METRICS = FACTUALITY_METRICS + SIMILARITY_METRICS

def load_jsonl(input_path):
    with open(input_path, "r", encoding="UTF-8") as f:
        data = [json.loads(line) for line in f]
    return data

def dump2jsonl(lines, output_path):
    with open(output_path, "w", encoding="UTF-8") as f:
        for line in lines:
            f.write(json.dumps(line) + '\n')

def postprocess_metrics(sample, metrics):
    binary_scores = [sample[metric]["score"] > 0.5 for metric in metrics]
    sample["min_score"] = min([sample[metric]["score"] for metric in metrics])
    sample["max_score"] = max([sample[metric]["score"] for metric in metrics])
    sample["avg_score"] = sum([sample[metric]["score"] for metric in metrics]) / len(metrics)
    sample["vote_score"] = int(sum(binary_scores) > (len(metrics) / 2))
    return sample

def aggregate_score(output_dir="output/", filter_fn=None, metric_list=None, vote_metrics=None):
    # Metrics aggregation
    models = os.listdir(output_dir)
    for model in models:
        eval_datasets = os.listdir(os.path.join(output_dir, model))
        for evalset in eval_datasets:
            dump_folder = os.path.join(output_dir, model, evalset)
            data = load_dataset('json', data_files=os.path.join(dump_folder, 'corrected.jsonl'),
                                split="train")
            if filter_fn is not None:
                data = data.filter(filter_fn)
            if metric_list is None:
                metric_list = [column for column in data.column_names \
                            if column.partition('#')[0] in ALL_METRICS]
            if vote_metrics is None:
                vote_metrics = [column for column in data.column_names \
                                if column.partition('#')[0] in FACTUALITY_METRICS]
            data = data.map(postprocess_metrics, fn_kwargs={"metrics": vote_metrics})
            data.to_json(os.path.join(dump_folder, 'voted.jsonl'), force_ascii=False)

            agg_metrics = {}
            for metric in metric_list:
                metric_scores = [sample[metric]['score'] for sample in data]
                agg_metrics[metric] = sum(metric_scores) / len(metric_scores)

            agg_metrics["min_score"] = sum(data["min_score"]) / len(data)
            agg_metrics["max_score"] = sum(data["max_score"]) / len(data)
            agg_metrics["avg_score"] = sum(data["avg_score"]) / len(data)
            agg_metrics["vote_score"] = sum(data["vote_score"]) / len(data)

            dump2jsonl([agg_metrics], os.path.join(dump_folder, 'score_summary.jsonl'))
        