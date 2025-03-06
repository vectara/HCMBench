from datasets import load_dataset, Dataset
import logging
import evaluation
from tqdm import tqdm

logger = logging.getLogger(__name__)

def run_metric(evalset, model, dump_to):
    if model.use_reference:
        output_data = model.predict_dataset(evalset, "corrected", "claim")
    else:
        output_data = model.predict_dataset(evalset, "corrected", "context")
    output_data.to_json(dump_to, force_ascii=False)

def main(eval_args, eval_metric:dict):
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    metric_class_name = list(eval_metric.keys())[0]
    MetricClass = getattr(evaluation, metric_class_name)
    if eval_metric[metric_class_name] is not None:
        metric = MetricClass(**eval_metric[metric_class_name])
    else:
        metric = MetricClass()
    logger.info(f"Metric: {eval_metric}")

    for evalset in eval_args.eval_datasets:
        dump_to = f'output/{eval_args.correction_model_args["model_name"]}/{evalset}/corrected.jsonl'
        logger.info(f"Loading {evalset}")
        data = load_dataset('json', data_files=dump_to, split="train")
        run_metric(data, metric, dump_to)
    