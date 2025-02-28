from datasets import load_dataset, Dataset
import logging
import evaluation
from tqdm import tqdm

logger = logging.getLogger(__name__)

def run_metric(evalset, metric_name, model, dump_to):
    outputs = []
    for sample in tqdm(evalset):
        output = model.predict_one(claim=sample["corrected"], 
            context=sample["context"])
        outputs.append({
            **{
                metric_name: {
                    "score": output.score,
                    "extra_outut": output.extra_output
                }
            }, **sample})
    dump_dataset = Dataset.from_list(outputs)
    dump_dataset.to_json(dump_to, force_ascii=False)

def main(eval_args, eval_metric:str):
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    MetricClass = getattr(evaluation, eval_metric)
    metric = MetricClass()
    logger.info(f"Metric: {eval_metric}")

    for evalset in eval_args.eval_datasets:
        dump_to = f'output/{eval_args.correction_model}/{evalset}/corrected.jsonl'
        logger.info(f"Loading {evalset}")
        data = load_dataset('json', data_files=dump_to, split="train")
        run_metric(data, eval_metric, metric, dump_to)
    