import logging
from configs import BenchmarkArguments, H4ArgumentParser
import sys
import run_correction
import run_metrics
import run_preprocess
import multiprocessing
from datasets import load_dataset
import utils
import evaluation

logger = logging.getLogger(__name__)

def postprocess_metrics(sample, metrics):
    binary_scores = [sample[metric]["score"] > 0.5 for metric in metrics]
    sample["min_score"] = min([sample[metric]["score"] for metric in metrics])
    sample["max_score"] = max([sample[metric]["score"] for metric in metrics])
    sample["avg_score"] = sum([sample[metric]["score"] for metric in metrics]) / len(metrics)
    sample["vote_score"] = int(sum(binary_scores) > (len(metrics) / 2))
    return sample

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    parser = H4ArgumentParser((BenchmarkArguments, ))
    eval_args = parser.parse()
    logger.info(sys.argv)
    logger.info(eval_args)

    if eval_args.run_correction:
        process = multiprocessing.Process(target=run_correction.main, args=(eval_args,))
        process.start()
        process.join()
    
    if eval_args.run_preprocess is not None:
        for preprocess in eval_args.preprocessors:
            process = multiprocessing.Process(target=run_preprocess.main, args=(eval_args, preprocess,))
            process.start()
            process.join()

    if eval_args.run_eval:
        for metric in eval_args.eval_metrics:
            process = multiprocessing.Process(target=run_metrics.main, args=(eval_args, metric,))
            process.start()
            process.join()

        # Metrics aggregation
        for evalset in eval_args.eval_datasets:
            dump_to = f'output/{eval_args.correction_model_args["model_name"]}/{evalset}/corrected.jsonl'
            logger.info(f"Loading {evalset}")
            data = load_dataset('json', data_files=dump_to, split="train")

            metric_list = [column for column in data.column_names if column.partition('#')[0] in evaluation.__all__]
            vote_metrics = [column for column in data.column_names if column.partition('#')[0] in evaluation.__factuality__]
            logger.info(f"Metric list: {metric_list}")
            data = data.map(postprocess_metrics, fn_kwargs={"metrics": vote_metrics})
            data.to_json(dump_to, force_ascii=False)

            agg_metrics = {}
            for metric in metric_list:
                metric_scores = [sample[metric]['score'] for sample in data]
                agg_metrics[metric] = sum(metric_scores) / len(metric_scores)

            agg_metrics["min_score"] = sum(data["min_score"]) / len(data)
            agg_metrics["max_score"] = sum(data["max_score"]) / len(data)
            agg_metrics["avg_score"] = sum(data["avg_score"]) / len(data)
            agg_metrics["vote_score"] = sum(data["vote_score"]) / len(data)

            utils.dump2jsonl([agg_metrics], f'output/{eval_args.correction_model_args["model_name"]}/{evalset}/score_summary.jsonl')
            logger.info(f"Aggregated scores for {evalset}:")
            logger.info(agg_metrics)
    
    logger.info("Done")