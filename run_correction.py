from datasets import Dataset
import logging
from configs import BenchmarkArguments, H4ArgumentParser
import correction
import BenchData
from tqdm import tqdm

logger = logging.getLogger(__name__)

def run_hcm(evalset, model, dump_to):
    outputs = []
    for sample in tqdm(evalset):
        output = model.correct_one(claim=sample["claim"], 
            context=sample["context"])
        outputs.append({**output.model_dump(), **sample})
    dump_dataset = Dataset.from_list(outputs)
    dump_dataset.to_json(dump_to, force_ascii=False)

def main(eval_args): 
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    HCMClass = getattr(correction, eval_args.correction_model)
    if eval_args.correction_model_args is not None:
        hcm = HCMClass(**eval_args.correction_model_args)
    else:
        hcm = HCMClass()
    logger.info(f"HCM Model: {HCMClass}")

    for evalset in eval_args.eval_datasets:
        logger.info(f"Loading {evalset}")
        dataloader = getattr(BenchData, f"load_{evalset}")
        data = dataloader()
        run_hcm(data, hcm, f"output/{eval_args.correction_model_args['model_name']}/{evalset}/corrected.jsonl")
    