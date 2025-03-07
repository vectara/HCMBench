from datasets import load_dataset, Dataset
import logging
from configs import BenchmarkArguments, H4ArgumentParser
import preprocess
from tqdm import tqdm

logger = logging.getLogger(__name__)

def run_processor(evalset, model, dump_to):
    outputs = []
    for sample in tqdm(evalset):
        output = model.process_one(sample[model.input_column])
        outputs.append({
            **{model.output_column: output}, 
            **sample
        })
    dump_dataset = Dataset.from_list(outputs)
    dump_dataset.to_json(dump_to, force_ascii=False)

def main(eval_args, processor:dict): 
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    processor_class_name = list(processor.keys())[0]
    ProcessorClass = getattr(preprocess, processor_class_name)
    if eval_args.correction_model_args is not None:
        processor = ProcessorClass(**processor[processor_class_name])
    else:
        processor = ProcessorClass()
    logger.info(f"Preprocessor: {ProcessorClass}")

    for evalset in eval_args.eval_datasets:
        dump_to = f'output/{eval_args.correction_model_args["model_name"]}/{evalset}/corrected.jsonl'
        logger.info(f"Loading {evalset}")
        data = load_dataset('json', data_files=dump_to, split="train")
        run_processor(data, processor, f"output/{eval_args.correction_model_args['model_name']}/{evalset}/corrected.jsonl")
    