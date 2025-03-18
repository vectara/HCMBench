""" Pipeline runner """
import sys
import os
import logging
import multiprocessing

from datasets import load_dataset

import bench_data
import pipeline
from configs import BenchmarkArguments, H4ArgumentParser
from pipeline.correction.correction_model import CorrectionModel

logger = logging.getLogger(__name__)

def run_processor(eval_args, processor_name, processor_args):
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    ProcessorClass = getattr(pipeline, processor_name)
    if processor_args is not None:
        processor = ProcessorClass(**processor_args)
    else:
        processor = ProcessorClass()
    logger.info(f"Processor: {ProcessorClass}")

    for evalset in eval_args.eval_datasets:
        logger.info(f"Loading {evalset}")
        dump_to = os.path.join(eval_args.output_path, f'{evalset}/corrected.jsonl')
        if os.path.exists(dump_to):
            logger.info(f"Using existing file: {dump_to}")
            data = load_dataset('json', data_files=dump_to, split="train")
        else:
            logger.info(f"Loading data from scratch.")
            dataloader = getattr(bench_data, f"load_{evalset}")
            data = dataloader()
        dump_dataset = processor.process_dataset(data)
        dump_dataset.to_json(dump_to, force_ascii=False)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    parser = H4ArgumentParser((BenchmarkArguments, ))
    eval_args = parser.parse()
    logger.info(sys.argv)
    logger.info(eval_args)

    for processor in eval_args.pipeline:
        processor_name = list(processor.keys())[0]
        process = multiprocessing.Process(target=run_processor,
                                          args=(eval_args,
                                                processor_name,
                                                processor[processor_name],))
        process.start()
        process.join()

    logger.info("Done")
