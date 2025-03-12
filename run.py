import logging
from configs import BenchmarkArguments, H4ArgumentParser
import sys
import multiprocessing
import BenchData
import pipeline
from pipeline.correction.correction_model import CorrectionModel
from datasets import load_dataset

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
        dump_to = f'output/{eval_args.correction_model_args["model_name"]}/{evalset}/corrected.jsonl'
        if isinstance(processor, CorrectionModel):
            dataloader = getattr(BenchData, f"load_{evalset}")
            data = dataloader()
        else:
            data = load_dataset('json', data_files=dump_to, split="train")
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

    if eval_args.run_correction:
        process = multiprocessing.Process(target=run_processor, 
                                          args=(eval_args, 
                                                eval_args.correction_model, 
                                                eval_args.correction_model_args))
        process.start()
        process.join()
    
    if eval_args.run_preprocess:
        for preprocess in eval_args.preprocessors:
            processor_name = list(preprocess.keys())[0]
            process = multiprocessing.Process(target=run_processor, 
                                              args=(eval_args, 
                                                    processor_name,
                                                    preprocess[processor_name]))
            process.start()
            process.join()

    if eval_args.run_eval:
        for metric in eval_args.eval_metrics:
            processor_name = list(metric.keys())[0]
            process = multiprocessing.Process(target=run_processor, 
                                              args=(eval_args, 
                                                    processor_name,
                                                    metric[processor_name]))
            process.start()
            process.join()
    
    logger.info("Done")