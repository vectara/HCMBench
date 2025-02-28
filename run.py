import logging
from configs import BenchmarkArguments, H4ArgumentParser
import sys
import run_correction
import run_metrics
import multiprocessing

logger = logging.getLogger(__name__)

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

    if eval_args.run_eval:
        for metric in eval_args.eval_metrics:
            process = multiprocessing.Process(target=run_metrics.main, args=(eval_args, metric,))
            process.start()
            process.join()
