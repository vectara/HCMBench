""" This file contains the Evaluator classes """
from typing import Optional, Any, Dict

from ..processor import Processor, ProcessorOutput

class MetricOutput(ProcessorOutput):
    score: float
    judge_model: str
    extra_output: Optional[Any] = None

class EvaluationModel(Processor):
    """ A class to evaluate generated output. """
    def __init__(self, model_name,
                 claim_column='corrected', context_column='context', **kwargs):
        """
        Args:
            model_name: the metric name recorded in the output data
            claim_column: the corrected claim (hypothesis) column in the dataset
            context_column: the context (reference) column in the dataset
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.claim_column = claim_column
        self.context_column = context_column

    def merge_output(self, sample: Dict, output: MetricOutput) -> Dict:
        return {
            **sample,
            output.judge_model: {
                "score": output.score,
                "extra_outut": output.extra_output
            }
        }

    def process_one(self, sample: Dict) -> MetricOutput:
        raise NotImplementedError
