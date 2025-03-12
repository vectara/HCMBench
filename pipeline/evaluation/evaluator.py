""" This file contains the Evaluator classes
"""
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from datasets import Dataset
from tqdm import tqdm
from ..processor import Processor, ProcessorOutput

class MetricOutput(ProcessorOutput):
    score: float
    judge_model: str
    extra_output: Optional[Any] = None

class EvaluationModel(Processor):
    """A class to evaluate generated output.
    """
    def __init__(self, model_name, revision='', claim_column='corrected', context_column='context', **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name + revision
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
