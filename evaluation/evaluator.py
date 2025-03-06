""" This file contains the Evaluator classes
"""
from pydantic import BaseModel
from typing import Optional, List, Any
from datasets import Dataset
from tqdm import tqdm

class MetricOutput(BaseModel):
    claim: str | List[str]
    context: str
    score: float
    judge_model: str
    extra_output: Optional[Any] = None

class EvaluationModel:
    """A class to evaluate generated output.
    """

    def __init__(self, model_name, claim_column = 'corrected', context_column = 'context', **kwargs):
        self.model_name = model_name
        self.claim_column = claim_column
        self.context_column = context_column
    
    def predict_dataset(self, data) -> Dataset:
        """Predict a dataset
           Default by for looping a dataset with predict_one
           Can be used to implement batch prediction for speedup
        """
        outputs = []
        for sample in tqdm(data):
            output = self.predict_one(claim=sample[self.claim_column], 
                context=sample[self.context_column])
            outputs.append({**{
                self.model_name: {
                    "score": output.score,
                    "extra_outut": output.extra_output
                }
            }, **sample})
        return Dataset.from_list(outputs)

    def predict_one(self, claim: str, context: str) -> MetricOutput:
        raise NotImplementedError
