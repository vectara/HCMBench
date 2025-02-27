""" This file contains the Evaluator classes
"""
from pydantic import BaseModel
from typing import Optional

class MetricOutput(BaseModel):
    claim: str
    context: str
    score: float
    judge_model: str
    extra_output: Optional[str] = None

class EvaluationModel:
    """A class to evaluate generated output.
    """

    def __init__(self, model_name):
        self.model_name = model_name
    
    def predict_one(self, claim: str, context: str) -> MetricOutput:
        raise NotImplementedError

