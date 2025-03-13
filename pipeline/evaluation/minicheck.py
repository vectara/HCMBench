"""
MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents
https://github.com/Liyan06/MiniCheck
"""
from minicheck.minicheck import MiniCheck
from datasets import Dataset
import numpy as np

from .evaluator import EvaluationModel, MetricOutput

class Minicheck(EvaluationModel):
    """ Minicheck model for evaluating generated output. """
    def __init__(self, model_path="Bespoke-MiniCheck-7B", **kwargs):
        super().__init__(**kwargs)
        self.scorer = MiniCheck(model_name=model_path)

    def process_one(self, sample: dict) -> MetricOutput:
        claim = sample[self.claim_column]
        context = sample[self.context_column]
        if isinstance(claim, str):
            claims = [claim]
            docs = [context]
        else:
            claims = claim
            docs = [context] * len(claims)
        _, raw_prob, _, _ = self.scorer.score(docs=docs, claims=claims)
        return MetricOutput(**{
            "judge_model": self.model_name,
            "score": min(raw_prob),
            "extra_output": np.argmin(raw_prob),
        })
