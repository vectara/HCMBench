from .evaluator import EvaluationModel, MetricOutput
from typing import List
from datasets import Dataset
import torch
import numpy as np

class Minicheck(EvaluationModel):
    """Minicheck model for evaluating generated output.
    """
    def __init__(self, model_name="Bespoke-MiniCheck-7B", **kwargs):
        super().__init__(model_name = type(self).__name__ + '#' + model_name, **kwargs)
        from minicheck.minicheck import MiniCheck
        self.scorer = MiniCheck(model_name=model_name)

    def predict_one(self, claim: str | List[str], context: str) -> MetricOutput:
        if isinstance(claim, str):
            claims = [claim]
            docs = [context]
        else:
            claims = claim
            docs = [context] * len(claims)
        pred_label, raw_prob, _, _ = self.scorer.score(docs=docs, claims=claims) 
        return MetricOutput(**{
            "claim": claim,
            "context": context,
            "judge_model": self.model_name,
            "score": min(raw_prob),
            "extra_output": np.argmin(scores),
        })

def main():
    model = Minicheck()
    claim = "The sky is blue."
    context = "The sky is blue because of the way the Earth's atmosphere scatters sunlight."
    judge = model.predict_one(claim, context)
    print(judge)

if __name__ == '__main__':
    main()