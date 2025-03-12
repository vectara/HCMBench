from .evaluator import EvaluationModel, MetricOutput
from datasets import Dataset
import numpy as np

class Minicheck(EvaluationModel):
    """Minicheck model for evaluating generated output.
    """
    def __init__(self, model_name="Bespoke-MiniCheck-7B", **kwargs):
        super().__init__(model_name = type(self).__name__ + '#' + model_name, **kwargs)
        from minicheck.minicheck import MiniCheck
        self.scorer = MiniCheck(model_name=model_name)

    def process_one(self, sample: dict) -> MetricOutput:
        claim = sample[self.claim_column]
        context = sample[self.context_column]
        if isinstance(claim, str):
            claims = [claim]
            docs = [context]
        else:
            claims = claim
            docs = [context] * len(claims)
        pred_label, raw_prob, _, _ = self.scorer.score(docs=docs, claims=claims) 
        return MetricOutput(**{
            "judge_model": self.model_name,
            "score": min(raw_prob),
            "extra_output": np.argmin(raw_prob),
        })

def main():
    model = Minicheck(claim_column='claim')
    sample = dict(
        claim = ["The sky is blue.", "Earth's atmosphere scatters moonlight."],
        context = "The sky is blue because of the way the Earth's atmosphere scatters sunlight."
    )
    judge = model.process_one(sample)
    print(judge)

if __name__ == '__main__':
    main()