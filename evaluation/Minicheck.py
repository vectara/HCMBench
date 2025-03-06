from .Evaluator import EvaluationModel, MetricOutput
import torch

class Minicheck(EvaluationModel):
    """Minicheck model for evaluating generated output.
    """
    def __init__(self, model_name="Bespoke-MiniCheck-7B", **kwargs):
        super().__init__(model_name = type(self).__name__ + '#' + model_name, **kwargs)
        from minicheck.minicheck import MiniCheck
        self.scorer = MiniCheck(model_name=model_name)

    def predict_one(self, claim: str, context: str) -> MetricOutput:
        pred_label, raw_prob, _, _ = self.scorer.score(docs=[context], claims=[claim]) 
        return MetricOutput(**{
            "claim": claim,
            "context": context,
            "judge_model": self.model_name,
            "score": raw_prob[0]
        })

if __name__ == '__main__':
    model = Minicheck()
    claim = "The sky is blue."
    context = "The sky is blue because of the way the Earth's atmosphere scatters sunlight."
    judge = model.predict_one(claim, context)
    print(judge)