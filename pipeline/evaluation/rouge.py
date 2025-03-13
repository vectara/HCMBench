"""
ROUGE
https://huggingface.co/spaces/evaluate-metric/rouge
"""
import evaluate

from .evaluator import EvaluationModel, MetricOutput

class Rouge(EvaluationModel):
    """Rouge score for evaluating the similarity before and after correctiom
    """
    def __init__(self, variant='rougeL', **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        self.rouge = evaluate.load('rouge')

    def process_one(self, sample:dict) -> MetricOutput:
        """Rouge being a reference metric
           claim -> corrected_text
           context -> original_claim
        """
        claim = sample[self.claim_column].lower()
        context = sample[self.context_column].lower()
        results = self.rouge.compute(predictions=[claim],
                                references=[[context]])
        return MetricOutput(**{
            "score": results[self.variant],
            "judge_model": self.model_name
        })
