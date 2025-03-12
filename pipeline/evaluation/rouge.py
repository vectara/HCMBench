from .evaluator import EvaluationModel, MetricOutput

class Rouge(EvaluationModel):
    """Rouge score for evaluating the similarity before and after correctiom
    """
    def __init__(self, variant='rougeL', **kwargs):
        super().__init__(model_name = type(self).__name__ + '#' + variant, **kwargs)
        import evaluate
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
    
def main():
    sample = {"corrected": "Hello world", "context": "Hello general"}
    rouge = Rouge()
    print(rouge.process_one(sample))