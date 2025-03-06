from .evaluator import EvaluationModel, MetricOutput

class Rouge(EvaluationModel):
    """Rouge score for evaluating the similarity before and after correctiom
    """
    def __init__(self, variant='rougeL', **kwargs):
        super().__init__(model_name = type(self).__name__ + '#' + variant, **kwargs)
        import evaluate
        self.variant = variant
        self.rouge = evaluate.load('rouge')
        self.use_reference = True

    def predict_one(self, claim:str, context:str) -> MetricOutput:
        """Rouge being a reference metric
           claim -> corrected_text
           context -> original_claim
        """
        claim = claim.lower()
        context = context.lower()
        results = self.rouge.compute(predictions=[claim],
                                references=[[context]])
        return MetricOutput(**{
            "claim": claim,
            "context": context,
            "score": results[self.variant],
            "judge_model": self.model_name
        })