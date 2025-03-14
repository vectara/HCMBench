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
        elif isinstance(claim, list) and isinstance(context, str):
            claims = claim
            docs = [context] * len(claims)
        else:
            claims = claim
            docs = context
        _, raw_prob, _, _ = self.scorer.score(docs=docs, claims=claims)
        return MetricOutput(**{
            "judge_model": self.model_name,
            "score": min(raw_prob),
            "extra_output": raw_prob,
        })

    def map_fn(self, sample, idx, scores, sample_boundary):
        sample_scores = scores[sample_boundary[idx]:sample_boundary[idx+1]]
        output = MetricOutput(
            score=min(sample_scores),
            extra_output=sample_scores,
            judge_model=self.model_name
        )
        return super().merge_output(sample, output)

    def process_dataset(self, data: Dataset) -> Dataset:
        claims = data[self.claim_column]
        contexts = data[self.context_column]
        sample_boundary = list(range(len(contexts))) + [len(contexts)]
        if isinstance(claims[0], list):
            new_contexts = [[contexts[idx]]*len(claim) for idx, claim in enumerate(claims)]
            sample_boundary = [0] + np.cumsum([len(claim) for claim in claims]).tolist()
            # Flatten the claim / context
            contexts = sum(new_contexts, [])
            claims = sum(claims, [])

        scores = self.process_one({
            self.claim_column: claims,
            self.context_column: contexts
        }).extra_output

        data = data.map(self.map_fn,
                        with_indices=True,
                        fn_kwargs={
                            "scores": scores,
                            "sample_boundary": sample_boundary
                        })
        return data

