""" https://huggingface.co/vectara/hallucination_evaluation_model """
import torch
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

from .evaluator import EvaluationModel, MetricOutput

HHEM_PROMPT = \
"""<pad> Determine if the hypothesis is true given the premise?

Premise: {text1}

Hypothesis: {text2}"""

class HHEM(EvaluationModel):
    """ HHEM model for evaluating generated output. """
    def __init__(self, model_path="vectara/hallucination_evaluation_model",
                 device="cuda:0", batch_size=1, **kwargs):
        super().__init__(**kwargs)

        self.model = AutoModelForTokenClassification.from_pretrained(model_path,
                                                                     trust_remote_code=True)
        self.device = device
        self.batch_size = batch_size
        self.model.eval()
        self.model.to(device)
        if model_path == "vectara/HHEM-2.1":
            self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        elif model_path == 'vectara/HHEM-2.2':
            self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-large")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    def process_one(self, sample:dict) -> MetricOutput:
        claim = sample[self.claim_column]
        context = sample[self.context_column]
        if isinstance(claim, str):
            inputs = self.tokenizer(HHEM_PROMPT.format(text1=context,
                                                       text2=claim),
                                                       return_tensors='pt').to(self.device)
        elif isinstance(claim, list) and isinstance(context, str):
            inputs = self.tokenizer(
                [HHEM_PROMPT.format(text1=context, text2=text) for text in claim],
                return_tensors='pt', padding='longest').to(self.device)
        else:
            inputs = self.tokenizer(
                [HHEM_PROMPT.format(text1=text1,
                                    text2=text2) for text1, text2 in zip(context, claim)],
                return_tensors='pt', padding='longest').to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        logits = output.logits
        logits = logits[:,0,:] # get the logits on the first token
        logits = torch.softmax(logits, dim=-1)
        scores = [round(x, 5) for x in logits[:, 1].tolist()] # list of float
        return MetricOutput(**{
            "score": min(scores),
            "extra_output": scores,
            "judge_model": self.model_name
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

        batch_idx = list(range(0, len(claims), self.batch_size)) + [len(claims)]
        batch_st, batch_ed = batch_idx[:-1], batch_idx[1:]
        scores = []
        for st, ed in tqdm(zip(batch_st, batch_ed), total=len(batch_idx)-1):
            sample = {
                self.claim_column: claims[st:ed],
                self.context_column: contexts[st:ed]
            }
            output = self.process_one(sample)
            scores.extend(output.extra_output)
        data = data.map(self.map_fn,
                        with_indices=True,
                        fn_kwargs={
                            "scores": scores,
                            "sample_boundary": sample_boundary
                        })
        return data
