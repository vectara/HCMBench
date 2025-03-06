from .evaluator import EvaluationModel, MetricOutput
from typing import List
import torch
import numpy as np

class HHEM(EvaluationModel):
    """HHEM model for evaluating generated output.
    """
    def __init__(self, model_path="vectara/hallucination_evaluation_model", device="cuda:0", **kwargs):
        super().__init__(model_name = type(self).__name__ + '#' + model_path, **kwargs)
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, trust_remote_code=True)
        self.device = device
        self.model.eval()
        self.model.to(device)
        if model_path == "vectara/HHEM-2.1":
            self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        elif model_path == 'vectara/HHEM-2.2':
            self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-large")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
        
    def predict_one(self, claim: str | List[str], context: str) -> MetricOutput:
        if isinstance(claim, str):
            inputs = self.tokenizer(self.prompt.format(text1=context, text2=claim), return_tensors='pt').to(self.device)
        else:
            inputs = self.tokenizer(
                [self.prompt.format(text1=context, text2=text) for text in claim], 
                return_tensors='pt', padding='longest').to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        logits = output.logits
        logits = logits[:,0,:] # get the logits on the first token
        logits = torch.softmax(logits, dim=-1)
        scores = [round(x, 5) for x in logits[:, 1].tolist()] # list of float
        return MetricOutput(**{
            "claim": claim,
            "context": context,
            "score": min(scores),
            "extra_output": np.argmin(scores),
            "judge_model": self.model_name
        })

def main():
    model = HHEM("vectara/HHEM-2.1")
    claim = ["The sky is blue", "The Earth's atmosphere scatters moonlight."]
    context = "The sky is blue because of the way the Earth's atmosphere scatters sunlight."
    judge = model.predict_one(claim, context)
    print(judge)
    
if __name__ == '__main__':
    main()