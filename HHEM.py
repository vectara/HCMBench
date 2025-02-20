from Evaluator import EvaluationModel
import torch

class HHEM(EvaluationModel):
    """HHEM model for evaluating generated output.
    """
    def __init__(self, model_path="vectara/HHEM-2.1", device="cuda:0"):
        from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
        config = AutoConfig.from_pretrained('google/flan-t5-large')
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, 
            config=config)
        self.device = device
        self.model.eval()
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
        
    def predict_one(self, claim: str, context: str):
        inputs = self.tokenizer(self.prompt.format(text1=context, text2=claim), return_tensors='pt').to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        logits = output.logits
        logits = logits[:,0,:] # get the logits on the first token
        logits = torch.softmax(logits, dim=-1)
        scores = [round(x, 5) for x in logits[:, 1].tolist()] # list of float
        return scores[0]

if __name__ == '__main__':
    model = HHEM()
    claim = "The sky is blue."
    context = "The sky is blue because of the way the Earth's atmosphere scatters sunlight."
    score = model.predict_one(claim, context)
    print(score)