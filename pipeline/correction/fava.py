"""
Fine-grained Hallucination Detection and Editing For Language Models
https://fine-grained-hallucination.github.io/ 
"""

import vllm
from bs4 import BeautifulSoup
from bs4.formatter import HTMLFormatter

from .correction_model import CorrectionModel, CorrectionOutput

FAVA_PROMPT = \
"""Read the following references:
{evidence}
Please identify all the errors in the following text using the information in the references provided and suggest edits if necessary:
[Text] {output}
[Edited] """

def post_process(edited_text):
    soup = BeautifulSoup(edited_text, "html.parser")
    for tag in soup.findAll(["delete", "subjective", "unverifiable", "invented", "contradictory"]):
        tag.decompose()
    clean_text = soup.get_text().strip()
    if "Edited:" in clean_text:
        clean_text = clean_text.rpartition("Edited:")[2].strip()
    return clean_text

class FAVA(CorrectionModel):
    def __init__(self, model_name="FAVA", vllm_kwargs={}, **kwargs):
        super().__init__(model_name, **kwargs)
        self.model = vllm.LLM(model="fava-uw/fava-model", **vllm_kwargs)
        self.sampling_params = vllm.SamplingParams(
            temperature=0,
            max_tokens=1024,
        )

    def process_one(self, sample: dict, debug=False) -> CorrectionOutput:
        prompts = [FAVA_PROMPT.format(evidence=sample["context"], output=sample["claim"])]
        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)
        if debug:
            print(outputs[0].outputs[0].text)
        return CorrectionOutput(
            corrected=post_process(outputs[0].outputs[0].text),
            correct_model=self.model_name
        )
