from .correction_model import CorrectionModel, CorrectionOutput
import vllm
from bs4 import BeautifulSoup
from bs4.formatter import HTMLFormatter

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
    def __init__(self, model_name="FAVA", vllm_kwargs={}):
        super().__init__(model_name)
        self.model = vllm.LLM(model="fava-uw/fava-model", **vllm_kwargs)
        self.sampling_params = vllm.SamplingParams(
            temperature=0,
            max_tokens=1024,
        )
        self.prompt = FAVA_PROMPT

    def process_one(self, sample: dict, debug=False) -> CorrectionOutput:
        prompts = [self.prompt.format(evidence=sample["context"], output=sample["claim"])]
        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)
        if debug:
            print(outputs[0].outputs[0].text)
        return CorrectionOutput(
            corrected=post_process(outputs[0].outputs[0].text),
            correct_model=self.model_name
        )
    
def main():
    context = "Banff National Park is Canada's oldest national park, established in 1885 as Rocky Mountains Park. Located in Alberta's Rocky Mountains, 110–180 kilometres (68–112 mi) west of Calgary, Banff encompasses 6,641 square kilometres (2,564 sq mi) of mountainous terrain."
    claim = "Canada's oldest national park, Banff, was established in 1886. It recently won a Nature's Choice 2023 award for its beautiful mountainous terrain. It's the best national park ever."
    fava = FAVA()
    print(fava.process_one({"claim": claim, "context": context}, debug=True))