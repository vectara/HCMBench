from .correction_model import CorrectionModel, CorrectionOutput
import vllm
from bs4 import BeautifulSoup
from bs4.formatter import HTMLFormatter

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
            top_p=1.0,
            max_tokens=1024,
        )
        self.prompt = "Read the following references:\n{evidence}\nPlease identify all the errors in the following text using the information in the references provided and suggest edits if necessary:\n[Text] {output}\n[Edited] "

    def correct_one(self, claim:str, context:str, debug=False):
        prompts = [self.prompt.format(evidence=context, output=claim)]
        outputs = self.model.generate(prompts, self.sampling_params)
        if debug:
            print(outputs[0].outputs[0].text)
        return CorrectionOutput(
            claim=claim,
            context=context,
            corrected=post_process(outputs[0].outputs[0].text),
            correct_model=self.model_name
        )
    
def main():
    context = "Banff National Park is Canada's oldest national park, established in 1885 as Rocky Mountains Park. Located in Alberta's Rocky Mountains, 110–180 kilometres (68–112 mi) west of Calgary, Banff encompasses 6,641 square kilometres (2,564 sq mi) of mountainous terrain."
    claim = "Canada's oldest national park, Banff, was established in 1886. It recently won a Nature's Choice 2023 award for its beautiful mountainous terrain. It's the best national park ever."
    fava = FAVA()
    print(fava.correct_one(claim, context, debug=True))