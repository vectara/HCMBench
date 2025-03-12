from .oai_hcm import OAI_HCM
from .correction_model import CorrectionOutput
import re

SYS_PROMPT = "You are a helful agent who follows the given intructions properly."

RAG_PROMPT = """Here are the documents: {context}

Here is the summary: {claim}

Read the documents and then determine if the summary is accurate, if the summary contains any mistakes or information not in the document rewrite it to make it accurate. Use only information present in the document and do not rely on any other sources of information. Make minimal changes to the summary to correct the summary and provide an answer grounding the information provided in the context in English. Do not write more than neccessary when correcting the summary. Please use the format of: /##/Corrections: [put corrections] \n\n /##/Corrected summary: [put corrected summary with minimal changes]."""

CORRECTIONS_PATTERN = r"/##/\s*Corrections:\s*(.*?)\s*/##/\s*Corrected [s|S]ummary:"
CORRECTED_SUMMARY_PATTERN = r"/##/\s*Corrected [s|S]ummary:\s*(.*)"

def extract_information(text):
    """
    Extracts the corrections made by the hcm model and the corrected summary.
    """
    correction_match = re.search(CORRECTIONS_PATTERN, text, re.DOTALL)
    summary_match = re.search(CORRECTED_SUMMARY_PATTERN, text, re.DOTALL)

    # Extract corrections made and corrected summary.
    if correction_match:
        correction_text = correction_match.group(1).strip()
    else:
        correction_text = None # Set None if nothing to extract.

    if summary_match:
        summary_text = summary_match.group(1).strip()
    else:
        summary_text = None # Set None if nothing to extract.
    
    return correction_text, summary_text

class VectaraHCM(OAI_HCM):
    def __init__(self, model_name, base_url, model, user_prompt=RAG_PROMPT, system_prompt=SYS_PROMPT, num_proc=20, RPS=0):
        super().__init__(model_name, base_url, model, user_prompt, system_prompt, num_proc=num_proc, RPS=RPS)

    def process_one(self, sample:dict, debug=False) -> CorrectionOutput:
        output = super().process_one(sample, debug)
        correction_text, summary_text = extract_information(output.corrected)
        if summary_text is not None: output.corrected = summary_text
        output.extra_output = correction_text
        return output
    
def main():
    context = "Banff National Park is Canada's oldest national park, established in 1885 as Rocky Mountains Park. Located in Alberta's Rocky Mountains, 110–180 kilometres (68–112 mi) west of Calgary, Banff encompasses 6,641 square kilometres (2,564 sq mi) of mountainous terrain."
    claim = "Canada's oldest national park, Banff, was established in 1886. It recently won a Nature's Choice 2023 award for its beautiful mountainous terrain. It's the best national park ever."
    hcm = VectaraHCM(model_name="qwen_hcm", base_url="http://localhost:8000/v1", model="Qwen/Qwen2.5-7B-Instruct")
    print(hcm.process_one(sample={"context": context, "claim": claim}, debug=True))
