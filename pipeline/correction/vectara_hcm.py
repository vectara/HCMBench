""" The file implements Vectara's HCM """
import re

from .oai_hcm import OAI_HCM
from .correction_model import CorrectionOutput

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
    def __init__(self, model_name, base_url, model,
                 user_prompt=RAG_PROMPT, system_prompt=SYS_PROMPT, **kwargs):
        super().__init__(model_name, base_url, model, user_prompt, system_prompt, **kwargs)

    def process_one(self, sample:dict, debug=False) -> CorrectionOutput:
        output = super().process_one(sample, debug)
        correction_text, summary_text = extract_information(output.corrected)
        if summary_text is not None:
            output.corrected = summary_text
        output.extra_output = correction_text
        return output
