"""
Claim Extraction adapted from 
HalluMeasure: Fine-grained Hallucination Measurement Using Chain-of-Thought Reasoning
https://aclanthology.org/2024.emnlp-main.837.pdf
"""
from typing import List

from .preprocessor import Preprocessor
from ..oai_utils import get_LLM_response

SYSTEM_PROMPT = """A claim is a short sentence containing a single piece of information. You will extract claims from a given text inside <text></text> XML tags.

Here are the "Task-rules" you must follow when generating the claims.
<task-rules>
    <rule>The claim should be entirely self-contained. For instance, the claim should be comprehended without relying on other claims.</rule>
    <rule>The claim should not contain pronouns. If there are pronouns in the input text, replace them with their corresponding nouns when generating the claims.</rule>
    <rule>You will always output a list of the extracted claims.</rule>
</task-rules>
"""

EXAMPLE1_INPUT = """<text>
Samsung’s Gear Blink could have a projected keyboard that allows you to type in the air. Ralph Lauren’s Polo Tech Shirt uses biosensing fabrics to monitor physical activity. Hush earplugs filter out unwelcome sounds while allowing phone calls and alarms to intrude.
</text>"""

EXAMPLE1_OUTPUT = """- Samsung has a product called Gear Blink.
- Gear Blink could have a projected keyboard.
- Gear Blink’s projected keyboard would allow typing in air.
- Ralph Lauren has a product called Polo Tech Shirt.
- Polo Tech Shirt uses bio-sensing fabrics.
- Polo Tech Shirt bio-sensing fabrics monitor physical activity.
- There is a product called Hush earplugs.
- Hush earplugs filter out unwelcome sounds.
- Hush earplugs allow phone calls to be heard.
- Hush earplugs allow alarms to be heard."""

INPUT_TEMPLATE = """<text>
{text}
</text>"""

def parse_output(output):
    lines = output.split('\n')
    claims = []
    for line in lines:
        line = line.strip()
        if line.startswith('-'):
            claims.append(line[1:].strip())
    if len(claims) == 0:
        claims = ["."]
    return claims

class ClaimExtractor(Preprocessor):
    """ Breakdown text into atomic facts. """
    def __init__(self, model_path="anthropic/claude-3.5-sonnet",
        base_url="https://openrouter.ai/api/v1", **kwargs):
        super().__init__(**kwargs)
        self.model = model_path
        self.base_url = base_url

    def process_one(self, sample: dict, debug=False) -> List[str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE1_INPUT},
            {"role": "assistant", "content": EXAMPLE1_OUTPUT},
            {"role": "user", "content": INPUT_TEMPLATE.format(text=sample[self.input_column])}
        ]
        completion = get_LLM_response(
            base_url = self.base_url,
            model = self.model,
            messages = messages,
            temperature = 0.0,
            max_tokens = 1000)
        llm_return = completion.choices[0].message.content
        if debug:
            print(llm_return)
        return parse_output(llm_return)
