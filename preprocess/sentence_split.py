# Sentence Decontextualization from MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents
# https://aclanthology.org/2024.emnlp-main.499.pdf
from .preprocessor import Preprocessor
from typing import List
import json
import spacy
import time
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from utils import get_LLM_response

SYSTEM_PROMPT = """You are provided with a context and a claim. Please first determine if the claim can stand alone without the context. If not, provide a decontextualized version of the claim that incorporates necessary information from the context to make it self-contained. The revision should be as minimum as possible. You will always change double quotes to single quotes in the claim. For example, write 'glass' instead of "glass". Please respond with a JSON format: {"label": "yes"/"no", "decontext": "NA"/decontextualized claim}."""

EXAMPLE1_INPUT = """Context: There are many reasons why poetry is important for children. Poetry can help children build confidence through memorizing and reciting poems. It can also provide an easy way for children to remember a lesson or value.

Claim: It can also provide an easy way for children to remember a lesson or value."""

EXAMPLE1_OUTPUT = """{"label": "no", "decontext": "Poetry can provide an easy way for children to remember a lesson or value."}"""

EXAMPLE2_INPUT = """Context: Yes, ancient societies had concepts of rights. The concept of rights first appeared in the theory of natural law which existed in the state of nature. In this state, people enjoyed certain rights sanctioned by natural law.

Claim: In this state, people enjoyed certain rights sanctioned by natural law."""

EXAMPLE2_OUTPUT = """{"label": "no", "decontext": "In the state of nature, people enjoyed certain rights sanctioned by natural law."}"""

EXAMPLE3_INPUT = """Context: The ancient Greeks had some concept of human rights, although there is no single word in classical Greek that captures the sense of "rights" as it is used in modern political thought. However, Greek customs and institutions provided protection to
private property unique in the ancient world, instilling a strong sense of equality. The idea of human rights spread quickly from Babylon to Greece and eventually Rome, where the concept of "natural law" arose.

Claim: The idea of human rights spread quickly from Babylon to Greece and eventually Rome, where the concept of "natural law" arose."""

EXAMPLE3_OUTPUT = """{"label": "yes", "decontext": "NA"}"""

INPUT_TEMPLATE = """Context: {context}

Claim: {claim}"""

class Sentencizer(Preprocessor):
    """Breakdown text into individual sentences.
    """
    def __init__(self, 
            model_path="anthropic/claude-3.5-sonnet", 
            base_url="https://openrouter.ai/api/v1",
            RPS=20, 
            decontext=False,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.model = model_path
        self.base_url = base_url
        self.RPS = RPS
        self.nlp = spacy.load('en_core_web_sm')
        self.decontext = decontext

    def process_one(self, text:str) -> List[str]:
        doc = self.nlp(text)
        sents = [sent.text.strip() for sent in doc.sents]
        if len(sents) == 0: sents = ['.']
        if self.decontext:
            decontext_sents = [sents[0]]
            for idx, sent in enumerate(sents[1:], start=1):
                decontext_sents.append(
                    self.decontextualize(
                        context = " ".join(sents[:idx+1]),
                        claim = sent
                    )
                )
            sents = decontext_sents
        return sents
    
    def decontextualize(self, context: str, claim: str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE1_INPUT},
            {"role": "assistant", "content": EXAMPLE1_OUTPUT},
            {"role": "user", "content": EXAMPLE2_INPUT},
            {"role": "assistant", "content": EXAMPLE2_OUTPUT},
            {"role": "user", "content": EXAMPLE3_INPUT},
            {"role": "assistant", "content": EXAMPLE3_OUTPUT},
            {"role": "user", "content": INPUT_TEMPLATE.format(context=context, claim=claim)}
        ]
        completion = get_LLM_response(
            base_url = self.base_url, 
            model = self.model, 
            messages = messages,
            temperature = 0.0,
            max_tokens = 1000)
        time.sleep(1.0/self.RPS)
        llm_return = completion.choices[0].message.content
        print(llm_return)
        try:
            resp = json.loads(llm_return)
            if resp["label"] == "no":
                return resp["decontext"]
        except:
            pass
        return claim

if __name__ == '__main__':
    sentencizer = Sentencizer(decontext=True)
    text = """A wild card, unknown when the strike began, was Taylor Swift’s concert film. It certainly filled an October void that existed before the strike and its SAG-AFTRA waver meant she could promote it."""

    print(sentencizer.process_one(text))