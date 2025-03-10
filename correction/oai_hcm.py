from .correction_model import CorrectionModel, CorrectionOutput
import time
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from utils import get_LLM_response

class OAI_HCM(CorrectionModel):
    def __init__(self, model_name, user_prompt, system_prompt=None, RPS=20):
        super().__init__(model_name)
        self.system_prompt=system_prompt
        self.user_prompt=user_prompt
        self.RPS=RPS

    def correct_one(self, claim:str, context:str):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.user_prompt.format(claim=claim, context=context)})
        completion = get_LLM_response(
            base_url = self.base_url, 
            model = self.model, 
            messages = messages,
            temperature=0.0,
            frequency_penalty=1,
            max_tokens=1000)
        time.sleep(1.0/self.RPS)
        llm_return = completion.choices[0].message.content
        return CorrectionOutput(
            claim=claim,
            context=context,
            corrected=llm_return,
            correct_model=self.model_name
        )