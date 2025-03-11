from .correction_model import CorrectionModel, CorrectionOutput
import time
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from utils import get_LLM_response

class OAI_HCM(CorrectionModel):
    def __init__(self, model_name, base_url, model, user_prompt, system_prompt=None, RPS=20):
        super().__init__(model_name)
        self.system_prompt=system_prompt
        self.user_prompt=user_prompt
        self.RPS=RPS
        self.base_url = base_url
        self.model = model

    def correct_one(self, claim:str, context:str, debug=False):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.user_prompt.format(claim=claim, context=context)})
        completion = get_LLM_response(
            base_url = self.base_url, 
            model = self.model, 
            messages = messages,
            temperature=0.0,
            max_tokens=1000)
        time.sleep(1.0/self.RPS)
        llm_return = completion.choices[0].message.content
        if debug: print(llm_return)
        return CorrectionOutput(
            claim=claim,
            context=context,
            corrected=llm_return,
            correct_model=self.model_name
        )