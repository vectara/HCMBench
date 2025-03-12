from .correction_model import CorrectionModel, CorrectionOutput
import time
from ..oai_utils import get_LLM_response

class OAI_HCM(CorrectionModel):
    def __init__(self, model_name, base_url, model, user_prompt, system_prompt=None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.system_prompt=system_prompt
        self.user_prompt=user_prompt
        self.base_url = base_url
        self.model = model

    def process_one(self, sample:dict, debug=False) -> CorrectionOutput:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.user_prompt.format(claim=sample["claim"], context=sample["context"])})
        completion = get_LLM_response(
            base_url = self.base_url, 
            model = self.model, 
            messages = messages,
            temperature=0.0,
            max_tokens=1000)
        llm_return = completion.choices[0].message.content
        if debug: print(llm_return)
        return CorrectionOutput(
            corrected=llm_return,
            correct_model=self.model_name
        )