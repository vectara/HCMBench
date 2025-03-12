# This file is for an abstract HCM model
from typing import Dict, Optional
from ..processor import Processor, ProcessorOutput

class CorrectionOutput(ProcessorOutput):
    corrected: str
    correct_model: str
    extra_output: Optional[str] = None

class CorrectionModel(Processor):
    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    def process_one(self, sample:Dict) -> CorrectionOutput:
        """ 
        The function implements the correction process taking sample as input.
        Each sample should include two columns: 
            "claim": model's generation, may or may not be factually error
            "context": the grounding material used to generate the claim 
        """
        raise NotImplementedError

class IdenticalCorrectionModel(CorrectionModel):
    def __init__(self, model_name='Identical'):
        super().__init__(model_name)

    def process_one(self, sample:Dict) -> CorrectionOutput:
        return CorrectionOutput(
            corrected=sample["claim"],
            correct_model=self.model_name,
        )

def main():
    icm = IdenticalCorrectionModel()
    print(icm.process_one({"claim": "You happy", "context": "I not happy"}))
