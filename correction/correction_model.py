# This file is for an abstract HCM model
from typing import Optional
from pydantic import BaseModel

class CorrectionOutput(BaseModel):
    claim: str
    context: str
    corrected: str
    correct_model: str
    extra_output: Optional[str] = None

class CorrectionModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def correct_one(self, generated:str, context:str) -> CorrectionOutput:
        raise NotImplementedError


class IdenticalCorrectionModel(CorrectionModel):
    def __init__(self, model_name='Identical'):
        super().__init__(model_name)

    def correct_one(self, claim:str, context:str):
        return CorrectionOutput(
            claim=claim,
            context=context,
            corrected=claim,
            correct_model=self.model_name
        )

if __name__ == '__main__':
    ICM = IdenticalCorrectionModel()
    print(ICM.correct_one("Hello", "Hello world!"))