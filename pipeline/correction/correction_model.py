""" This file is for an abstract HCM model """
from typing import Dict, Optional
from ..processor import Processor, ProcessorOutput

class CorrectionOutput(ProcessorOutput):
    corrected: str
    correct_model: str
    extra_output: Optional[str] = None

class CorrectionModel(Processor):
    """ Abstract class for a correction model """
    def __init__(self, model_name,
                 claim_column='claim',
                 context_column='context',
                 filter_exp=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.claim_column = claim_column
        self.context_column = context_column
        if filter_exp is not None:
            self.filter_fn = eval(filter_exp)
            self.process_fn = self.process_one
            self.process_one = self.process_one_with_filter

    def process_one_with_filter(self, sample:Dict, **kwargs) -> CorrectionOutput:
        if self.filter_fn(sample):
            return self.process_fn(sample, **kwargs)
        return CorrectionOutput(
            corrected=sample["claim"],
            correct_model=self.model_name,
            extra_output=None
        )

    def process_one(self, sample:Dict) -> CorrectionOutput:
        """ 
        The function implements the correction process taking sample as input.
        Each sample should include two columns: 
            
            "claim": model's generation, may or may not be factually error
            "context": the grounding material used to generate the claim 
        """
        raise NotImplementedError

class IdenticalCorrectionModel(CorrectionModel):
    """ A pseudo correction model that returns the exact same response """
    def __init__(self, model_name='Identical'):
        super().__init__(model_name)

    def process_one(self, sample:Dict) -> CorrectionOutput:
        return CorrectionOutput(
            corrected=sample[self.claim_column],
            correct_model=self.model_name,
        )
