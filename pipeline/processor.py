""" The file contains the base Processor call inherited by metric/hcm/preprocessor """

import time
from typing import Dict

from datasets import Dataset
from pydantic import BaseModel

class ProcessorOutput(BaseModel):
    pass

class Processor:
    """ Abstract class for a processor """
    def __init__(self, num_proc=1, RPS=0, **kwargs):
        """
        Basic configurations for a processor
        Args:
            num_proc: the number of CPU process used when apply Dataset.map
            RPS: rate per second, 0 meaning no limit constraint, used for handling LLM API calls
        """
        self.num_proc = num_proc
        self.RPS = RPS
        if len(kwargs) > 0:
            print("Unused kwargs:", kwargs)

    def process_one(self, sample: Dict) -> ProcessorOutput:
        """ The main processing step happens here """
        raise NotImplementedError

    def merge_output(self, sample: Dict , output: ProcessorOutput) -> Dict:
        """ Merging output results back to the sample """
        return {**sample, **output.model_dump()}

    def map_fn(self, sample: Dict) -> Dict:
        """ Map function called with RPS handling """
        output = self.process_one(sample)
        if self.RPS != 0:
            time.sleep(1/self.RPS * self.num_proc)
        return self.merge_output(sample, output)

    def process_dataset(self, data: Dataset, num_proc=1) -> Dataset:
        """ Process a dataset with this processor, can be override for batch processing """
        data = data.map(
            self.map_fn,
            num_proc=num_proc
        )
        return data
