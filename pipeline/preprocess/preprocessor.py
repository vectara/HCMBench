""" The file contains abstract class for a preprocessor """
from typing import Any, Dict
from ..processor import Processor

class Preprocessor(Processor):
    """ Abstract class for claim preprocessor """
    def __init__(self, input_column="corrected", output_column="processed", **kwargs):
        super().__init__(**kwargs)
        self.input_column = input_column
        self.output_column = output_column

    def merge_output(self, sample: Dict, output: Any) -> Dict:
        return {**sample, self.output_column: output}

    def process_one(self, sample: Dict) -> Any:
        raise NotImplementedError
