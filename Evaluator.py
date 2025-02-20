""" This file contains the Evaluator classes
"""

class EvaluationModel:
    """A class to evaluate generated output.
    """

    def __init__(self):
        raise NotImplementedError
    
    def predict_one(self, claim: str, context: str):
        raise NotImplementedError
