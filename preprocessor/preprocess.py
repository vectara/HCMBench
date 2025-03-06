class Preprocessor:
    """ Abstract class for claim preprocessor """
    def __init__(self, input_column="corrected", output_column="processed", **kwargs):
        self.input_column = input_column
        self.output_column = output_column

    def process_one(self):
        raise NotImplementedError