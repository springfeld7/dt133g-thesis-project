import re
from typing import List, Tuple, Union


class Preprocessor:
    @staticmethod
    def build(data_file, args) -> Tuple["Preprocessor", "Preprocessor"]:
        if "idbench" in data_file:
            print(f"Using code processor for {data_file}")
            return CodePreprocessor.from_args(args), CodePreprocessor.from_args(args)
        elif "20k" in data_file:
            return Preprocessor(), Preprocessor()
        else:
            raise NotImplementedError

    def __call__(self, sentence):
        return sentence


class CodePreprocessor(Preprocessor):
    def __init__(self, tokenization=None, sp_model=None):
        self.tokenization = tokenization

    @staticmethod
    def from_args(args) -> "CodePreprocessor":
        return CodePreprocessor(args.tokenization, args.sp_model)

    def __call__(self, var: Union[str, List[str]]):
        if isinstance(var, str):
            return self._process(var)
        elif isinstance(var, list) and all(isinstance(v, str) for v in var):
            return [self._process(v) for v in var]
        else:
            raise NotImplementedError

    def _process(self, var):
        var = var.replace("@", "")
        var = re.sub("([a-z]|^)([A-Z]{1})", r"\1_\2", var).lower().replace("_", " ").strip()
        return var
