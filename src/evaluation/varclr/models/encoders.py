import os
from typing import List, Union

import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..data.preprocessor import CodePreprocessor
from . import urls_pretrained_model


class Encoder(nn.Module):
    @staticmethod
    def build(args) -> "Encoder":
        return {"bert": BERT}[args.model].from_args(
            args
        )

    @staticmethod
    def from_pretrained(model_name: str, save_path: str = "src/evaluation/varclr/saved/") -> "Encoder":
        return {
            "varclr-codebert": BERT,
        }[model_name].load(save_path)

    @staticmethod
    def from_args(args) -> "Encoder":
        raise NotImplementedError

    @staticmethod
    def load(save_path: str) -> "Encoder":
        raise NotImplementedError

    def forward(self, idxs, lengths):
        raise NotImplementedError

    def encode(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        raise NotImplementedError

    def score(
        self, inputx: Union[str, List[str]], inputy: Union[str, List[str]]
    ) -> List[float]:
        if type(inputx) != type(inputy):
            raise Exception("Input X and Y must be either string or list of strings.")
        if isinstance(inputx, list) and len(inputx) != len(inputy):
            raise Exception("Input X and Y must have the same length")
        embx = self.encode(inputx)
        emby = self.encode(inputy)
        return F.cosine_similarity(embx, emby).tolist()

    def cross_score(
        self, inputx: Union[str, List[str]], inputy: Union[str, List[str]]
    ) -> List[List[float]]:
        if isinstance(inputx, str):
            inputx = [inputx]
        if isinstance(inputy, str):
            inputy = [inputy]
        assert all(isinstance(inp, str) for inp in inputx)
        assert all(isinstance(inp, str) for inp in inputy)
        embx = self.encode(inputx)
        embx /= embx.norm(dim=1, keepdim=True)
        emby = self.encode(inputy)
        emby /= emby.norm(dim=1, keepdim=True)
        return (embx @ emby.t()).tolist()

    @staticmethod
    def decor_bert_forward(model_forward):
        """Decorate an encoder's forward pass to deal with raw inputs."""
        processor = CodePreprocessor()
        tokenizer = AutoTokenizer.from_pretrained(
            urls_pretrained_model.PRETRAINED_TOKENIZER
        )

        def tokenize_and_forward(self, inputs: Union[str, List[str]]) -> torch.Tensor:
            inputs = processor(inputs)
            return_dict = tokenizer(inputs, return_tensors="pt", padding=True)
            return model_forward(
                self, return_dict["input_ids"], return_dict["attention_mask"]
            )[0].detach()

        return tokenize_and_forward

class BERT(Encoder):
    """VarCLR-CodeBERT Model."""

    def __init__(self, bert_model: str, last_n_layer_output: int = 4):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(bert_model)
        self.last_n_layer_output = last_n_layer_output

    @staticmethod
    def from_args(args):
        return BERT(args.bert_model, args.last_n_layer_output)

    @staticmethod
    def load(save_path: str) -> "BERT":
        gdown.cached_download( # type: ignore
            urls_pretrained_model.PRETRAINED_CODEBERT_URL,
            os.path.join(save_path, "bert.zip"),
            hash=f"md5:{urls_pretrained_model.PRETRAINED_CODEBERT_MD5}",
            postprocess=gdown.extractall, # type: ignore
        )
        return BERT(
            bert_model=os.path.join(
                save_path, urls_pretrained_model.PRETRAINED_CODEBERT_FOLDER
            )
        )

    def forward(self, input_ids, attention_mask):
        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_hids = output.hidden_states
        pooled = all_hids[-self.last_n_layer_output][:, 0]

        return pooled, (all_hids, attention_mask)

    encode = Encoder.decor_bert_forward(forward)
