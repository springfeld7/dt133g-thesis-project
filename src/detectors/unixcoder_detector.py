"""
Authorship Detection Framework - UniXcoder Implementation
========================================================
This module provides the concrete implementation for the UniXcoder-Authorship
detector. UniXcoder is a unified model that supports multiple modes 
(Encoder, Decoder, etc.) via specialized attention masks.

Technical Specification:
- Architecture: Unified Transformer (RoBERTa-based)
- Pooling Strategy: [CLS] Token (Index 0)
- Metric Support: Categorical Prediction, Softmax Confidence, Latent Embeddings
"""

from transformers import AutoModelForSequenceClassification
import torch
from .base_detector import AuthorshipDetector


class UniXcoderDetector(AuthorshipDetector):
    """
    Concrete implementation of the AuthorshipDetector for UniXcoder.

    UniXcoder's unified architecture allows it to capture deep semantic
    relationships. We utilize the global summary representation at index 0
    to measure stylistic drift.
    """

    def _load_model(self, model_path: str):
        """
        Loads the UniXcoder model with a sequence classification head.

        Args:
            model_path (str): The path to the fine-tuned UniXcoder weights.

        Returns:
            AutoModelForSequenceClassification: The loaded transformer model.
        """
        # UniXcoder often requires trust_remote_code=True if using custom
        # HuggingFace implementations.
        return AutoModelForSequenceClassification.from_pretrained(
            model_path, output_hidden_states=True, trust_remote_code=True
        )

    def _extract_pooling_state(self, outputs, inputs) -> torch.Tensor:
        """
        Extracts the stylistic latent vector from the UniXcoder summary token.

        Args:
            outputs: The Raw ModelOutput from the forward pass.
            inputs: The tokenized input dictionary (retained for interface parity).

        Returns:
            torch.Tensor: A 1D stylistic embedding vector of size (768,).
        """
        # UniXcoder, like CodeBERT, stores hidden states in a unified list.
        # We target the last layer [-1].
        last_hidden_state = outputs.hidden_states[-1]

        # Extract the summary vector at index 0.
        # In UniXcoder, this represents the unified context of the snippet.
        embedding = last_hidden_state[:, 0, :]

        return embedding
