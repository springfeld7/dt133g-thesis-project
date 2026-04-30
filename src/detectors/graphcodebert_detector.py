"""
Authorship Detection Framework - GraphCodeBERT Implementation
===========================================================
This module provides the concrete implementation for the GraphCodeBERT 
detector. GraphCodeBERT extends CodeBERT by incorporating the 
semantic structure of code (Data Flow Graphs).

Technical Specification:
- Architecture: Encoder-only (RoBERTa-based with DFG)
- Pooling Strategy: [CLS] Token (Index 0)
- Metric Support: Categorical Prediction, Softmax Confidence, Latent Embeddings
"""

from transformers import AutoModelForSequenceClassification
import torch
from .base_detector import AuthorshipDetector


class GraphCodeBERTDetector(AuthorshipDetector):
    """
    Concrete implementation of the AuthorshipDetector for GraphCodeBERT.

    GraphCodeBERT is particularly robust against simple mutations because
    it maps variable dependencies. We extract the latent representation
    from the [CLS] token to measure how mutations affect this graph-aware
    embedding.
    """

    def _load_model(self, model_path: str):
        """
        Loads the GraphCodeBERT model with a sequence classification head.

        Args:
            model_path (str): The HuggingFace hub path or local directory
                              containing the weights fine-tuned for authorship.

        Returns:
            AutoModelForSequenceClassification: The loaded transformer model.
        """
        return AutoModelForSequenceClassification.from_pretrained(
            model_path, output_hidden_states=True, trust_remote_code=True
        )

    def _extract_pooling_state(self, outputs, inputs) -> torch.Tensor:
        """
        Extracts the stylistic latent vector from the GraphCodeBERT summary token.

        Args:
            outputs: The Raw ModelOutput from the forward pass.
            inputs: The tokenized input dictionary (kept for interface consistency).

        Returns:
            torch.Tensor: A 1D stylistic embedding vector of size (768,).
        """
        # GraphCodeBERT uses a standard transformer encoder stack.
        # The hidden_states[-1] represents the final layer including graph information.
        last_hidden_state = outputs.hidden_states[-1]

        # Extract the [CLS] vector at index 0.
        # This vector incorporates both textual syntax and data flow semantics.
        embedding = last_hidden_state[:, 0, :]

        return embedding
