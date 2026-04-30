"""
Authorship Detection Framework - CodeBERT Implementation
=======================================================
This module provides the concrete implementation for the CodeBERT-Authorship
detector. CodeBERT is a bimodal pre-trained model for programming languages, 
leveraging a RoBERTa-based architecture.

Technical Specification:
- Architecture: Encoder-only (RoBERTa)
- Pooling Strategy: [CLS] Token (Index 0)
- Metric Support: Categorical Prediction, Softmax Confidence, Latent Embeddings
"""

from transformers import AutoModelForSequenceClassification
import torch
from .base_detector import AuthorshipDetector


class CodeBERTDetector(AuthorshipDetector):
    """
    Concrete implementation of the AuthorshipDetector for the CodeBERT model.

    CodeBERT treats code as a sequence of tokens and uses the [CLS] token at
    index 0 to aggregate stylistic and semantic information for classification.
    """

    def _load_model(self, model_path: str):
        """
        Loads the CodeBERT model with a sequence classification head.

        This method initializes the model and its 'id2label' mapping. By enabling
        'output_hidden_states', we can access the [CLS] vector from the 12th
        layer for Stylistic Drift analysis.

        Args:
            model_path (str): The HuggingFace hub path or local directory containing
                              the fine-tuned CodeBERT weights.

        Returns:
            AutoModelForSequenceClassification: The loaded transformer model.
        """
        return AutoModelForSequenceClassification.from_pretrained(
            model_path, output_hidden_states=True, trust_remote_code=True
        )

    def _extract_pooling_state(self, outputs, inputs) -> torch.Tensor:
        """
        Extracts the stylistic latent vector from the [CLS] token.

        In CodeBERT's RoBERTa architecture, the [CLS] token (index 0) is
        specifically trained to represent the entire sequence for classification
        tasks.

        Args:
            outputs: The Raw ModelOutput from the forward pass.
            inputs: The tokenized input dictionary (kept for interface consistency).

        Returns:
            torch.Tensor: A 1D stylistic embedding vector of size (768,).
        """
        # CodeBERT is encoder-only, so we look directly at 'hidden_states'.
        # We take the last layer [-1].
        last_hidden_state = outputs.hidden_states[-1]

        # Extract the [CLS] vector at index 0.
        # Shape: (batch_size, sequence_length, hidden_dim) -> (batch_size, hidden_dim)
        embedding = last_hidden_state[:, 0, :]

        return embedding
