"""
Authorship Detection Framework - Inference Contract
==================================================
This module defines the abstract interface for SOTA authorship detectors.
It is designed to support black-box evaluation of 5,100 samples across
multiple mutation tiers to quantify the Robustness Gradient (RQ3).

Supported Architectures:
- CodeT5-Authorship (Encoder-Decoder)
- CodeBERT (Encoder-only)
- GraphCodeBERT (Graph-aware)
- UniXcoder (AST-aware)
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


class AuthorshipDetector(torch.nn.Module):
    """
    Abstract Base Class for Authorship Detectors.

    Ensures that regardless of the underlying transformer architecture,
    the experiment script can consistently extract predictions and
    high-dimensional embeddings for stylistic drift analysis.

    Attributes:
        device (str): The computation device ('cuda' or 'cpu').
        model_path (str): The HuggingFace hub path or local directory of the model.
        tokenizer: The tokenizer associated with the model.
        model: The loaded transformer model for inference.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initializes the detector and moves the model to the specified device.

        Args:
            model_path (str): The HuggingFace hub path or local directory.
            device (str): 'cuda' or 'cpu'. Defaults to auto-detection.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        print(f"[*] Loading model: {model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = self._load_model(model_path).to(self.device)
        self.model.eval()

    @abstractmethod
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Factory method to load the specific model class.
        Some models use SequenceClassification, others may require
        a base model with a custom head.
        """
        pass

    @torch.no_grad()
    def get_inference_data(self, code_snippet: str, original_embedding: Optional[torch.Tensor] = None):
        """
        The core experiment method. Performs a single forward pass to
        extract both the categorical prediction and the latent embedding.

        Args:
            code_snippet (str): Raw source code to evaluate.
            original_embedding (Tensor, optional): The embedding of the non-mutated
                version of this code for RQ3 similarity calculation.

        Returns:
            dict: {
                'prediction': int,
                'confidence': float,
                'embedding': torch.Tensor,
                'cosine_similarity': float (if original_embedding provided)
            }
        """
        inputs = self.tokenizer(
            code_snippet, return_tensors="pt", truncation=True, padding="max_length", max_length=512
        ).to(self.device)

        # Force model to return hidden states for embedding extraction
        outputs = self.model(**inputs, output_hidden_states=True)

        # 1. Extract Prediction & Confidence
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1).item()
        confidence = torch.max(probs).item()

        # 2. Extract Embedding (CLS / Pooled State)
        # We take the [CLS] token from the last hidden layer
        embedding = self._extract_pooling_state(outputs, inputs)

        # 3. Calculate Robustness Gradient (RQ3)
        similarity = 1.0
        if original_embedding is not None:
            similarity = F.cosine_similarity(
                embedding.view(1, -1), original_embedding.view(1, -1)
            ).item()

        return {
            "prediction": prediction,
            "confidence": confidence,
            "embedding": embedding,
            "cosine_similarity": similarity,
        }

    @abstractmethod
    def _extract_pooling_state(self, outputs, inputs) -> torch.Tensor:
        """
        Architecture-specific logic to extract the representative
        author-style vector (e.g., [CLS] token or Encoder output).
        """
        pass
