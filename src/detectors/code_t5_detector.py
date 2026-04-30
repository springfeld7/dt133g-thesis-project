"""
Authorship Detection Framework - CodeT5 Implementation
=====================================================
This module provides the concrete implementation for the CodeT5-Authorship
detector. CodeT5 is a unified pre-trained encoder-decoder transformer 
specifically designed for code-related tasks.

In this framework, we utilize the model's encoder to extract stylistic 
fingerprints. The latent representation is captured from the encoder's 
summary token, which serves as the primary feature vector for the 
Authorship Attribution classification head.

Technical Specification:
- Architecture: Encoder-Decoder (T5)
- Pooling Strategy: Encoder First-Token ([CLS]/<s> equivalent)
- Metric Support: Categorical Prediction, Softmax Confidence, Latent Embeddings
"""

from transformers import AutoModelForSequenceClassification
import torch
from .base_detector import AuthorshipDetector

class CodeT5Detector(AuthorshipDetector):
    """
    Concrete implementation of the AuthorshipDetector for the CodeT5-Authorship model.
    
    This class handles the specific architectural nuances of the Encoder-Decoder 
    structure used by CodeT5. It focuses on extracting the stylistic latent 
    representation from the Encoder for use in Stylistic Drift analysis (RQ3).
    """

    def _load_model(self, model_path: str):
        """
        Loads the CodeT5 model with a sequence classification head.

        This method initializes the model using the configuration stored at the 
        model_path, which includes the 'id2label' mapping that defines the 
        classification targets (e.g., 0: Human, 1: AI). By enabling 
        'output_hidden_states', the model is configured to return the full 
        internal representational layers required for Stylistic Drift analysis.

        Args:
            model_path (str): The HuggingFace hub path or local directory containing 
                              the fine-tuned weights, architecture config, and 
                              label mappings for authorship attribution.
        
        Returns:
            AutoModelForSequenceClassification: The loaded transformer model 
                                                configured for sequence classification 
                                                and hidden state extraction.
        """
        return AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            output_hidden_states=True,
            trust_remote_code=True
        )

    def _extract_pooling_state(self, outputs, inputs) -> torch.Tensor:
        """
        Extracts the latent style vector from the CodeT5 Encoder's summary token.
        
        Args:
            outputs: The Raw ModelOutput from the forward pass containing 
                     'encoder_hidden_states'.
            inputs: The tokenized input dictionary used for masking or indexing.

        Returns:
            torch.Tensor: A 1D stylistic embedding vector of size (hidden_dim,).
        """
        # Target the Encoder's last hidden layer. 
        # The encoder contains the primary representational understanding of the code.
        if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states:
            last_hidden_state = outputs.encoder_hidden_states[-1]
        else:
            # Fallback for specific model configurations
            last_hidden_state = outputs.hidden_states[-1]

        # Extract the summary vector ([CLS]-equivalent).
        # We take the first token of the sequence (index 0) for the batch.
        # This vector is the one passed to the classification layers.
        embedding = last_hidden_state[:, 0, :]
        
        return embedding
