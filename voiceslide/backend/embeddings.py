"""
VoiceSlide — Embeddings Service
Provides a shared module to lazily load and utilize the sentence-transformers model.
We use a singleton to prevent multiple instances from eating RAM.
"""

from typing import List
import logging

logger = logging.getLogger("voiceslide.nlp.embeddings")

# Use a fast, small model that works well for generalized semantic matching
MODEL_NAME = "all-MiniLM-L6-v2"

_model_instance = None


def get_embedding_model():
    """Lazy-loads the sentence-transformers model onto the optimal device."""
    global _model_instance
    if _model_instance is None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Determine the best device available
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading sentence-transformers model ({MODEL_NAME}) on {device_name}...")
            
            _model_instance = SentenceTransformer(MODEL_NAME, device=device_name)
            logger.info("SentenceTransformer model loaded successfully.")
            
        except ImportError:
            logger.error("sentence_transformers or torch is not installed.")
            raise
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings on preferred device: {e}. Falling back to cpu")
            _model_instance = SentenceTransformer(MODEL_NAME, device="cpu")
            
    return _model_instance


def encode(texts: List[str]):
    """
    Returns the tensor embeddings for a list of strings.
    """
    model = get_embedding_model()
    return model.encode(texts, convert_to_tensor=True)
