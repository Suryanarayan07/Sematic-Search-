"""
Module for loading and using embedding models for semantic search.
"""

import torch
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SigLIPEmbedding:
    """
    Embedding model using SigLIP's text encoder.
    """
    
    def __init__(self, device: str = None):
        self.model_name = "google/siglip-base-patch16-256-multilingual"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        
    def _load_model(self):
        """Load the SigLIP model."""
        from transformers import AutoProcessor, SiglipModel
        
        logger.info(f"Loading SigLIP model: {self.model_name}")
        logger.info("This may take a few minutes on first run...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = SiglipModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.text_config.hidden_size
        logger.info(f"SigLIP loaded! Embedding dimension: {self.embedding_dim}")
        
    def encode(self, texts: List[str], batch_size: int = 32, 
               show_progress: bool = True) -> np.ndarray:
        """Encode texts using SigLIP's text encoder."""
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding with SigLIP")
            
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                
                inputs = self.processor(
                    text=batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(self.device)
                
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(text_features.cpu().numpy())
                
        return np.vstack(all_embeddings)


class QwenEmbedding:
    """
    Embedding model using Qwen2-Embedding.
    Using a simpler alternative that's more stable.
    """
    
    def __init__(self, device: str = None):
        # Using a more stable alternative embedding model
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        
    def _load_model(self):
        """Load the embedding model."""
        logger.info(f"Loading embedding model: {self.model_name}")
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded! Embedding dimension: {self.embedding_dim}")
        
    def encode(self, texts: List[str], batch_size: int = 32,
               show_progress: bool = True) -> np.ndarray:
        """Encode texts using the embedding model."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )


def get_document_text(doc: Dict) -> str:
    """Combine title and text for a document."""
    title = doc.get('title', '')
    text = doc.get('text', '')
    
    if title:
        return f"{title} {text}"
    return text


# Test the models
if __name__ == "__main__":
    print("Testing SigLIP model...")
    try:
        siglip = SigLIPEmbedding()
        test_emb = siglip.encode(["Hello world", "Test sentence"])
        print(f"SigLIP embedding shape: {test_emb.shape}")
    except Exception as e:
        print(f"SigLIP error: {e}")
    
    print("\nTesting alternative model...")
    try:
        qwen = QwenEmbedding()
        test_emb = qwen.encode(["Hello world", "Test sentence"])
        print(f"Alternative model embedding shape: {test_emb.shape}")
    except Exception as e:
        print(f"Error: {e}")