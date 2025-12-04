"""
Module for FAISS indexing and retrieval.
"""

import faiss
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    FAISS-based vector index for semantic search.
    """
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = None
        self.doc_ids = None
        self.doc_texts = None
        
    def build_index(self, embeddings: np.ndarray, doc_ids: List[str],
                    doc_texts: List[str] = None):
        """Build FAISS index from embeddings."""
        embeddings = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
        
    def search(self, query_embeddings: np.ndarray, top_k: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar documents."""
        query_embeddings = query_embeddings.astype(np.float32).copy()
        faiss.normalize_L2(query_embeddings)
        
        scores, indices = self.index.search(query_embeddings, top_k)
        
        return scores, indices
    
    def get_results(self, query_embedding: np.ndarray, top_k: int = 25) -> List[Dict]:
        """Get search results with document details."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        scores, indices = self.search(query_embedding, top_k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            result = {
                'rank': rank + 1,
                'doc_id': self.doc_ids[idx],
                'score': float(score),
            }
            if self.doc_texts:
                result['text'] = self.doc_texts[idx][:200] + "..."
            results.append(result)
            
        return results


# Test the index
if __name__ == "__main__":
    print("Testing FAISS index...")
    
    # Create dummy data
    dim = 384
    n_docs = 100
    
    embeddings = np.random.randn(n_docs, dim).astype(np.float32)
    doc_ids = [f"doc_{i}" for i in range(n_docs)]
    
    # Build index
    index = FAISSIndex(dim)
    index.build_index(embeddings, doc_ids)
    
    # Search
    query = np.random.randn(1, dim).astype(np.float32)
    results = index.get_results(query, top_k=5)
    
    print("Top 5 results:")
    for r in results:
        print(f"  Rank {r['rank']}: {r['doc_id']} (score: {r['score']:.4f})")