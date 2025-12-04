"""
Module for evaluation metrics, specifically MAP@25.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_relevant(relevance_score: int, threshold: int = 1) -> bool:
    """Determine if a document is relevant based on score."""
    return relevance_score >= threshold


def compute_ap_at_k(retrieved_docs: List[str], 
                    relevant_docs: set,
                    k: int = 25) -> Tuple[float, Dict]:
    """
    Compute Average Precision at K for a single query.
    """
    retrieved_docs = retrieved_docs[:k]
    
    num_relevant = len(relevant_docs)
    if num_relevant == 0:
        return 0.0, {'ap': 0.0, 'num_relevant': 0, 'hits': 0}
    
    hits = 0
    sum_precisions = 0.0
    
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i
    
    denominator = min(k, num_relevant)
    ap = sum_precisions / denominator if denominator > 0 else 0.0
    
    return ap, {
        'ap': ap,
        'num_relevant': num_relevant,
        'num_relevant_in_top_k': min(k, num_relevant),
        'hits': hits,
        'recall': hits / num_relevant if num_relevant > 0 else 0,
    }


def compute_map_at_k(results: Dict[str, Dict[str, float]],
                     qrels: Dict[str, Dict[str, int]],
                     k: int = 25) -> Tuple[float, Dict]:
    """
    Compute Mean Average Precision at K across all queries.
    """
    ap_scores = {}
    detailed_metrics = {}
    
    for query_id in results.keys():
        if query_id not in qrels:
            logger.warning(f"Query {query_id} not found in qrels")
            continue
            
        query_results = results[query_id]
        retrieved_docs = sorted(
            query_results.keys(),
            key=lambda x: query_results[x],
            reverse=True
        )[:k]
        
        relevant_docs = set(
            doc_id for doc_id, score in qrels[query_id].items()
            if is_relevant(score)
        )
        
        ap, metrics = compute_ap_at_k(retrieved_docs, relevant_docs, k)
        ap_scores[query_id] = ap
        detailed_metrics[query_id] = {
            **metrics,
            'retrieved_docs': retrieved_docs,
            'relevant_docs': relevant_docs
        }
    
    map_score = np.mean(list(ap_scores.values())) if ap_scores else 0.0
    
    return map_score, {
        'map': map_score,
        'per_query_ap': ap_scores,
        'detailed': detailed_metrics
    }


class Evaluator:
    """Comprehensive evaluator for semantic search results."""
    
    def __init__(self, qrels: Dict[str, Dict[str, int]], k: int = 25):
        self.qrels = qrels
        self.k = k
        
    def evaluate(self, results: Dict[str, Dict[str, float]]) -> Dict:
        """Evaluate search results against ground truth."""
        map_score, detailed = compute_map_at_k(results, self.qrels, self.k)
        
        all_recalls = []
        all_precisions = []
        all_hits = []
        
        for query_id, metrics in detailed['detailed'].items():
            all_recalls.append(metrics['recall'])
            all_precisions.append(metrics['hits'] / self.k if self.k > 0 else 0)
            all_hits.append(metrics['hits'])
            
        return {
            f'MAP@{self.k}': map_score,
            f'Mean Recall@{self.k}': np.mean(all_recalls) if all_recalls else 0,
            f'Mean Precision@{self.k}': np.mean(all_precisions) if all_precisions else 0,
            f'Mean Hits@{self.k}': np.mean(all_hits) if all_hits else 0,
            'per_query_metrics': detailed['detailed']
        }