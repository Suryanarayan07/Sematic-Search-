"""
Module for detailed False Positive and False Negative analysis.
"""

import numpy as np
from typing import Dict, List, Set
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FPFNAnalyzer:
    """
    Analyzer for False Positives and False Negatives in search results.
    """
    
    def __init__(self, corpus: Dict, qrels: Dict, k: int = 25):
        self.corpus = corpus
        self.qrels = qrels
        self.k = k
        
    def analyze_query(self, query_id: str, query_text: str,
                      retrieved_docs: List[str],
                      scores: Dict[str, float]) -> Dict:
        """Analyze FP and FN for a single query."""
        
        relevant_docs = set(
            doc_id for doc_id, score in self.qrels.get(query_id, {}).items()
            if score > 0
        )
        
        retrieved_set = set(retrieved_docs[:self.k])
        
        true_positives = retrieved_set & relevant_docs
        false_positives = retrieved_set - relevant_docs
        false_negatives = relevant_docs - retrieved_set
        
        # Detailed FP analysis
        fp_analysis = []
        for doc_id in list(false_positives)[:5]:  # Top 5 FPs
            doc = self.corpus.get(doc_id, {})
            rank = retrieved_docs.index(doc_id) + 1 if doc_id in retrieved_docs else -1
            
            fp_analysis.append({
                'doc_id': doc_id,
                'rank': rank,
                'score': scores.get(doc_id, 0),
                'title': doc.get('title', '')[:80],
                'text_snippet': doc.get('text', '')[:150],
            })
            
        # Detailed FN analysis
        fn_analysis = []
        for doc_id in list(false_negatives)[:5]:  # Top 5 FNs
            doc = self.corpus.get(doc_id, {})
            
            fn_analysis.append({
                'doc_id': doc_id,
                'ground_truth_score': self.qrels.get(query_id, {}).get(doc_id, 0),
                'title': doc.get('title', '')[:80],
                'text_snippet': doc.get('text', '')[:150],
            })
        
        fp_analysis.sort(key=lambda x: x['rank'])
        fn_analysis.sort(key=lambda x: x['ground_truth_score'], reverse=True)
        
        return {
            'query_id': query_id,
            'query_text': query_text,
            'num_relevant': len(relevant_docs),
            'num_retrieved': len(retrieved_set),
            'num_true_positives': len(true_positives),
            'num_false_positives': len(false_positives),
            'num_false_negatives': len(false_negatives),
            'precision': len(true_positives) / len(retrieved_set) if retrieved_set else 0,
            'recall': len(true_positives) / len(relevant_docs) if relevant_docs else 0,
            'true_positives': list(true_positives),
            'false_positives': fp_analysis,
            'false_negatives': fn_analysis
        }
    
    def analyze_all_queries(self, queries: Dict, results: Dict) -> Dict:
        """Analyze FP and FN for all queries."""
        all_analysis = {}
        aggregate_stats = defaultdict(list)
        
        for query_id in queries:
            if query_id not in results:
                continue
                
            query_results = results[query_id]
            retrieved_docs = sorted(
                query_results.keys(),
                key=lambda x: query_results[x],
                reverse=True
            )[:self.k]
            
            analysis = self.analyze_query(
                query_id,
                queries[query_id],
                retrieved_docs,
                query_results
            )
            
            all_analysis[query_id] = analysis
            
            aggregate_stats['precision'].append(analysis['precision'])
            aggregate_stats['recall'].append(analysis['recall'])
            aggregate_stats['num_fp'].append(analysis['num_false_positives'])
            aggregate_stats['num_fn'].append(analysis['num_false_negatives'])
            aggregate_stats['num_tp'].append(analysis['num_true_positives'])
            
        return {
            'per_query': all_analysis,
            'aggregate': {
                'mean_precision': np.mean(aggregate_stats['precision']),
                'mean_recall': np.mean(aggregate_stats['recall']),
                'total_fp': sum(aggregate_stats['num_fp']),
                'total_fn': sum(aggregate_stats['num_fn']),
                'total_tp': sum(aggregate_stats['num_tp']),
                'mean_fp_per_query': np.mean(aggregate_stats['num_fp']),
                'mean_fn_per_query': np.mean(aggregate_stats['num_fn'])
            }
        }
    
    def generate_report(self, analysis: Dict, model_name: str) -> str:
        """Generate a human-readable report."""
        report = []
        report.append("=" * 70)
        report.append(f"FALSE POSITIVE / FALSE NEGATIVE ANALYSIS REPORT")
        report.append(f"Model: {model_name}")
        report.append("=" * 70)
        
        agg = analysis['aggregate']
        report.append(f"\n### AGGREGATE STATISTICS ###")
        report.append(f"Mean Precision@{self.k}: {agg['mean_precision']:.4f}")
        report.append(f"Mean Recall@{self.k}: {agg['mean_recall']:.4f}")
        report.append(f"Total True Positives: {agg['total_tp']}")
        report.append(f"Total False Positives: {agg['total_fp']}")
        report.append(f"Total False Negatives: {agg['total_fn']}")
        
        report.append(f"\n### PER-QUERY ANALYSIS ###")
        
        for query_id, qa in analysis['per_query'].items():
            report.append(f"\n--- Query: {query_id} ---")
            report.append(f"Text: {qa['query_text'][:80]}...")
            report.append(f"Relevant: {qa['num_relevant']} | TP: {qa['num_true_positives']} | FP: {qa['num_false_positives']} | FN: {qa['num_false_negatives']}")
            report.append(f"Precision: {qa['precision']:.4f} | Recall: {qa['recall']:.4f}")
            
            if qa['false_positives']:
                report.append("\nTop False Positives (wrongly retrieved):")
                for fp in qa['false_positives'][:2]:
                    report.append(f"  - Rank {fp['rank']}: {fp['title'][:50]}...")
                    
            if qa['false_negatives']:
                report.append("\nTop False Negatives (missed relevant docs):")
                for fn in qa['false_negatives'][:2]:
                    report.append(f"  - Relevance {fn['ground_truth_score']}: {fn['title'][:50]}...")
        
        return "\n".join(report)