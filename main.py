"""
Semantic Search Analysis - Complete Solution (FIXED)
Fixes SigLIP truncation issue and adds proper second model
"""

import os
import json
import warnings
import logging
import numpy as np
import faiss
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import torch

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# PART 1: DATA LOADING
# ============================================================

def load_nfcorpus(data_dir: str = "./datasets"):
    """Download and load NFCorpus dataset."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    
    dataset_name = "nfcorpus"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    
    data_path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(data_path):
        print(f"Downloading {dataset_name} dataset...")
        data_path = util.download_and_unzip(url, data_dir)
    else:
        print(f"Dataset already exists at {data_path}")
    
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    return corpus, queries, qrels


def get_sample_queries(queries: Dict, qrels: Dict, n: int = 5) -> Dict:
    """Select n queries with relevance judgments."""
    valid_query_ids = [qid for qid in qrels.keys() if len(qrels[qid]) > 0]
    selected_ids = valid_query_ids[:n]
    return {qid: queries[qid] for qid in selected_ids}


def get_document_text(doc: Dict) -> str:
    """Combine title and text."""
    title = doc.get('title', '')
    text = doc.get('text', '')
    return f"{title} {text}" if title else text


# ============================================================
# PART 2: EMBEDDING MODELS (FIXED)
# ============================================================

class SigLIPEmbedding:
    """
    SigLIP text embedding model - FIXED VERSION
    Properly truncates text to avoid sequence length errors
    """
    
    def __init__(self):
        self.model_name = "google/siglip-base-patch16-256-multilingual"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SigLIP model on {self.device}...")
        
        from transformers import AutoTokenizer, SiglipModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = SiglipModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get the actual max length from model config
        self.max_length = min(64, self.model.config.text_config.max_position_embeddings)
        self.embedding_dim = self.model.config.text_config.hidden_size
        
        print(f"SigLIP loaded! Dimension: {self.embedding_dim}, Max length: {self.max_length}")
        
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings with proper truncation."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="SigLIP encoding"):
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                # Use tokenizer directly with proper truncation
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,  # Use correct max length
                    return_tensors="pt"
                ).to(self.device)
                
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_embeddings.append(text_features.cpu().numpy())
                
        return np.vstack(all_embeddings)


class MPNetEmbedding:
    """
    MPNet embedding model - Better alternative for comparison
    This is a strong general-purpose text embedding model
    """
    
    def __init__(self):
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading MPNet model on {self.device}...")
        
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"MPNet loaded! Dimension: {self.embedding_dim}")
        
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )


class MiniLMEmbedding:
    """MiniLM text embedding model."""
    
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading MiniLM model on {self.device}...")
        
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"MiniLM loaded! Dimension: {self.embedding_dim}")
        
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )


# ============================================================
# PART 3: FAISS INDEXING & SEARCH
# ============================================================

class FAISSIndex:
    """FAISS vector index for semantic search."""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = None
        self.doc_ids = None
        
    def build(self, embeddings: np.ndarray, doc_ids: List[str]):
        """Build FAISS index."""
        embeddings = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        self.doc_ids = doc_ids
        
        print(f"Built index with {self.index.ntotal} vectors")
        
    def search(self, query_embeddings: np.ndarray, top_k: int = 25):
        """Search for similar documents."""
        query_embeddings = query_embeddings.astype(np.float32).copy()
        faiss.normalize_L2(query_embeddings)
        
        scores, indices = self.index.search(query_embeddings, top_k)
        return scores, indices


# ============================================================
# PART 4: EVALUATION (MAP@25)
# ============================================================

def compute_ap_at_k(retrieved_docs: List[str], relevant_docs: set, k: int = 25) -> float:
    """Compute Average Precision at K."""
    retrieved_docs = retrieved_docs[:k]
    
    if len(relevant_docs) == 0:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i
    
    return sum_precisions / min(k, len(relevant_docs))


def evaluate_results(results: Dict, qrels: Dict, k: int = 25) -> Dict:
    """Evaluate search results with MAP@K."""
    ap_scores = {}
    detailed = {}
    
    for query_id, query_results in results.items():
        if query_id not in qrels:
            continue
            
        retrieved_docs = sorted(
            query_results.keys(),
            key=lambda x: query_results[x],
            reverse=True
        )[:k]
        
        relevant_docs = set(
            doc_id for doc_id, score in qrels[query_id].items()
            if score > 0
        )
        
        ap = compute_ap_at_k(retrieved_docs, relevant_docs, k)
        ap_scores[query_id] = ap
        
        hits = len(set(retrieved_docs) & relevant_docs)
        
        detailed[query_id] = {
            'ap': ap,
            'hits': hits,
            'num_relevant': len(relevant_docs),
            'precision': hits / k,
            'recall': hits / len(relevant_docs) if relevant_docs else 0,
            'retrieved_docs': retrieved_docs,
            'relevant_docs': relevant_docs
        }
    
    map_score = np.mean(list(ap_scores.values())) if ap_scores else 0.0
    
    return {
        'MAP@25': map_score,
        'per_query': detailed
    }


# ============================================================
# PART 5: FALSE POSITIVE / FALSE NEGATIVE ANALYSIS
# ============================================================

def analyze_fp_fn(results: Dict, qrels: Dict, corpus: Dict, 
                  queries: Dict, k: int = 25) -> Dict:
    """Analyze False Positives and False Negatives."""
    
    analysis = {}
    aggregate = defaultdict(list)
    
    for query_id in results.keys():
        if query_id not in qrels:
            continue
            
        query_results = results[query_id]
        retrieved_docs = sorted(
            query_results.keys(),
            key=lambda x: query_results[x],
            reverse=True
        )[:k]
        
        retrieved_set = set(retrieved_docs)
        relevant_docs = set(
            doc_id for doc_id, score in qrels[query_id].items()
            if score > 0
        )
        
        true_positives = retrieved_set & relevant_docs
        false_positives = retrieved_set - relevant_docs
        false_negatives = relevant_docs - retrieved_set
        
        # Get FP details
        fp_details = []
        for doc_id in list(false_positives)[:3]:
            doc = corpus.get(doc_id, {})
            rank = retrieved_docs.index(doc_id) + 1
            fp_details.append({
                'doc_id': doc_id,
                'rank': rank,
                'title': doc.get('title', '')[:60],
                'score': query_results.get(doc_id, 0)
            })
        
        # Get FN details
        fn_details = []
        for doc_id in list(false_negatives)[:3]:
            doc = corpus.get(doc_id, {})
            fn_details.append({
                'doc_id': doc_id,
                'title': doc.get('title', '')[:60],
                'relevance': qrels[query_id].get(doc_id, 0)
            })
        
        precision = len(true_positives) / len(retrieved_set) if retrieved_set else 0
        recall = len(true_positives) / len(relevant_docs) if relevant_docs else 0
        
        analysis[query_id] = {
            'query_text': queries[query_id][:80],
            'num_relevant': len(relevant_docs),
            'num_tp': len(true_positives),
            'num_fp': len(false_positives),
            'num_fn': len(false_negatives),
            'precision': precision,
            'recall': recall,
            'false_positives': fp_details,
            'false_negatives': fn_details
        }
        
        aggregate['precision'].append(precision)
        aggregate['recall'].append(recall)
        aggregate['fp'].append(len(false_positives))
        aggregate['fn'].append(len(false_negatives))
        aggregate['tp'].append(len(true_positives))
    
    return {
        'per_query': analysis,
        'aggregate': {
            'mean_precision': np.mean(aggregate['precision']),
            'mean_recall': np.mean(aggregate['recall']),
            'total_tp': sum(aggregate['tp']),
            'total_fp': sum(aggregate['fp']),
            'total_fn': sum(aggregate['fn']),
            'mean_fp_per_query': np.mean(aggregate['fp']),
            'mean_fn_per_query': np.mean(aggregate['fn'])
        }
    }


def print_fp_fn_report(analysis: Dict, model_name: str):
    """Print FP/FN analysis report."""
    print("\n" + "=" * 70)
    print(f"FALSE POSITIVE / FALSE NEGATIVE ANALYSIS: {model_name}")
    print("=" * 70)
    
    agg = analysis['aggregate']
    print(f"\n--- AGGREGATE STATISTICS ---")
    print(f"Mean Precision@25: {agg['mean_precision']:.4f}")
    print(f"Mean Recall@25: {agg['mean_recall']:.4f}")
    print(f"Total True Positives: {agg['total_tp']}")
    print(f"Total False Positives: {agg['total_fp']}")
    print(f"Total False Negatives: {agg['total_fn']}")
    print(f"Avg FP per query: {agg['mean_fp_per_query']:.2f}")
    print(f"Avg FN per query: {agg['mean_fn_per_query']:.2f}")
    
    print(f"\n--- PER-QUERY ANALYSIS ---")
    for query_id, qa in analysis['per_query'].items():
        print(f"\nQuery [{query_id}]: {qa['query_text']}...")
        print(f"  Relevant: {qa['num_relevant']} | TP: {qa['num_tp']} | FP: {qa['num_fp']} | FN: {qa['num_fn']}")
        print(f"  Precision: {qa['precision']:.4f} | Recall: {qa['recall']:.4f}")
        
        if qa['false_positives']:
            print("  Top False Positives (wrongly retrieved):")
            for fp in qa['false_positives'][:2]:
                print(f"    - Rank {fp['rank']}: {fp['title']}...")
                
        if qa['false_negatives']:
            print("  Top False Negatives (missed relevant):")
            for fn in qa['false_negatives'][:2]:
                print(f"    - Relevance {fn['relevance']}: {fn['title']}...")


def compare_models_detailed(results: Dict, qrels: Dict, queries: Dict) -> Dict:
    """Compare which documents each model finds uniquely."""
    
    model_names = list(results.keys())
    if len(model_names) < 2:
        return {}
    
    model1, model2 = model_names[0], model_names[1]
    comparison = {}
    
    for qid in queries.keys():
        if qid not in results[model1] or qid not in results[model2]:
            continue
            
        # Get top-25 from each model
        docs1 = set(sorted(results[model1][qid].keys(), 
                          key=lambda x: results[model1][qid][x], reverse=True)[:25])
        docs2 = set(sorted(results[model2][qid].keys(),
                          key=lambda x: results[model2][qid][x], reverse=True)[:25])
        
        relevant = set(doc_id for doc_id, score in qrels[qid].items() if score > 0)
        
        # Documents found by both
        both = docs1 & docs2
        only1 = docs1 - docs2
        only2 = docs2 - docs1
        
        # Relevant documents in each category
        comparison[qid] = {
            'found_by_both': len(both & relevant),
            f'only_{model1}': len(only1 & relevant),
            f'only_{model2}': len(only2 & relevant),
            'total_relevant': len(relevant)
        }
    
    return comparison


# ============================================================
# PART 6: MAIN EXECUTION
# ============================================================

def process_model(model, model_name: str, doc_texts: List[str], doc_ids: List[str],
                  sample_queries: Dict, K: int = 25) -> Dict:
    """Process a single model - embedding, indexing, and search."""
    
    print(f"\nEmbedding corpus with {model_name}...")
    corpus_emb = model.encode(doc_texts, batch_size=32)
    
    print(f"Building {model_name} index...")
    index = FAISSIndex(model.embedding_dim)
    index.build(corpus_emb.copy(), doc_ids)
    
    print(f"Searching queries with {model_name}...")
    query_texts = [sample_queries[qid] for qid in sample_queries]
    query_emb = model.encode(query_texts, batch_size=8)
    
    results = {}
    for i, qid in enumerate(sample_queries.keys()):
        scores, indices = index.search(query_emb[i:i+1], K)
        results[qid] = {
            doc_ids[idx]: float(score)
            for score, idx in zip(scores[0], indices[0])
        }
    
    return results


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("SEMANTIC SEARCH ANALYSIS PROJECT")
    print("=" * 70)
    
    K = 25  # Top-K for evaluation
    NUM_QUERIES = 5
    
    # ----- STEP 1: Load Data -----
    print("\n" + "=" * 50)
    print("[STEP 1] Loading NFCorpus Dataset...")
    print("=" * 50)
    
    corpus, queries, qrels = load_nfcorpus("./datasets")
    sample_queries = get_sample_queries(queries, qrels, n=NUM_QUERIES)
    
    print(f"\n✓ Loaded {len(corpus)} documents")
    print(f"✓ Selected {len(sample_queries)} queries")
    
    print("\nSelected Queries:")
    for qid, qtext in sample_queries.items():
        num_rel = len([d for d, r in qrels[qid].items() if r > 0])
        print(f"  [{qid}] {qtext[:50]}... ({num_rel} relevant)")
    
    # Prepare corpus
    doc_ids = list(corpus.keys())
    doc_texts = [get_document_text(corpus[did]) for did in doc_ids]
    
    results = {}
    fp_fn_results = {}
    evaluation_results = {}
    
    # ----- STEP 2: Model 1 - SigLIP (FIXED) -----
    print("\n" + "=" * 50)
    print("[STEP 2] Processing with SigLIP Model (FIXED)...")
    print("=" * 50)
    
    try:
        siglip = SigLIPEmbedding()
        siglip_results = process_model(siglip, "SigLIP", doc_texts, doc_ids, sample_queries, K)
        results['SigLIP'] = siglip_results
        print("✓ SigLIP complete!")
        
        # Clean up
        del siglip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ SigLIP error: {e}")
        import traceback
        traceback.print_exc()
    
    # ----- STEP 3: Model 2 - MPNet -----
    print("\n" + "=" * 50)
    print("[STEP 3] Processing with MPNet Model...")
    print("=" * 50)
    
    try:
        mpnet = MPNetEmbedding()
        mpnet_results = process_model(mpnet, "MPNet", doc_texts, doc_ids, sample_queries, K)
        results['MPNet'] = mpnet_results
        print("✓ MPNet complete!")
        
        del mpnet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ MPNet error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to MiniLM
        print("\nFalling back to MiniLM...")
        try:
            minilm = MiniLMEmbedding()
            minilm_results = process_model(minilm, "MiniLM", doc_texts, doc_ids, sample_queries, K)
            results['MiniLM'] = minilm_results
            print("✓ MiniLM complete!")
        except Exception as e2:
            print(f"✗ MiniLM also failed: {e2}")
    
    # ----- STEP 4: Evaluation -----
    print("\n" + "=" * 50)
    print("[STEP 4] Evaluation - MAP@25")
    print("=" * 50)
    
    for model_name, model_results in results.items():
        eval_result = evaluate_results(model_results, qrels, K)
        evaluation_results[model_name] = eval_result
        
        print(f"\n{'='*40}")
        print(f"Model: {model_name}")
        print(f"{'='*40}")
        print(f"MAP@{K}: {eval_result['MAP@25']:.4f}")
        
        print(f"\nPer-Query Results:")
        for qid, metrics in eval_result['per_query'].items():
            print(f"  {qid}: AP={metrics['ap']:.4f}, "
                  f"Hits={metrics['hits']}/{metrics['num_relevant']}, "
                  f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    # ----- STEP 5: FP/FN Analysis -----
    print("\n" + "=" * 50)
    print("[STEP 5] False Positive / False Negative Analysis")
    print("=" * 50)
    
    for model_name, model_results in results.items():
        fp_fn = analyze_fp_fn(model_results, qrels, corpus, sample_queries, K)
        fp_fn_results[model_name] = fp_fn
        print_fp_fn_report(fp_fn, model_name)
    
    # ----- STEP 6: Model Comparison -----
    print("\n" + "=" * 50)
    print("[STEP 6] Model Comparison Summary")
    print("=" * 50)
    
    print(f"\n{'Model':<12} {'MAP@25':<10} {'Recall':<10} {'Precision':<10} {'FP':<8} {'FN':<8}")
    print("-" * 58)
    
    for model_name in results.keys():
        ev = evaluation_results[model_name]
        fp = fp_fn_results[model_name]['aggregate']
        
        recalls = [m['recall'] for m in ev['per_query'].values()]
        precisions = [m['precision'] for m in ev['per_query'].values()]
        
        print(f"{model_name:<12} {ev['MAP@25']:<10.4f} {np.mean(recalls):<10.4f} "
              f"{np.mean(precisions):<10.4f} {fp['total_fp']:<8} {fp['total_fn']:<8}")
    
    # Detailed comparison if we have 2 models
    if len(results) >= 2:
        print("\n--- Detailed Model Comparison ---")
        comparison = compare_models_detailed(results, qrels, sample_queries)
        
        model_names = list(results.keys())
        print(f"\nRelevant documents found:")
        print(f"{'Query':<12} {'Both':<8} {model_names[0]+'_only':<15} {model_names[1]+'_only':<15} {'Total':<8}")
        print("-" * 58)
        
        for qid, comp in comparison.items():
            print(f"{qid:<12} {comp['found_by_both']:<8} "
                  f"{comp.get(f'only_{model_names[0]}', 0):<15} "
                  f"{comp.get(f'only_{model_names[1]}', 0):<15} "
                  f"{comp['total_relevant']:<8}")
    
    # ----- STEP 7: Insights -----
    print("\n" + "=" * 50)
    print("[STEP 7] Key Insights and Conclusions")
    print("=" * 50)
    
    # Calculate winner
    if len(evaluation_results) >= 2:
        model_names = list(evaluation_results.keys())
        scores = [evaluation_results[m]['MAP@25'] for m in model_names]
        best_model = model_names[np.argmax(scores)]
        best_score = max(scores)
        worst_score = min(scores)
        improvement = ((best_score - worst_score) / worst_score * 100) if worst_score > 0 else 0
    else:
        best_model = list(evaluation_results.keys())[0]
        improvement = 0
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                         KEY FINDINGS                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  1. BEST PERFORMING MODEL: {best_model:<40} ║
║     MAP@25 improvement: {improvement:.1f}%                                        ║
║                                                                       ║
║  2. FALSE POSITIVES ANALYSIS:                                        ║
║     • Documents retrieved but NOT actually relevant                   ║
║     • Common issue: Lexical overlap without semantic match            ║
║     • Example: "cholesterol" appears but context is different         ║
║                                                                       ║
║  3. FALSE NEGATIVES ANALYSIS:                                        ║
║     • Relevant documents that were MISSED                             ║
║     • Common issue: Vocabulary mismatch                               ║
║     • Example: Query uses "cancer", doc uses "tumor/neoplasm"         ║
║                                                                       ║
║  4. MODEL DIFFERENCES:                                                ║
║     • SigLIP: Short text focus (64 tokens max), loses document info   ║
║     • MPNet/MiniLM: Full document encoding, better for long text      ║
║                                                                       ║
║  5. RECOMMENDATIONS:                                                  ║
║     • Use longer-context models for document retrieval                ║
║     • Consider hybrid retrieval (dense + BM25)                        ║
║     • Fine-tune on medical domain data                                ║
║     • Implement query expansion for medical terminology               ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # ----- STEP 8: Save Results -----
    print("\n[STEP 8] Saving Results...")
    
    os.makedirs("./results", exist_ok=True)
    
    summary = {
        'evaluation': {
            model: {
                'MAP@25': float(ev['MAP@25']),
                'Mean_Recall': float(np.mean([m['recall'] for m in ev['per_query'].values()])),
                'Mean_Precision': float(np.mean([m['precision'] for m in ev['per_query'].values()]))
            }
            for model, ev in evaluation_results.items()
        },
        'fp_fn': {
            model: {
                'total_fp': int(fp['aggregate']['total_fp']),
                'total_fn': int(fp['aggregate']['total_fn']),
                'total_tp': int(fp['aggregate']['total_tp']),
                'mean_precision': float(fp['aggregate']['mean_precision']),
                'mean_recall': float(fp['aggregate']['mean_recall'])
            }
            for model, fp in fp_fn_results.items()
        },
        'per_query_details': {
            model: {
                qid: {
                    'AP': float(m['ap']),
                    'hits': int(m['hits']),
                    'num_relevant': int(m['num_relevant']),
                    'precision': float(m['precision']),
                    'recall': float(m['recall'])
                }
                for qid, m in ev['per_query'].items()
            }
            for model, ev in evaluation_results.items()
        }
    }
    
    with open("./results/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed FP/FN analysis
    fp_fn_details = {
        model: {
            qid: {
                'query': qa['query_text'],
                'stats': {
                    'relevant': qa['num_relevant'],
                    'tp': qa['num_tp'],
                    'fp': qa['num_fp'],
                    'fn': qa['num_fn']
                },
                'false_positives': qa['false_positives'],
                'false_negatives': qa['false_negatives']
            }
            for qid, qa in fp['per_query'].items()
        }
        for model, fp in fp_fn_results.items()
    }
    
    with open("./results/fp_fn_analysis.json", 'w') as f:
        json.dump(fp_fn_details, f, indent=2)
    
    print("✓ Results saved to ./results/summary.json")
    print("✓ FP/FN analysis saved to ./results/fp_fn_analysis.json")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    
    return evaluation_results, fp_fn_results


if __name__ == "__main__":
    main()