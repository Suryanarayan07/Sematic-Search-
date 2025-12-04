"""
Module for loading and preprocessing the NFCorpus dataset from BEIR.
"""

import os
from typing import Dict, Tuple, List
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFCorpusLoader:
    """
    Loader class for NFCorpus dataset from BEIR benchmark.
    """
    
    def __init__(self, data_dir: str = "./datasets"):
        self.data_dir = data_dir
        self.dataset_name = "nfcorpus"
        self.dataset_path = None
        
    def download_dataset(self) -> str:
        """Download NFCorpus dataset if not already present."""
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        
        logger.info(f"Downloading {self.dataset_name} dataset...")
        self.dataset_path = util.download_and_unzip(url, self.data_dir)
        logger.info(f"Dataset downloaded to: {self.dataset_path}")
        
        return self.dataset_path
    
    def load_data(self, split: str = "test") -> Tuple[Dict, Dict, Dict]:
        """
        Load corpus, queries, and relevance judgments.
        """
        if self.dataset_path is None:
            self.download_dataset()
            
        corpus, queries, qrels = GenericDataLoader(
            data_folder=self.dataset_path
        ).load(split=split)
        
        logger.info(f"Loaded {len(corpus)} documents")
        logger.info(f"Loaded {len(queries)} queries")
        logger.info(f"Loaded {len(qrels)} query relevance judgments")
        
        return corpus, queries, qrels
    
    def get_sample_queries(self, queries: Dict, qrels: Dict, n: int = 5) -> Dict:
        """
        Select n sample queries that have relevance judgments.
        """
        # Select queries that have at least some relevant documents
        valid_query_ids = [qid for qid in qrels.keys() if len(qrels[qid]) > 0]
        
        # Take first n queries
        selected_ids = valid_query_ids[:n]
        
        sampled_queries = {qid: queries[qid] for qid in selected_ids}
        
        logger.info(f"Selected {len(sampled_queries)} queries for evaluation")
        for qid, query in sampled_queries.items():
            num_relevant = len([d for d, r in qrels[qid].items() if r > 0])
            logger.info(f"  Query {qid}: '{query[:50]}...' ({num_relevant} relevant docs)")
            
        return sampled_queries


# Test the loader
if __name__ == "__main__":
    loader = NFCorpusLoader()
    corpus, queries, qrels = loader.load_data(split="test")
    sample_queries = loader.get_sample_queries(queries, qrels, n=5)
    
    print("\n=== Dataset Statistics ===")
    print(f"Total documents: {len(corpus)}")
    print(f"Total queries: {len(queries)}")
    print(f"Sample queries: {len(sample_queries)}")