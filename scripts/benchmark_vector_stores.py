import os
import re
import sys
from pathlib import Path
import argparse
from datetime import datetime
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Optional, Tuple, Any

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from app.services.embedding_service import EmbeddingService, EmbeddingType
from app.core.config import get_settings
from langchain_chroma import Chroma

settings = get_settings()

class VectorStoreBenchmark:
    def __init__(self, output_dir: str, dataset_path: str, top_k: int):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.dataset_path = dataset_path
        self.top_k = top_k
        self.qrels_by_query: Dict[str, List[Dict[str, Any]]] = {}
        self.query_texts: Dict[str, str] = {}
        self._load_dataset()

    def _load_dataset(self):
        """Load qrels-style golden dataset from CSV."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. "
                f"Create it from the template in data/benchmark_results/."
            )

        df = pd.read_csv(self.dataset_path)
        required_cols = {
            "query_id",
            "query",
            "gold_source",
            "gold_keywords",
            "relevance_grade",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Dataset missing required columns: {sorted(missing)}"
            )

        # Keep only usable rows
        df = df.fillna("")
        for row in df.to_dict(orient="records"):
            query_id = str(row["query_id"]).strip()
            query = str(row["query"]).strip()
            gold_source = str(row["gold_source"]).strip()
            if not query_id or not query or not gold_source:
                continue

            try:
                grade = int(row["relevance_grade"])
            except (TypeError, ValueError):
                grade = 1
            grade = max(0, min(3, grade))

            keywords = [
                k.strip().lower()
                for k in str(row["gold_keywords"]).split("|")
                if k.strip()
            ]
            section_regex = str(row.get("gold_section_regex", "")).strip()
            gold_faq_id = str(row.get("gold_faq_id", "")).strip()
            gold_legal_id = str(row.get("gold_legal_id", "")).strip()

            qrel = {
                "gold_source": gold_source.lower(),
                "gold_keywords": keywords,
                "gold_section_regex": section_regex,
                "gold_faq_id": gold_faq_id,
                "gold_legal_id": gold_legal_id,
                "relevance_grade": grade,
            }
            self.qrels_by_query.setdefault(query_id, []).append(qrel)
            self.query_texts[query_id] = query

        if not self.qrels_by_query:
            raise ValueError("Dataset loaded, but no valid qrels found.")
    
    def get_model_key_from_dir(self, persist_dir: str) -> Optional[str]:
        """Best-effort fallback model key parser from legacy directory names."""
        dir_name = os.path.basename(persist_dir)
        if dir_name.startswith("hf_"):
            return dir_name.replace("hf_", "", 1)
        if dir_name.startswith("ollama_"):
            return dir_name.replace("ollama_", "", 1)
        if dir_name == "ollama":
            return None
        return None
    
    def load_metadata(self, persist_dir: str) -> Optional[Dict]:
        """Load metadata for a vector store"""
        metadata_path = os.path.join(persist_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            print(f"Warning: No metadata found at {metadata_path}")
            return None
        
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    def verify_embedding_dimension(self, persist_dir: str, embedding_service: EmbeddingService) -> bool:
        """Verify that the embedding dimension matches the stored metadata"""
        metadata = self.load_metadata(persist_dir)
        if not metadata:
            return False
        
        # Get current embedding dimension
        test_text = "This is a test document to check embedding dimensions."
        current_dim = len(embedding_service.get_embedding(test_text))
        stored_dim = metadata.get("embedding_dimension")
        
        if current_dim != stored_dim:
            print(f"Error: Embedding dimension mismatch!")
            print(f"Expected dimension: {stored_dim}")
            print(f"Current dimension: {current_dim}")
            return False
        
        return True
    
    def load_vector_store(self, persist_dir: str) -> Chroma:
        """Load a vector store with the correct embedding model."""
        metadata = self.load_metadata(persist_dir)
        if metadata:
            embedding_type = metadata.get("embedding_type", "ollama")
            model_key = metadata.get("model_key")
        else:
            # Fallback for older stores missing metadata
            model_key = self.get_model_key_from_dir(persist_dir)
            embedding_type = "huggingface" if (model_key and persist_dir.endswith(model_key) and "hf_" in os.path.basename(persist_dir)) else "ollama"
        
        embedding_service = EmbeddingService(
            embedding_type=EmbeddingType(embedding_type),
            model_key=model_key
        )
        
        # Verify embedding dimension
        if not self.verify_embedding_dimension(persist_dir, embedding_service):
            raise ValueError(f"Embedding dimension mismatch for {persist_dir}")
        
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_service.embeddings
        )
    
    def _doc_matches_qrel(self, doc, qrel: Dict[str, Any]) -> bool:
        source = doc.metadata.get("source", "").lower()
        if qrel["gold_source"] not in source:
            return False

        gold_faq_id = str(qrel.get("gold_faq_id", "")).strip()
        gold_legal_id = str(qrel.get("gold_legal_id", "")).strip()
        if gold_faq_id:
            if doc.metadata.get("faq_id") != gold_faq_id:
                return False
        elif gold_legal_id:
            if doc.metadata.get("legal_id") != gold_legal_id:
                return False
        elif qrel["gold_section_regex"]:
            section = str(doc.metadata.get("section", ""))
            heading = str(doc.metadata.get("legal_heading", ""))
            target = f"{section}\n{heading}\n{doc.page_content[:800]}"
            try:
                if not re.search(qrel["gold_section_regex"], target, re.IGNORECASE):
                    return False
            except re.error:
                return False

        if qrel["gold_keywords"]:
            text = doc.page_content.lower()
            keywords = qrel["gold_keywords"]
            if gold_faq_id or gold_legal_id:
                # id already pins the chunk; require most keywords, not all
                hits = sum(1 for k in keywords if k in text)
                if hits < min(2, len(keywords)):
                    return False
            elif not all(k in text for k in keywords):
                return False

        return True

    def _evaluate_query(self, docs: List[Any], qrels: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute Precision@k, Recall@k, MRR, MAP, nDCG@k for one query."""
        if not qrels:
            return {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "map": 0.0,
                "ndcg_at_k": 0.0,
            }

        matched_qrel_indices = set()
        per_rank_rel_binary: List[int] = []
        per_rank_rel_grade: List[int] = []

        for doc in docs:
            best_match_idx = None
            best_grade = -1
            for idx, qrel in enumerate(qrels):
                if idx in matched_qrel_indices:
                    continue
                if self._doc_matches_qrel(doc, qrel):
                    if qrel["relevance_grade"] > best_grade:
                        best_grade = qrel["relevance_grade"]
                        best_match_idx = idx

            if best_match_idx is not None:
                matched_qrel_indices.add(best_match_idx)
                per_rank_rel_binary.append(1)
                per_rank_rel_grade.append(best_grade)
            else:
                per_rank_rel_binary.append(0)
                per_rank_rel_grade.append(0)

        k = len(docs)
        relevant_retrieved = sum(per_rank_rel_binary)
        total_relevant = len(qrels)

        precision_at_k = relevant_retrieved / k if k else 0.0
        recall_at_k = relevant_retrieved / total_relevant if total_relevant else 0.0

        mrr = 0.0
        for rank, rel in enumerate(per_rank_rel_binary, start=1):
            if rel == 1:
                mrr = 1.0 / rank
                break

        ap_acc = 0.0
        hit_count = 0
        for rank, rel in enumerate(per_rank_rel_binary, start=1):
            if rel == 1:
                hit_count += 1
                ap_acc += hit_count / rank
        map_score = ap_acc / total_relevant if total_relevant else 0.0

        dcg = 0.0
        for rank, rel_grade in enumerate(per_rank_rel_grade, start=1):
            if rel_grade > 0:
                dcg += (2**rel_grade - 1) / math.log2(rank + 1)

        ideal_grades = sorted(
            (q["relevance_grade"] for q in qrels), reverse=True
        )[:k]
        idcg = 0.0
        for rank, rel_grade in enumerate(ideal_grades, start=1):
            if rel_grade > 0:
                idcg += (2**rel_grade - 1) / math.log2(rank + 1)
        ndcg_at_k = (dcg / idcg) if idcg > 0 else 0.0

        return {
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "mrr": mrr,
            "map": map_score,
            "ndcg_at_k": ndcg_at_k,
        }

    def benchmark_vector_store(self, vectorstore: Chroma, model_name: str) -> List[Dict]:
        """Benchmark retrieval metrics against a qrels-style golden dataset."""
        results = []
        for query_id, query in self.query_texts.items():
            start_time = time.time()
            docs = vectorstore.similarity_search(query, k=self.top_k)
            end_time = time.time()

            qrels = self.qrels_by_query.get(query_id, [])
            retrieval_metrics = self._evaluate_query(docs, qrels)

            results.append({
                "query_id": query_id,
                'query': query,
                'response_time': end_time - start_time,
                'precision_at_k': retrieval_metrics["precision_at_k"],
                'recall_at_k': retrieval_metrics["recall_at_k"],
                'mrr': retrieval_metrics["mrr"],
                'map': retrieval_metrics["map"],
                'ndcg_at_k': retrieval_metrics["ndcg_at_k"],
                'model': model_name,
            })
        return results
    
    def plot_performance_metrics(self, df: pd.DataFrame):
        """Create visualizations of the performance metrics"""
        sns.set_theme()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Response Time Box Plot
        sns.boxplot(data=df, x='model', y='response_time', ax=ax1)
        ax1.set_title('Response Time Distribution by Model')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Response Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # nDCG@k Box Plot
        sns.boxplot(data=df, x='model', y='ndcg_at_k', ax=ax2)
        ax2.set_title('nDCG@k Distribution by Model')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('nDCG@k')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'vector_store_performance.png'))
        plt.close()
    
    def run_benchmark(self):
        """Run the complete benchmarking process"""
        all_results = []
        model_metrics = {}
        
        # Get all vector store directories
        vector_store_dirs = [d for d in os.listdir(settings.CHROMA_PERSIST_DIRECTORY) 
                           if os.path.isdir(os.path.join(settings.CHROMA_PERSIST_DIRECTORY, d))]
        
        for dir_name in vector_store_dirs:
            print(f"\nBenchmarking {dir_name} embeddings...")
            persist_dir = os.path.join(settings.CHROMA_PERSIST_DIRECTORY, dir_name)

            metadata_path = os.path.join(persist_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                print(f"Skipping {dir_name}: no metadata.json (likely legacy/partial index)")
                continue
            
            try:
                vectorstore = self.load_vector_store(persist_dir)
                results = self.benchmark_vector_store(vectorstore, dir_name)
                all_results.extend(results)
            except Exception as e:
                print(f"Error benchmarking {dir_name}: {str(e)}")
                continue
        
        if not all_results:
            print("Error: No vector stores found to benchmark")
            return
        
        # Convert results to DataFrame
        df = pd.DataFrame(all_results)
        
        # Calculate performance metrics
        metrics = df.groupby('model').agg({
            'response_time': ['mean', 'std', 'min', 'max'],
            'precision_at_k': ['mean', 'std', 'min', 'max'],
            'recall_at_k': ['mean', 'std', 'min', 'max'],
            'mrr': ['mean', 'std', 'min', 'max'],
            'map': ['mean', 'std', 'min', 'max'],
            'ndcg_at_k': ['mean', 'std', 'min', 'max'],
        }).round(4)
        
        # Save metrics for each model
        for model in metrics.index:
            model_metrics[model] = {
                'response_time': {
                    'mean': metrics.loc[model, ('response_time', 'mean')],
                    'std': metrics.loc[model, ('response_time', 'std')],
                    'min': metrics.loc[model, ('response_time', 'min')],
                    'max': metrics.loc[model, ('response_time', 'max')]
                },
                'precision_at_k': {
                    'mean': metrics.loc[model, ('precision_at_k', 'mean')],
                    'std': metrics.loc[model, ('precision_at_k', 'std')],
                    'min': metrics.loc[model, ('precision_at_k', 'min')],
                    'max': metrics.loc[model, ('precision_at_k', 'max')]
                },
                'recall_at_k': {
                    'mean': metrics.loc[model, ('recall_at_k', 'mean')],
                    'std': metrics.loc[model, ('recall_at_k', 'std')],
                    'min': metrics.loc[model, ('recall_at_k', 'min')],
                    'max': metrics.loc[model, ('recall_at_k', 'max')]
                },
                'mrr': {
                    'mean': metrics.loc[model, ('mrr', 'mean')],
                    'std': metrics.loc[model, ('mrr', 'std')],
                    'min': metrics.loc[model, ('mrr', 'min')],
                    'max': metrics.loc[model, ('mrr', 'max')]
                },
                'map': {
                    'mean': metrics.loc[model, ('map', 'mean')],
                    'std': metrics.loc[model, ('map', 'std')],
                    'min': metrics.loc[model, ('map', 'min')],
                    'max': metrics.loc[model, ('map', 'max')]
                },
                'ndcg_at_k': {
                    'mean': metrics.loc[model, ('ndcg_at_k', 'mean')],
                    'std': metrics.loc[model, ('ndcg_at_k', 'std')],
                    'min': metrics.loc[model, ('ndcg_at_k', 'min')],
                    'max': metrics.loc[model, ('ndcg_at_k', 'max')]
                },
            }
        
        # Find best model based on nDCG@k (ranking quality)
        best_model = metrics[('ndcg_at_k', 'mean')].idxmax()
        best_embedding_type = "huggingface" if best_model.startswith("hf_") else "ollama"
        best_embedding_model = best_model.split("_")[1] if best_model.startswith("hf_") else None
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to CSV
        df.to_csv(os.path.join(self.output_dir, f'vector_store_benchmark_{timestamp}.csv'), index=False)
        
        # Save metrics to JSON
        with open(os.path.join(self.output_dir, f'model_metrics_{timestamp}.json'), 'w') as f:
            json.dump({
                'benchmark_config': {
                    'dataset_path': self.dataset_path,
                    'top_k': self.top_k,
                    'query_count': len(self.query_texts),
                },
                'model_metrics': model_metrics,
                'best_model': best_model,
                'best_embedding_type': best_embedding_type,
                'best_embedding_model': best_embedding_model
            }, f, indent=2)
        
        # Create visualizations
        self.plot_performance_metrics(df)
        
        print("\nBenchmark Results:")
        print("=" * 50)
        print(f"Best performing model: {best_model}")
        print(f"Best embedding type: {best_embedding_type}")
        print(f"Best embedding model: {best_embedding_model}")
        print("\nDetailed metrics:")
        print(metrics)
        print(f"\nResults saved to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark vector stores")
    parser.add_argument(
        "--output-dir",
        default=str(settings.DATA_DIR / "benchmark_results"),
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--dataset-path",
        default=str(settings.DATA_DIR / "benchmark_results" / "hr_retrieval_golden_template.csv"),
        help="Path to qrels-style benchmark CSV dataset"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.TOP_K_RESULTS,
        help="Top-k retrieval depth used for IR metrics"
    )
    
    args = parser.parse_args()
    
    # Create and run benchmark
    benchmark = VectorStoreBenchmark(args.output_dir, args.dataset_path, args.top_k)
    benchmark.run_benchmark()

if __name__ == "__main__":
    main() 