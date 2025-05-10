import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Optional

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from app.services.embedding_service import EmbeddingService, EmbeddingType
from app.core.config import get_settings
from langchain_chroma import Chroma

settings = get_settings()

class VectorStoreBenchmark:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Test queries for benchmarking
        self.test_queries = [
            "What is the company's policy on remote work?",
            "How many vacation days do employees get?",
            "What is the process for requesting time off?",
            "What are the working hours?",
            "How does the performance review process work?",
            "What is the dress code policy?",
            "How do I report a workplace issue?",
            "What are the benefits offered to employees?",
            "How do I request a salary review?",
            "What is the process for taking sick leave?"
        ]
    
    def get_model_key_from_dir(self, persist_dir: str) -> str:
        """Extract model key from the persist directory name"""
        dir_name = os.path.basename(persist_dir)
        if dir_name == "ollama":
            return None
        return dir_name.replace("hf_", "")
    
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
        """Load a vector store with the correct embedding model"""
        model_key = self.get_model_key_from_dir(persist_dir)
        embedding_type = "huggingface" if model_key else "ollama"
        
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
    
    def benchmark_vector_store(self, vectorstore: Chroma, model_name: str) -> List[Dict]:
        """Benchmark a vector store with test queries"""
        results = []
        for query in self.test_queries:
            start_time = time.time()
            docs = vectorstore.similarity_search(query, k=settings.TOP_K_RESULTS)
            end_time = time.time()
            
            relevance_score = sum(1 for doc in docs if any(word.lower() in doc.page_content.lower() 
                                                         for word in query.split()))
            relevance_score = relevance_score / settings.TOP_K_RESULTS
            
            results.append({
                'query': query,
                'response_time': end_time - start_time,
                'relevance_score': relevance_score,
                'model': model_name
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
        
        # Relevance Score Box Plot
        sns.boxplot(data=df, x='model', y='relevance_score', ax=ax2)
        ax2.set_title('Relevance Score Distribution by Model')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Relevance Score')
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
            'relevance_score': ['mean', 'std', 'min', 'max']
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
                'relevance_score': {
                    'mean': metrics.loc[model, ('relevance_score', 'mean')],
                    'std': metrics.loc[model, ('relevance_score', 'std')],
                    'min': metrics.loc[model, ('relevance_score', 'min')],
                    'max': metrics.loc[model, ('relevance_score', 'max')]
                }
            }
        
        # Find best model based on relevance score and response time
        best_model = metrics[('relevance_score', 'mean')].idxmax()
        best_embedding_type = "huggingface" if best_model.startswith("hf_") else "ollama"
        best_embedding_model = best_model.split("_")[1] if best_model.startswith("hf_") else None
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to CSV
        df.to_csv(os.path.join(self.output_dir, f'vector_store_benchmark_{timestamp}.csv'), index=False)
        
        # Save metrics to JSON
        with open(os.path.join(self.output_dir, f'model_metrics_{timestamp}.json'), 'w') as f:
            json.dump({
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
    
    args = parser.parse_args()
    
    # Create and run benchmark
    benchmark = VectorStoreBenchmark(args.output_dir)
    benchmark.run_benchmark()

if __name__ == "__main__":
    main() 