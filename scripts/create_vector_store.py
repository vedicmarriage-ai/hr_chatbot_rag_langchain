import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
import json
from typing import Dict, Any

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from app.services.embedding_service import EmbeddingService, EmbeddingType
from app.core.config import get_settings
from app.services.document_service import DocumentService

settings = get_settings()

def create_vector_store(
    docs_dir: str,
    persist_dir: str,
    embedding_type: str,
    model_key: str = None
) -> Dict[str, Any]:
    """Create a vector store with the specified embedding model"""
    print("\n" + "=" * 80)
    print(f"Creating vector store for {embedding_type} - {model_key or 'default'}")
    print(f"Starting at {datetime.now()}")
    print(f"Documents directory: {docs_dir}")
    print(f"Persist directory: {persist_dir}")
    
    # Create embedding service
    embedding_service = EmbeddingService(
        embedding_type=EmbeddingType(embedding_type),
        model_key=model_key
    )
    
    # Get embedding dimension
    test_text = "This is a test document to check embedding dimensions."
    embedding = embedding_service.get_embedding(test_text)
    embedding_dim = len(embedding)
    
    print(f"\nEmbedding dimension: {embedding_dim}")
    
    # Create document service
    doc_service = DocumentService()
    
    # Load and process documents
    print("\nLoading documents...")
    docs = doc_service.load_documents(docs_dir)
    print(f"Loaded {len(docs)} documents")
    
    print("\nSplitting documents...")
    chunks = doc_service.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    
    # Create vector store
    print("\nCreating vector store...")
    vectorstore = doc_service.create_vector_store(
        chunks,
        persist_dir,
        embedding_service.embeddings
    )
    
    # Save metadata
    metadata = {
        "embedding_type": embedding_type,
        "model_key": model_key,
        "embedding_dimension": embedding_dim,
        "created_at": datetime.now().isoformat(),
        "num_chunks": len(chunks),
        "model_info": embedding_service.get_model_info()
    }
    
    metadata_path = os.path.join(persist_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Vector store created and saved at: {persist_dir}")
    print(f"Vector store contains {len(chunks)} chunks")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Completed at {datetime.now()}")
    print("=" * 80 + "\n")
    
    return metadata

def create_all_vector_stores(docs_dir: str):
    """Create vector stores for all available embedding models"""
    print("Creating vector stores for all available embedding models...")
    
    # Get total number of models
    total_models = 1 + len(settings.AVAILABLE_EMBEDDING_MODELS)  # 1 for Ollama
    print(f"Total models to process: {total_models}\n")
    
    # Create Ollama vector store
    print("Processing Ollama embeddings...")
    ollama_dir = os.path.join(settings.CHROMA_PERSIST_DIRECTORY, "ollama")
    create_vector_store(docs_dir, ollama_dir, "ollama")
    
    # Create HuggingFace vector stores
    for model_key in settings.AVAILABLE_EMBEDDING_MODELS.keys():
        print(f"\nProcessing HuggingFace {model_key} embeddings...")
        model_config = settings.AVAILABLE_EMBEDDING_MODELS[model_key]
        print(f"Model details: {model_config['name']}")
        print(f"Description: {model_config['description']}")
        print(f"Size: {model_config['size']}")
        print(f"Quality: {model_config['quality']}")
        print(f"Speed: {model_config['speed']}")
        print(f"Dimensions: {model_config['dimensions']}\n")
        
        hf_dir = os.path.join(settings.CHROMA_PERSIST_DIRECTORY, f"hf_{model_key}")
        create_vector_store(docs_dir, hf_dir, "huggingface", model_key)
    
    print("\nAll vector stores have been created successfully!")

def main():
    parser = argparse.ArgumentParser(description="Create vector stores for different embedding models")
    parser.add_argument(
        "--docs-dir",
        default=str(settings.DOCS_DIR),
        help="Directory containing the documents to process"
    )
    parser.add_argument(
        "--embedding-type",
        choices=["ollama", "huggingface", "all"],
        default="all",
        help="Type of embedding to use"
    )
    parser.add_argument(
        "--model-key",
        help="Specific model key to use (for HuggingFace embeddings)"
    )
    
    args = parser.parse_args()
    
    # Verify documents directory exists
    if not os.path.exists(args.docs_dir):
        print(f"Error: Documents directory not found at {args.docs_dir}")
        sys.exit(1)
    
    if args.embedding_type == "all":
        create_all_vector_stores(args.docs_dir)
    else:
        if args.embedding_type == "huggingface" and not args.model_key:
            print("Error: model-key is required for HuggingFace embeddings")
            sys.exit(1)
        
        persist_dir = os.path.join(
            settings.CHROMA_PERSIST_DIRECTORY,
            "ollama" if args.embedding_type == "ollama" else f"hf_{args.model_key}"
        )
        create_vector_store(args.docs_dir, persist_dir, args.embedding_type, args.model_key)

if __name__ == "__main__":
    main() 