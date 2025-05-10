import sys
from pathlib import Path
import os

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from app.services.embedding_service import EmbeddingService, EmbeddingType
from langchain_community.vectorstores import Chroma
from app.core.config import get_settings

def test_vector_store():
    settings = get_settings()
    
    # Initialize embedding service with the same model used for creation
    embedding_service = EmbeddingService(embedding_type=EmbeddingType.OLLAMA)
    
    # Load the existing vector store
    vectorstore = Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        embedding_function=embedding_service.embeddings
    )
    
    # Test queries
    test_queries = [
        "What is the refund policy?",
        "How can I cancel my subscription?",
        "What are the terms and conditions?",
        "What is VedicMarriage?",
        "How do I create an account?",
        "What are the privacy policies?"
    ]
    
    print("\nTesting Vector Store with Similarity Search:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        # Perform similarity search with scores
        results = vectorstore.similarity_search_with_score(query, k=3)
        
        # Print results
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nResult {i} (Similarity Score: {score:.4f}):")
            print(f"Content: {doc.page_content[:300]}...")  # Print first 300 chars
            
            # Get filename from source path
            source = doc.metadata.get('source', 'Unknown')
            filename = os.path.basename(source) if source != 'Unknown' else 'Unknown'
            
            print(f"Source File: {filename}")
            if 'page' in doc.metadata:
                print(f"Page: {doc.metadata['page']}")
            print("-" * 50)

if __name__ == "__main__":
    test_vector_store() 