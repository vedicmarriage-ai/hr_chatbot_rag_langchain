#!/usr/bin/env python3
"""
Test script to check vector store contents
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.embedding_service import EmbeddingService, EmbeddingType
from app.core.config import get_settings
from langchain_chroma import Chroma

def test_vector_store():
    print("🔍 Testing vector store contents...")
    
    try:
        settings = get_settings()
        
        # Initialize embedding service
        embedding_service = EmbeddingService(
            embedding_type=EmbeddingType.HUGGINGFACE,
            model_key="bge-base"
        )
        
        # Load vector store
        vectorstore = Chroma(
            persist_directory=str(settings.CHROMA_PERSIST_DIRECTORY),
            embedding_function=embedding_service.embeddings
        )
        
        # Get all documents
        print(f"\n📊 Vector store info:")
        print(f"   Persist directory: {settings.CHROMA_PERSIST_DIRECTORY}")
        
        # Try to get collection info
        collection = vectorstore._collection
        if collection:
            count = collection.count()
            print(f"   Total documents: {count}")
            
            # Get some sample documents
            if count > 0:
                print(f"\n📄 Sample documents:")
                results = collection.get(limit=5)
                if results and 'metadatas' in results:
                    for i, metadata in enumerate(results['metadatas'][:3]):
                        source = metadata.get('source', 'Unknown') if metadata else 'Unknown'
                        print(f"   {i+1}. {source}")
                        
                        # Get a snippet of the document
                        if 'documents' in results and results['documents']:
                            doc = results['documents'][i] if i < len(results['documents']) else ''
                            snippet = doc[:200] + "..." if len(doc) > 200 else doc
                            print(f"      Snippet: {snippet}")
                            print()
        
        # Test similarity search
        print(f"\n🔍 Testing similarity search for 'vedic marriage website free paid':")
        query = "vedic marriage website free paid"
        docs = vectorstore.similarity_search(query, k=3)
        
        if docs:
            print(f"   Found {len(docs)} relevant documents:")
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                print(f"   {i+1}. {source}")
                print(f"      Content: {content}")
                print()
        else:
            print("   ⚠️ No relevant documents found")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_store()
