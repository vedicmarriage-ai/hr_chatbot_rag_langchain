#!/usr/bin/env python3
"""
Test script to check RAG functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.rag_service import RAGService
from app.services.embedding_service import EmbeddingType

def test_rag():
    print("🔍 Testing RAG functionality...")

    try:
        # Initialize RAG service
        rag_service = RAGService(
            model_name="mistral:7b-instruct-v0.3-q2_K",  # Using a better model for generation
            embedding_type=EmbeddingType.HUGGINGFACE,
            embedding_model="bge-base"
        )

        # Test 1: Question that should be in documents
        test_question_1 = "Is the vedic marriage website free or do I need to pay for registration and basic features?"
        print(f"\n📝 Test Question 1 (Should be in documents): {test_question_1}")

        # Process the question
        response_1, session_id_1 = rag_service.process_message(test_question_1, session_id=None)
        print(f"\n💡 RAG Response 1: {response_1}")
        print(f"📚 Sources: {[doc.metadata.get('source', 'Unknown') for doc in rag_service.chain.retriever.get_relevant_documents(test_question_1)]}")

        # Test 2: Question that's NOT in documents (to test generation)
        test_question_2 = "What is the weather like today in New York?"
        print(f"\n📝 Test Question 2 (NOT in documents - should test generation): {test_question_2}")

        # Process the question
        response_2, session_id_2 = rag_service.process_message(test_question_2, session_id=None)
        print(f"\n💡 RAG Response 2: {response_2}")
        print(f"📚 Sources: {[doc.metadata.get('source', 'Unknown') for doc in rag_service.chain.retriever.get_relevant_documents(test_question_2)]}")

        # Test 3: Check if LLM is actually generating
        print(f"\n🔍 Testing LLM generation capability...")
        llm_response = rag_service.llm_service.generate_response("What is 2+2? Answer in one word.")
        print(f"🤖 Direct LLM Response: {llm_response}")

        # Analysis
        print(f"\n📊 ANALYSIS:")
        print(f"1. RAG Response 1 (in documents): {len(response_1)} characters")
        print(f"2. RAG Response 2 (not in documents): {len(response_2)} characters")
        print(f"3. Direct LLM Response: {len(llm_response)} characters")
        
        if len(response_2) > 50:  # If response is substantial, it's likely generating
            print(f"✅ RAG is doing GENERATION - Response 2 is substantial ({len(response_2)} chars)")
        else:
            print(f"⚠️ RAG might be doing only RETRIEVAL - Response 2 is short ({len(response_2)} chars)")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    test_rag()
