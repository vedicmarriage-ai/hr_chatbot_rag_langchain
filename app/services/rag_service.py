from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from app.core.config import get_settings
from app.core.rag_stacks import is_valid_combination
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService, EmbeddingType
from app.services.document_service import DocumentService
from app.services.reranker_service import rerank_documents
from typing import Any, List
import uuid
import os

settings = get_settings()


class RerankRetriever(BaseRetriever):
    """Fetch a wide candidate set, rerank with a cross-encoder, return top-k diverse chunks."""

    vectorstore: Any
    k: int = 5
    fetch_k: int = 15
    use_reranker: bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:
        docs = self.vectorstore.similarity_search(query, k=self.fetch_k)
        if not self.use_reranker:
            return docs[: self.k]
        return rerank_documents(query, docs, self.k)


class RAGService:
    def __init__(
        self,
        model_name: str = None,
        embedding_type: EmbeddingType = EmbeddingType.OLLAMA,
        embedding_model: str = None,
    ):
        if embedding_model is None:
            embedding_model = (
                settings.DEFAULT_OLLAMA_EMBEDDING_MODEL_KEY
                if embedding_type == EmbeddingType.OLLAMA
                else settings.DEFAULT_EMBEDDING_MODEL
            )

        llm_name = model_name or settings.OLLAMA_MODEL
        if not is_valid_combination(
            llm_name,
            embedding_type.value,
            embedding_model,
        ):
            raise ValueError(
                f"Invalid RAG stack: llm={llm_name}, "
                f"embedding_type={embedding_type.value}, "
                f"embedding_model={embedding_model}. "
                f"Choose a combination from app.core.rag_stacks.RAG_STACKS."
            )

        self.llm_service = LLMService(model_name=model_name)
        self.embedding_service = EmbeddingService(
            embedding_type=embedding_type,
            model_key=embedding_model
        )
        self.document_service = DocumentService()
        
        vector_store_subdir = EmbeddingService.get_vector_store_subdir(
            embedding_type, embedding_model
        )
        vector_store_path = settings.CHROMA_PERSIST_DIRECTORY / vector_store_subdir
        
        if not os.path.exists(vector_store_path):
            raise ValueError(
                f"Vector store not found at {vector_store_path}. "
                f"Please run create_vector_store.py first for {embedding_type} - {embedding_model}."
            )
        
        print(f"🔍 Loading vector store from: {vector_store_path}")
        
        self.vectorstore = Chroma(
            persist_directory=str(vector_store_path),
            embedding_function=self.embedding_service.embeddings
        )
        
        self.qa_template = """You are an HR assistant for Vedic Marriage. Answer using only the context below. Be concise and direct.

If the answer is not in the context, say: "I don't have that information in our HR documents."

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""
        
        self.QA_PROMPT = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        self.retriever = RerankRetriever(
            vectorstore=self.vectorstore,
            k=settings.TOP_K_RESULTS,
            fetch_k=settings.RERANK_CANDIDATES,
            use_reranker=settings.RERANK_ENABLED,
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm_service.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.QA_PROMPT},
            return_source_documents=True,
            verbose=True
        )
    
    def process_message(self, message: str, session_id: str = None) -> tuple[str, str]:
        if not session_id:
            session_id = str(uuid.uuid4())
            
        response = self.chain({"question": message})
        
        answer = response["answer"]
        source_docs = response.get("source_documents", [])
        
        if source_docs:
            seen_sources: set[str] = set()
            source_lines: List[str] = []
            for doc in source_docs:
                src = doc.metadata.get("source", "Unknown")
                if src not in seen_sources:
                    seen_sources.add(src)
                    source_lines.append(f"- {src}")
            sources = "\n\nSources:\n" + "\n".join(source_lines)
            answer = answer + sources
            
        return answer, session_id
    
    def update_knowledge_base(self, documents_directory: str):
        """Update the knowledge base with new documents"""
        self.vectorstore = self.document_service.process_documents(documents_directory)
        self.retriever = RerankRetriever(
            vectorstore=self.vectorstore,
            k=settings.TOP_K_RESULTS,
            fetch_k=settings.RERANK_CANDIDATES,
            use_reranker=settings.RERANK_ENABLED,
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm_service.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.QA_PROMPT},
            return_source_documents=True,
            verbose=True
        )
    
    @staticmethod
    def get_available_models():
        """Return list of available models"""
        return LLMService.get_available_models()
    
    @staticmethod
    def get_available_embedding_types():
        """Return list of available embedding types"""
        return EmbeddingService.get_available_embedding_types()
