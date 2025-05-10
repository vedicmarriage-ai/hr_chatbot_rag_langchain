from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from app.core.config import get_settings
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService, EmbeddingType
from app.services.document_service import DocumentService
import uuid
import os

settings = get_settings()

class RAGService:
    def __init__(
        self,
        model_name: str = None,
        embedding_type: EmbeddingType = EmbeddingType.OLLAMA,
        embedding_model: str = "minilm"
    ):
        self.llm_service = LLMService(model_name=model_name)
        self.embedding_service = EmbeddingService(
            embedding_type=embedding_type,
            model_key=embedding_model
        )
        self.document_service = DocumentService()
        
        # Check if vector store exists
        if not os.path.exists(settings.CHROMA_PERSIST_DIRECTORY):
            raise ValueError(
                f"Vector store not found at {settings.CHROMA_PERSIST_DIRECTORY}. "
                "Please run create_vector_store.py first."
            )
        
        # Load existing vector store
        self.vectorstore = Chroma(
            persist_directory=str(settings.CHROMA_PERSIST_DIRECTORY),
            embedding_function=self.embedding_service.embeddings
        )
        
        # Custom prompt template for better responses
        self.qa_template = """
        You are an HR assistant chatbot. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Human: {question}
        Assistant: """
        
        self.QA_PROMPT = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Initialize retriever with search parameters
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant documents
        )
        
        # Initialize the chain
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
            
        # Process the message through the chain
        response = self.chain({"question": message})
        
        # Extract answer and source documents
        answer = response["answer"]
        source_docs = response.get("source_documents", [])
        
        # Format the response with sources if available
        if source_docs:
            sources = "\n\nSources:\n" + "\n".join([f"- {doc.metadata.get('source', 'Unknown')}" for doc in source_docs])
            answer = answer + sources
            
        return answer, session_id
    
    def update_knowledge_base(self, documents_directory: str):
        """Update the knowledge base with new documents"""
        self.vectorstore = self.document_service.process_documents(documents_directory)
        # Reinitialize the retriever with the new vector store
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        # Reinitialize the chain with the new retriever
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