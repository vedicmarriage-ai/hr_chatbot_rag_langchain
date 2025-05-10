from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from app.services.embedding_service import EmbeddingService
from app.core.config import get_settings
import os
from typing import List, Optional
from langchain.schema import Document

settings = get_settings()

class DocumentService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
    def load_documents(self, directory: str) -> List[Document]:
        """Load documents from a directory"""
        # Load .docx files
        docx_loader = DirectoryLoader(
            directory,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader
        )
        docx_docs = docx_loader.load()
        
        # Load .txt files
        txt_loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        txt_docs = txt_loader.load()
        
        # Combine all documents
        return docx_docs + txt_docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
    
    def create_vector_store(
        self,
        documents: List[Document],
        persist_directory: str,
        embedding_function: Optional[object] = None
    ) -> Chroma:
        """Create and persist a vector store from documents"""
        if persist_directory is None:
            persist_directory = settings.CHROMA_PERSIST_DIRECTORY
            
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Create and persist vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=persist_directory
        )
        
        # Persist the vector store
        vectorstore.persist()
        return vectorstore
    
    def process_documents(self, directory: str):
        """Complete document processing pipeline"""
        # Load documents
        documents = self.load_documents(directory)
        
        # Split documents
        splits = self.split_documents(documents)
        
        # Create and persist vector store
        vectorstore = self.create_vector_store(splits)
        
        return vectorstore 