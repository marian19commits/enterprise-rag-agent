import os
import logging
from typing import List, Tuple, Iterator, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from src.config import settings

logger = logging.getLogger(__name__)

class RAGEngine:
    """Enterprise RAG engine wrapping document ingestion and semantic search."""
    
    def __init__(self, provider: str = "ollama", persist_directory: str = None):
        self.persist_directory = persist_directory or settings.PERSIST_DIRECTORY
        self.provider = provider.lower()
        self.vector_store = None

        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initializes LLM and Embedding providers based on selection."""
        if self.provider == "ollama":
            base_url = settings.OLLAMA_BASE_URL
            logger.info(f"Initializing Ollama provider at {base_url}")
            
            self.embeddings = OllamaEmbeddings(
                model=settings.DEFAULT_EMBED_MODEL,
                base_url=base_url
            )
            self.llm = ChatOllama(
                model=settings.DEFAULT_LLM_MODEL,
                temperature=0.2,
                base_url=base_url
            )

        elif self.provider == "azure":
            logger.info("Initializing Azure OpenAI provider")
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=settings.AZURE_EMBEDDING_DEPLOYMENT,
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY
            )
            self.llm = AzureChatOpenAI(
                azure_deployment=settings.AZURE_CHAT_DEPLOYMENT,
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                temperature=0.2,
                streaming=True
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def process_pdf(self, file_path: str, original_filename: str) -> int:
        """Ingests a PDF, splits it into semantic chunks, and persists to Vector DB."""
        logger.info(f"Processing PDF file: {original_filename}")
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()

        for doc in raw_docs:
            doc.metadata["source_file"] = original_filename
            doc.metadata["page"] = doc.metadata.get("page", 0) + 1

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(raw_docs)

        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vector_store.add_documents(chunks)

        return len(chunks)

    def query(self, question: str) -> Tuple[Iterator[Any], List[Document]]:
        """Queries the vector database and returns a response stream + citation sources."""
        if not self.vector_store:
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                raise ValueError("Vector database is empty. Please upload and index documents first.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        source_docs = retriever.invoke(question)

        context_text = "\n\n---\n\n".join(
            f"[Zdroj: {doc.metadata.get('source_file')}, Strana: {doc.metadata.get('page')}]\n{doc.page_content}"
            for doc in source_docs
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Jsi firemní asistent pro vyhledávání v dokumentaci.
Odpovídej přesně a výhradně na základě poskytnutého kontextu. Pokud informace v kontextu chybí, jasně to sděl.

Pravidla:
1. Odpověz strukturovaně v českém jazyce.
2. Uváděj přesné citace (soubor a stranu).

Kontext:
{context}"""),
            ("user", "{question}")
        ])

        chain = prompt_template | self.llm
        response_stream = chain.stream({"context": context_text, "question": question})

        return response_stream, source_docs