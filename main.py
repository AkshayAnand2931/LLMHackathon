import os
import json
from typing import List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader
from langchain.schema import Document
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

class PharmaceuticalRAG:
    def __init__(self, data_path: str, lmstudio_api_base: str = "http://localhost:1234/v1"):
        self.data_path = data_path
        self.lmstudio_api_base = lmstudio_api_base
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.qa_chain = None

    def load_json_files(self) -> List[Document]:
        documents = []
        for filename in os.listdir(self.data_path):
            if filename.endswith('.json'):
                file_path = os.path.join(self.data_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Process each section of the JSON separately
                    for key, value in data.items():
                        if isinstance(value, str):
                            doc = Document(
                                page_content=value,
                                metadata={
                                    "source": filename,
                                    "section": key
                                }
                            )
                            documents.append(doc)
        return documents

    def create_vector_store(self, documents: List[Document]):
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

    def setup_retrieval_qa(self):
        # Initialize LMStudio client
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        llm = OpenAI(
            base_url=self.lmstudio_api_base,
            api_key="not-needed",  # LMStudio doesn't need an API key
            streaming=True,
            callback_manager=callback_manager,
            temperature=0.7,
        )

        # Custom prompt template for pharmaceutical queries
        prompt_template = """You are a helpful AI assistant that provides information about pharmaceutical products. 
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="mmr",  # Use MMR for better diversity in retrieved documents
                search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
            ),
            chain_type_kwargs={"prompt": PROMPT}
        )

    def initialize(self):
        """Initialize the RAG system"""
        documents = self.load_json_files()
        self.create_vector_store(documents)
        self.setup_retrieval_qa()

    def query(self, question: str) -> str:
        """Query the RAG system"""
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Call initialize() first.")
        
        response = self.qa_chain.run(question)
        return response

# Example usage
if __name__ == "__main__":
    rag = PharmaceuticalRAG(
        data_path="datasets/microlabs_usa",
        lmstudio_api_base="http://localhost:1234/v1"  # Adjust port as needed
    )
    rag.initialize()
    
    # Example query
    question = "What is the composition and primary use of Amoxicillin?"
    answer = rag.query(question)
    print(f"Q: {question}\nA: {answer}")