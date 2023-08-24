import os
from dotenv import load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

class OpenAIConfig:
    def __init__(self, api_key, api_base):
        # Configure OpenAI API settings
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key = api_key
        openai.api_base = api_base

class DocumentProcessor:
    def __init__(self, pdf_path):
        # Load PDF documents and initialize text splitter
        self.loader = PyPDFLoader(pdf_path)
        self.documents = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    def split_documents(self):
        # Split loaded documents into chunks using text splitter
        return self.text_splitter.split_documents(self.documents)

class EmbeddingProcessor:
    def __init__(self, model_name):
        # Initialize instructor embeddings using HuggingFace model
        self.instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    
    def create_embedding_database(self, docs):
        # Create embedding database using FAISS for the given documents and embeddings
        return FAISS.from_documents(docs, self.instructor_embeddings)

class DocumentSearch:
    def __init__(self, embedding_db):
        # Initialize DocumentSearch with an embedding database
        self.embedding_db = embedding_db
    
    def search_similarity(self, query, k=5):
        # Perform similarity search on the embedding database
        return self.embedding_db.similarity_search(query, k=k)

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    
    # Configure OpenAI API using provided API key and base URL
    openai_config = OpenAIConfig(api_key, api_base)
    
    # Initialize ChatOpenAI model
    chat_model = ChatOpenAI(temperature=0.0, model_kwargs={"engine": "GPT3-5"})
    
    # Initialize DocumentProcessor with PDF path
    document_processor = DocumentProcessor('./input/Python Part 7 Inheritance.pdf')
    # Split PDF documents into chunks
    docs = document_processor.split_documents()
    
    # Initialize EmbeddingProcessor with model name
    embedding_processor = EmbeddingProcessor(model_name="hkunlp/instructor-xl")
    # Create embedding database using documents and instructor embeddings
    db_instructEmbedd = embedding_processor.create_embedding_database(docs)
    
    # Initialize DocumentSearch with embedding database
    search_processor = DocumentSearch(embedding_db=db_instructEmbedd)
    # Perform similarity search on the database
    
    
    # Print search results
    chain = load_qa_chain(chat_model, chain_type="stuff")
    results = search_processor.search_similarity("What is multi-level inheritance in python.", k=5)
    print(chain.run(input_documents=results, question=results))

 