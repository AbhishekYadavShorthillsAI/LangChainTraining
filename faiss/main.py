import os
import json
from dotenv import load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from helicone.openai_proxy import openai

class OpenAIConfig:
    def __init__(self, api_key, api_base):
        # Configure OpenAI API settings
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key = api_key
        openai.api_base = api_base

class DocumentProcessor:
    def __init__(self, dir_path):
        # Load PDF documents and initialize text splitter
        
        self.loader = DirectoryLoader(dir_path, glob="./*.pdf", loader_cls=PyPDFLoader)
        self.documents = self.loader.load()
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    def split_documents(self):
        # Split loaded documents into chunks using text splitter
        return self.text_splitter.split_documents(self.documents)

class EmbeddingProcessor:
    def __init__(self, model_name):
        # Initialize instructor embeddings using HuggingFace model
        self.embeddings = SentenceTransformerEmbeddings(model_name=model_name)
    
    def create_embedding_database(self, docs):
        # Create embedding database using FAISS for the given documents and embeddings
    
        try:
            db = FAISS.load_local("faiss_index", self.embeddings)
        except:
            db = FAISS.from_documents(docs, self.embeddings)
            db.save_local("faiss_index")
            
        return db

class DocumentSearch:
    def __init__(self, embedding_db):
        # Initialize DocumentSearch with an embedding database
        self.embedding_db = embedding_db
    
    def search_similarity(self, query, k=5):
        # Perform similarity search on the embedding database
        return self.embedding_db.similarity_search(query, k=k)
    
    def get_similarity_metadata(self, results):
        src_meta_list = []
        
        for page in results:
            src_meta_list.append(page.metadata)
        return json.dumps(src_meta_list)

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    openai_config = OpenAIConfig(api_key, api_base)
    
    # Initialize ChatOpenAI model
    chat_model =  ChatOpenAI(temperature=0.0, model_kwargs={"engine": "GPT3-5"}, headers={
                            "Helicone-Auth": "Bearer sk-helicone-jocztra-rzquezq-vupgixi-ovqylny",
                            "Helicone-User-Id": "Abhishek.Yadav"})
    
    # Initialize DocumentProcessor with PDF path
    document_processor = DocumentProcessor('./input/')
    docs = document_processor.split_documents()
    
    # Initialize EmbeddingProcessor with model name
    embedding_processor = EmbeddingProcessor(model_name="all-MiniLM-L6-v2")
    # Create embedding database using documents and instructor embeddings
    faiss_vector_store = embedding_processor.create_embedding_database(docs)
    
    # Initialize DocumentSearch with embedding database
    search_processor = DocumentSearch(embedding_db=faiss_vector_store)
    
    
    # Print search results
    chain = load_qa_chain(chat_model, chain_type="stuff")
    
    # Perform similarity search on the database
    results = search_processor.search_similarity("What is discussed about LEGISLATIVE CHANGES IN GST LAWS", k=5)
    
    similarity_results_src = search_processor.get_similarity_metadata(results)
    print(similarity_results_src)
    print(chain.run(input_documents=results, question=results))
    
    
    
    
    # Print search results
    chain = load_qa_chain(chat_model, chain_type="stuff")
    
    # Perform similarity search on the database
    results = search_processor.search_similarity("What is discussed about LEGISLATIVE CHANGES IN GST LAWS", k=5)
    
    similarity_results_src = search_processor.get_similarity_metadata(results)
    print(similarity_results_src)
    print(chain.run(input_documents=results, question=results))
    
    
    