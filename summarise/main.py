import os
import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API configuration
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

class FileSummarizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.chat_model = ChatOpenAI(temperature=0.0, model_kwargs={"engine": "GPT3-5"})

    def file_loader(self):
        # Load the PDF file using PyPDFLoader
        loader = PyPDFLoader(self.file_path)
        pages = loader.load()
        
        # Extract content from each page
        pages_content_list = [page.page_content for page in pages]
        
        return pages_content_list
    
    def splitter(self, pages_content_list):
        # Split text into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            length_function=len
        )
        texts = text_splitter.create_documents(pages_content_list)

        return texts
        
    def map_reduce_summary(self):
        # Load content from the file and split into chunks
        pages_content_list = self.file_loader()
        texts = self.splitter(pages_content_list)
        
        # Load the summarize chain
        chain = load_summarize_chain(self.chat_model, chain_type="map_reduce")
        
        # Run the chain on the first two chunks
        print(chain.run(texts[0:2]))

# Path to the PDF file
file_path = "/home/shtlp_0146/Desktop/summarise/budget_speech.pdf"

# Create a FileSummarizer instance
summarizer = FileSummarizer(file_path)

# Call the map_reduce_summary method to generate summaries
summarizer.map_reduce_summary()
