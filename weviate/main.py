import os
from dotenv import load_dotenv
from llm_usage import UsageUpdater

import openai
import weaviate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Weaviate
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from helicone.openai_proxy import openai
from langchain.callbacks import get_openai_callback


class LangChainApp:
    def __init__(self):
        load_dotenv()
        self.setup_openai()
        self.setup_weaviate()
        self.setup_components()
        self.usage_updater = UsageUpdater()

    def setup_openai(self):
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE")
    
    #creates weviate client
    def setup_weaviate(self):
        self.weaviate_url = os.getenv('WEAVIATE_URL')
        self.weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
        self.weaviate_client = weaviate.Client(url=self.weaviate_url, auth_client_secret=weaviate.AuthApiKey(self.weaviate_api_key))
    
    #initialises openai and embeddings
    def setup_components(self):
        self.llm = ChatOpenAI(temperature=0.0, model_kwargs={"engine": "GPT3-5"}, headers={
                            "Helicone-Auth": "Bearer sk-helicone-jocztra-rzquezq-vupgixi-ovqylny",
                            "Helicone-User-Id": "Abhishek.Yadav"})
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    #loads all pdfs from given directory recursively and returns chunks of all loaded data
    def load_documents_create_chunks(self, input_directory):
        loader = DirectoryLoader(input_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(pages)

   #clears all the dimensaions strored in initialised weviate client
    def clear_dimensions(self):
        self.weaviate_client.schema.delete_all()
    
    #gets all the stored classes or collections on wiviate
    def get_collections(self):
        return self.weaviate_client.schema.get()
    
    def build_vector_store(self, documents):
        # try:
        #     # Add the class to the schema
        #     self.weaviate_client.schema.create_class(class_obj)
        # except:
        #     print("Class already exists")
        # #self.weaviate_client.schema.create_class(class_obj)

        # load docs into the vectorstore
        vectorstore = Weaviate.from_documents(documents=documents, client=self.weaviate_client, embedding=self.embeddings, by_text=False)
        
        return vectorstore.as_retriever()
    
    #create conversation chain
    def build_qa_chain(self, retriever):

        #buffer memory for the conversation chain
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", 
            k=10, 
            input_key='question', 
            output_key='answer', 
            return_messages=True)

        qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm, 
            retriever=retriever, 
            memory=memory, 
            return_source_documents=True)
        return qa


    def query_qa(self, query, qa_chain):
        with get_openai_callback() as cb:
            res = qa_chain({"question": query})
            #notes the llm usage
            self.usage_updater.update_usage(cb)
            answer = res.get('answer')
        return answer

    
if __name__ == "__main__":
    app = LangChainApp()
    documents = app.load_documents_create_chunks('./input')
    retriever = app.build_vector_store(documents)
    qa_chain = app.build_qa_chain(retriever)

    queries = [
        "Vision for Amrit Kaal",
        "Pradhan Mantri PVTG Development Mission",
        "Vivad se Vishwas"
    ]
    for query in queries:
        print(query + ":", app.query_qa(query, qa_chain))
        print("****************************************************************************")

    print(app.usage_updater.get_daily_usage())
    # app.clear_dimensions()
    # print(app.get_collections())
