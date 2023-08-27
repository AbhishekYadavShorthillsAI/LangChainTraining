import os
import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

class ChatBot:
    def __init__(self):
        load_dotenv()
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE")

        self.llm = ChatOpenAI(temperature=0, model_kwargs={"engine": "GPT3-5"})
        self.memory = ConversationSummaryMemory(llm=self.llm)
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory)

    def start(self):
        while True:
            message = input("Enter your message or type quit: ")
            if message.lower() == "quit":
                break
            result = self.conversation.predict(input=message)
            print("result:", result)
            
            print("\n*************************************")
            

if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.start()
    print(chatbot.memory.load_memory_variables({}))
