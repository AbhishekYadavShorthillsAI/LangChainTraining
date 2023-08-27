import os
import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.utilities import SerpAPIWrapper

class LangChainManager:
    def __init__(self):
        load_dotenv()
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE")

        self.llm = ChatOpenAI(temperature=0, model_kwargs={"engine": "GPT3-5"})
        self.tools = load_tools(["serpapi", "llm-math"], llm=self.llm)

        self.zero_shot_agent = self.initialize_zero_shot_agent()

    def initialize_zero_shot_agent(self):
        return initialize_agent(
            agent="zero-shot-react-description",
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            max_iterations=3
        )

    def run_queries(self):
        queries = [
            "what is (4.5*2.1)^2.2?",
            """if Mary has four apples and Giorgio brings two and a half apple 
            boxes (apple box contains eight apples), how many apples do we 
            have?""",
            "what is the capital of Norway?"
        ]

        for query in queries:
            response = self.zero_shot_agent(query)
            print(response)

if __name__ == "__main__":
    manager = LangChainManager()
    manager.run_queries()
