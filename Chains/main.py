import os
import openai
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_experimental.pal_chain import PALChain

class AIInterface:
    def __init__(self):
        load_dotenv()
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE")

        self.llm = OpenAI(temperature=0, model_kwargs={"engine": "GPT3-5"})

class FactExtractionChain:
    def __init__(self, llm):
        self.llm = llm
        self.fact_extraction_prompt = PromptTemplate(
            input_variables=["text_input"],
            template="Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences. :\n\n {text_input}"
        )
        self.fact_extraction_chain = LLMChain(llm=self.llm, prompt=self.fact_extraction_prompt)
    
    def run(self, article):
        facts = self.fact_extraction_chain.run(article)
        return facts

class InvestorUpdateChain:
    def __init__(self, llm):
        self.llm = llm
        self.investor_update_prompt = PromptTemplate(
            input_variables=["facts"],
            template="You are a Goldman Sachs analyst. Take the following list of facts and use them to write a short paragraph for investors. Don't leave out key info:\n\n {facts}"
        )
        self.investor_update_chain = LLMChain(llm=self.llm, prompt=self.investor_update_prompt)
    
    def run(self, facts):  
        investor_update = self.investor_update_chain.run(facts)
        return investor_update

class FullSeqChain:
    def __init__(self, fact_extraction_chain, investor_update_chain):
        self.full_chain = SimpleSequentialChain(chains=[fact_extraction_chain, investor_update_chain], verbose=True)

    def run(self, article):
      response = self.full_chain.run(article)
      return response
      

class PALChainWrapper:
    def __init__(self, llm):
        self.pal_chain = PALChain.from_math_prompt(llm, verbose=True)
    
    def run(self, question):
        pal_result = self.pal_chain.run(question)
        return pal_result

if __name__ == "__main__":
    ai_interface = AIInterface()

    article = '''Coinbase, the second-largest crypto exchange by trading volume, released its Q4 2022 earnings on Tuesday, giving shareholders and market players alike an updated look into its financials. In response to the report, the company's shares are down modestly in early after-hours trading.In the fourth quarter of 2022, Coinbase generated $605 million in total revenue, down sharply from $2.49 billion in the year-ago quarter. Coinbase's top line was not enough to cover its expenses: The company lost $557 million in the three-month period on a GAAP basis (net income) worth -$2.46 per share, and an adjusted EBITDA deficit of $124 million.Wall Street expected Coinbase to report $581.2 million in revenue and earnings per share of -$2.44 with adjusted EBITDA of -$201.8 million driven by 8.4 million monthly transaction users (MTUs), according to data provided by Yahoo Finance.Before its Q4 earnings were released, Coinbase's stock had risen 86% year-to-date. Even with that rally, the value of Coinbase when measured on a per-share basis is still down significantly from its 52-week high of $206.79.That Coinbase beat revenue expectations is notable in that it came with declines in trading volume; Coinbase historically generated the bulk of its revenues from trading fees, making Q4 2022 notable. Consumer trading volumes fell from $26 billion in the third quarter of last year to $20 billion in Q4, while institutional volumes across the same timeframe fell from $133 billion to $125 billion.The overall crypto market capitalization fell about 64%, or $1.5 trillion during 2022, which resulted in Coinbase's total trading volumes and transaction revenues to fall 50% and 66% year-over-year, respectively, the company reported.As you would expect with declines in trading volume, trading revenue at Coinbase fell in Q4 compared to the third quarter of last year, dipping from $365.9 million to $322.1 million. (TechCrunch is comparing Coinbase's Q4 2022 results to Q3 2022 instead of Q4 2021, as the latter comparison would be less useful given how much the crypto market has changed in the last year; we're all aware that overall crypto activity has fallen from the final months of 2021.)There were bits of good news in the Coinbase report. While Coinbase's trading revenues were less than exuberant, the company's other revenues posted gains. What Coinbase calls its "subscription and services revenue" rose from $210.5 million in Q3 2022 to $282.8 million in Q4 of the same year, a gain of just over 34% in a single quarter.And even as the crypto industry faced a number of catastrophic events, including the Terra/LUNA and FTX collapses to name a few, there was still growth in other areas. The monthly active developers in crypto have more than doubled since 2020 to over 20,000, while major brands like Starbucks, Nike and Adidas have dived into the space alongside social media platforms like Instagram and Reddit.With big players getting into crypto, industry players are hoping this move results in greater adoption both for product use cases and trading volumes. Although there was a lot of movement from traditional retail markets and Web 2.0 businesses, trading volume for both consumer and institutional users fell quarter-over-quarter for Coinbase.Looking forward, it'll be interesting to see if these pieces pick back up and trading interest reemerges in 2023, or if platforms like Coinbase will have to keep looking elsewhere for revenue (like its subscription service) if users continue to shy away from the market.'''

    # fact_extraction_obj = FactExtractionChain(ai_interface.llm)
    # facts = fact_extraction_obj.run(article)
    # # print(facts)

    # investor_update_obj = InvestorUpdateChain(ai_interface.llm)
    # investor_update = investor_update_obj.run(facts)
    # # print(investor_update)

    # full_chain = FullSeqChain(fact_extraction_obj.fact_extraction_chain, investor_update_obj.investor_update_chain)
    # response = full_chain.run(article)
    
    # print(response)

    pal_chain_wrapper = PALChainWrapper(ai_interface.llm)
    question_02 = "The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more, how many apples do they have?"

    pal_result = pal_chain_wrapper.run(question_02)

    print(pal_result)
