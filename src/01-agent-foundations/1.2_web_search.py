from dotenv import load_dotenv 
from pprint import pprint 
from typing import Any 

from langchain_groq import ChatGroq 
from langchain.tools import tool 
from langchain.agents import create_agent 
from langchain.messages import HumanMessage 
from tavily import TavilyClient 


# Load environment variables 

load_dotenv()

# model 
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# create tavily client 
tavily_client = TavilyClient()

# create agent without web search 
agent = create_agent(
    model= model
)
question = HumanMessage(
    content="How up to date is your training knowledge?"
)
response = agent.invoke(
    {"messages":[question]}
)
print(response["messages"][-1].content)


# Define a web search tool 
@tool
def web_search(query:str)-> dict[str,Any]:
    """
    Search the web for information.
    """
    return tavily_client.search(query)

# Test the web search tool directly 
search_result = web_search.invoke(
    "who is the current mayor of the san Francisco?"
)
pprint(search_result)    

# Add web search tool to the agent 
agent = create_agent(
    model=model,
    tools=[web_search]
)
question = HumanMessage(
    content="who is the current mayor of san Francisco?"
)
response = agent.invoke(
    {"messages": [question]}
)

pprint(response["messages"])