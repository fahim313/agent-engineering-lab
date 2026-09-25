from dotenv import load_dotenv
from pprint import pprint
from typing import Any

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient


# Load environment variables
load_dotenv()


#  model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# Create Tavily client
tavily_client = TavilyClient()


# Define web search tool
@tool
def web_search(query: str) -> dict[str, Any]:
    """Search the web for information."""
    return tavily_client.search(query)


# Define agent instructions
system_prompt = """
You are a personal chef.

The user will give you a list of ingredients they have
left over in their house.

Using the web search tool, search the web for recipes
that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe
instructions to the user, if requested.
"""


# Create personal chef agent
agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)


# Define conversation thread
config = {
    "configurable": {
        "thread_id": "1"
    }
}


# Ask for recipe suggestions
response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "I have some leftover chicken and rice. "
                    "What can I make?"
                )
            )
        ]
    },
    config
)

print(response["messages"][-1].content)


# Inspect the full response
pprint(response)