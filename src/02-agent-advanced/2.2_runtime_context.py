from dataclasses import dataclass
from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime
from langchain_groq import ChatGroq

load_dotenv()


# Define the runtime context
@dataclass
class ColourContext:
    favourite_colour: str = "blue"
    least_favourite_colour: str = "yellow"

# model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# Create an agent with a context schema
agent = create_agent(
    model=model,
    context_schema=ColourContext,
)


# Pass context to the agent at runtime
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="What is my favourite colour?")
        ]
    },
    context=ColourContext(),
)

pprint(response)


# Access runtime context from tools
@tool
def get_favourite_colour(
    runtime: ToolRuntime[ColourContext],
) -> str:
    """Get the favourite colour of the user."""
    return runtime.context.favourite_colour


@tool
def get_least_favourite_colour(
    runtime: ToolRuntime[ColourContext],
) -> str:
    """Get the least favourite colour of the user."""
    return runtime.context.least_favourite_colour


# Create an agent with context-aware tools
agent = create_agent(
    model=model,
    tools=[
        get_favourite_colour,
        get_least_favourite_colour,
    ],
    context_schema=ColourContext,
)


# Use the default context values
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="What is my favourite colour?")
        ]
    },
    context=ColourContext(),
)

pprint(response)


# Override the context for this run
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="What is my favourite colour?")
        ]
    },
    context=ColourContext(
        favourite_colour="green"
    ),
)

pprint(response)