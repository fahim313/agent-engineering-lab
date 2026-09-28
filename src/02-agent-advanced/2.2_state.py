from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()


# Define the custom agent state
class CustomState(AgentState):
    favourite_colour: str


# Write to state
@tool
def update_favourite_colour(
    favourite_colour: str,
    runtime: ToolRuntime,
) -> Command:
    """Update the favourite colour of the user in the state once they've revealed it."""
    return Command(
        update={
            "favourite_colour": favourite_colour,
            "messages": [
                ToolMessage(
                    "Successfully updated favourite colour",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# Create an agent with custom state
agent = create_agent(
    model=model,
    tools=[update_favourite_colour],
    checkpointer=InMemorySaver(),
    state_schema=CustomState,
)


# Store the favourite colour in state
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="My favourite colour is green")
        ]
    },
    {"configurable": {"thread_id": "1"}},
)

pprint(response)


# Pass state directly with the input
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="Hello, how are you?")
        ],
        "favourite_colour": "green",
    },
    {"configurable": {"thread_id": "10"}},
)

pprint(response)


# Read state
@tool
def read_favourite_colour(runtime: ToolRuntime) -> str:
    """Read the favourite colour of the user from the state."""
    try:
        return runtime.state["favourite_colour"]
    except KeyError:
        return "No favourite colour found in state"


# Create an agent with read and write state tools
agent = create_agent(
    model=model,
    tools=[
        update_favourite_colour,
        read_favourite_colour,
    ],
    checkpointer=InMemorySaver(),
    state_schema=CustomState,
)


# Update the favourite colour in state
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="My favourite colour is green")
        ]
    },
    {"configurable": {"thread_id": "1"}},
)

pprint(response)


# Read the favourite colour from state
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="What's my favourite colour?")
        ]
    },
    {"configurable": {"thread_id": "1"}},
)

pprint(response)