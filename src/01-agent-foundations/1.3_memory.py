from dotenv import load_dotenv
from pprint import pprint

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver


# Load environment variables
load_dotenv()


# Model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# Agent without memory
agent = create_agent(
    model=model
)

question = HumanMessage(
    content="Hello my name is Seán and my favourite colour is green"
)

response = agent.invoke(
    {"messages": [question]}
)

pprint(response)


# Ask the agent about previous information
question = HumanMessage(
    content="What's my favourite colour?"
)

response = agent.invoke(
    {"messages": [question]}
)

pprint(response)


# Agent with memory
agent = create_agent(
    model=model,
    checkpointer=InMemorySaver()
)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

question = HumanMessage(
    content="Hello my name is Seán and my favourite colour is green"
)

response = agent.invoke(
    {"messages": [question]},
    config
)

pprint(response)


# Ask the agent about previous information
question = HumanMessage(
    content="What's my favourite colour?"
)

response = agent.invoke(
    {"messages": [question]},
    config
)

pprint(response)