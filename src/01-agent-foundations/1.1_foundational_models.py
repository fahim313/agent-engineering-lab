from dotenv import load_dotenv
from pprint import pprint

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage


# Load environment variables
load_dotenv()


# Initialize Groq model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# Invoke model
response = model.invoke(
    "What is LangGraph? Answer in one sentence."
)

print(response.content)


# Response metadata
pprint(response.response_metadata)


# Create agent
agent = create_agent(
    model=model
)


# Invoke agent
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="What is LangGraph?")
        ]
    }
)

print(response["messages"][-1].content)


# Conversation
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="What is LangGraph?"),
            AIMessage(
                content="LangGraph is a framework for building stateful AI agents."
            ),
            HumanMessage(content="What can I use it for?")
        ]
    }
)

print(response["messages"][-1].content)


# Streaming output
for token, metadata in agent.stream(
    {
        "messages": [
            HumanMessage(
                content="Explain LangGraph in simple terms."
            )
        ]
    },
    stream_mode="messages"
):
    if token.content:
        print(token.content, end="", flush=True)