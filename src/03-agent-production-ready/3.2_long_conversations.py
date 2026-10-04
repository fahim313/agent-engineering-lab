from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# Summarize old messages when the conversation gets long
middleware = [
    SummarizationMiddleware(
        model=model,
        max_tokens_before_summary=4000,
        messages_to_keep=10,
    )
]


# Create the agent
agent = create_agent(
    model=model,
    middleware=middleware,
    checkpointer=InMemorySaver(),
)


config = {
    "configurable": {
        "thread_id": "conversation-1",
    }
}


# First message
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "My name is Fahim and I am learning AI and machine learning.",
            }
        ]
    },
    config,
)

print("\nAssistant:")
print(response["messages"][-1].content)


# Continue the conversation
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What am I learning?",
            }
        ]
    },
    config,
)

print("\nAssistant:")
print(response["messages"][-1].content)


# Continue again
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is my name?",
            }
        ]
    },
    config,
)

print("\nAssistant:")
print(response["messages"][-1].content)