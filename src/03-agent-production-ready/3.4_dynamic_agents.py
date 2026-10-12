from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_groq import ChatGroq

load_dotenv()


basic_model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

advanced_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# Choose a model based on the request
@wrap_model_call
def dynamic_model_selection(request, handler):
    messages = request.state["messages"]
    last_message = messages[-1].content

    if isinstance(last_message, str) and len(last_message) > 500:
        model = advanced_model
        print("Using advanced model")
    else:
        model = basic_model
        print("Using basic model")

    return handler(request.override(model=model))


# Create the agent
agent = create_agent(
    model=basic_model,
    middleware=[dynamic_model_selection],
)


# Run the agent
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain the difference between AI and machine learning.",
            }
        ]
    }
)

print("Messages:", response["messages"])
print("Final message:", repr(response["messages"][-1].content))