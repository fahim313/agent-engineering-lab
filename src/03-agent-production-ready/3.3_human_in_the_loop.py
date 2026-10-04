from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# Define a  tool
@tool
def send_email(to: str, subject: str, message: str) -> str:
    """Send an email to a user."""
    return f"Email sent to {to} with subject: {subject}"


# Add human approval before actions
middleware = [
    HumanInTheLoopMiddleware(
        interrupt_on={
            "send_email": True,
        }
    )
]


# Create the agent
agent = create_agent(
    model=model,
    tools=[send_email],
    middleware=middleware,
    checkpointer=InMemorySaver(),
)


config = {
    "configurable": {
        "thread_id": "email-1",
    }
}


# Ask the agent to send an email
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Send an email to john@example.com saying "
                    "that the meeting has been moved to tomorrow."
                ),
            }
        ]
    },
    config,
)


print("\nAgent stopped for approval.")
print(response)


# Check the interrupted state
state = agent.get_state(config)

if state.interrupts:
    print("\nHuman approval required.")
    print("Approve the email?")

    approval = input("Type yes or no: ")

    if approval.lower() == "yes":
        response = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {
                            "type": "approve",
                        }
                    ]
                }
            ),
            config,
        )

        print("\nFinal Answer:")
        print(response["messages"][-1].content)

    else:
        print("\nAction rejected.")