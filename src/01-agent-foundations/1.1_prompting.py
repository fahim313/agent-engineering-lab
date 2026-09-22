from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.messages import HumanMessage


# Load environment variables
load_dotenv()


# Initialize Groq model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# Basic prompting
agent = create_agent(
    model=model
)

question = HumanMessage(
    content="What's the capital of the moon?"
)

response = agent.invoke(
    {"messages": [question]}
)

print(response["messages"][-1].content)


# System prompt
system_prompt = (
    "You are a science fiction writer. "
    "Create a capital city at the user's request."
)

scifi_agent = create_agent(
    model=model,
    system_prompt=system_prompt
)

response = scifi_agent.invoke(
    {"messages": [question]}
)

print(response["messages"][-1].content)


# Few-shot examples
system_prompt = """
You are a science fiction writer.
Create a space capital city at the user's request.

User: What is the capital of Mars?
Scifi Writer: Marsialis

User: What is the capital of Venus?
Scifi Writer: Venusovia
"""

scifi_agent = create_agent(
    model=model,
    system_prompt=system_prompt
)

response = scifi_agent.invoke(
    {"messages": [question]}
)

print(response["messages"][-1].content)


# Structured prompt
system_prompt = """
You are a science fiction writer.
Create a space capital city at the user's request.

Please keep to the following structure:

Name: The name of the capital city

Location: Where it is based

Vibe: 2-3 words to describe its vibe

Economy: Main industries
"""

scifi_agent = create_agent(
    model=model,
    system_prompt=system_prompt
)

response = scifi_agent.invoke(
    {"messages": [question]}
)

print(response["messages"][-1].content)


# Structured output
class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str


agent = create_agent(
    model=model,
    system_prompt=(
        "You are a science fiction writer. "
        "Create a capital city at the user's request."
    ),
    response_format=CapitalInfo
)

question = HumanMessage(
    content="What is the capital of The Moon?"
)

response = agent.invoke(
    {"messages": [question]}
)

capital_info = response["structured_response"]

print(capital_info)


capital_name = capital_info.name
capital_location = capital_info.location

print(
    f"{capital_name} is a city located at {capital_location}"
)