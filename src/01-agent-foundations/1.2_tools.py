from dotenv import load_dotenv 
from pprint import pprint 

from langchain_groq import ChatGroq 
from langchain.tools import tool 
from langchain.agents import create_agent 
from langchain.messages import HumanMessage 


# Load environment variables 
load_dotenv()

# model 
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# Define a tool 
@tool
def square_root(x:float)->float:
    """
    Calculates the square root of a number.
    
    """
    return x**0.5 

# Define a tool with a custom name
@tool("square_root")
def tool1(x:float)->float:
    """
    calculates the square root of a number.
    """
    return x**0.5

# Define a tool with a custom description
@tool(
    "square_root",
    description="Calculates the square root of a number."
) 
def tool1(x:float)->float:
    return x**0.5

# Invoke the tool
result = tool1.invoke({"x": 467})
print(result)

# Add the tool to an agent 
agent = create_agent(
    model= model,
    tools=[tool1],
    system_prompt=(
        "You are an arithmetic agenntic wizerd."
        "Use your tools to calculate the square root of a number."
        "and square of any number."
    )
)

# Ask the agent to use the tool 
question = HumanMessage(
    content="what is the square root of 467?"
)
response = agent.invoke(
    {"messages":[question]}
)
print(response["messages"][-1].content)

# Inspect the agent messages
pprint(response["messages"])

# Inspect the tool call 
print(response["messages"][1].tool_calls)