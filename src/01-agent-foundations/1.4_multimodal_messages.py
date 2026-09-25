from dotenv import load_dotenv
import base64

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.messages import HumanMessage


# Load environment variables
load_dotenv()


# model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# Create text agent
agent = create_agent(
    model=model,
    system_prompt=(
        "You are a science fiction writer. "
        "Create a capital city at the user's request."
    )
)


# Text input
question = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What is the capital of The Moon?"
        }
    ]
)

response = agent.invoke(
    {"messages": [question]}
)

print(response["messages"][-1].content)


# Initialize vision model
vision_model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)


# Create vision agent
vision_agent = create_agent(
    model=vision_model,
    system_prompt=(
        "You are a science fiction writer. "
        "Describe the capital city shown in the image."
    )
)


# Read image file
image_path = "src/01-agent-foundations/resources/moon.png"

with open(image_path, "rb") as image_file:
    image_bytes = image_file.read()


# Encode image as base64
image_b64 = base64.b64encode(image_bytes).decode("utf-8")


# Image + text input
multimodal_question = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Tell me about this capital."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}"
            }
        }
    ]
)

response = vision_agent.invoke(
    {"messages": [multimodal_question]}
)

print(response["messages"][-1].content)