from dotenv import load_dotenv
from langchain.agents import create_agent 
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_groq import ChatGroq 

load_dotenv() 


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# Add middleware to the model call 
@wrap_model_call
def long_model_call(request: ModelRequest, handler) -> ModelResponse:
    response= handler(request)
    return response


# create agent 
agent = create_agent(
    model =model,
    middleware = [long_model_call],
)

# Run agent 

response = agent.invoke(
    {
        "messages":[
            {
                "role":"user",
                "content":"What is the capital of France?"
            }
        ]
    }
)


print(response["messages"][-1].content)