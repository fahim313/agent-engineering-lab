import asyncio
from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "time": {
                "transport": "stdio",
                "command": "uv",
                "args": [
                    "run",
                    "python",
                    "-m",
                    "mcp_server_time",
                    "--local-timezone=America/New_York",
                ],
            }
        }
    )

    tools = await client.get_tools()

    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )

    agent = create_agent(
        model=model,
        tools=tools,
    )

    response = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="What time is it?")
            ]
        }
    )

    pprint(response)


if __name__ == "__main__":
    asyncio.run(main())