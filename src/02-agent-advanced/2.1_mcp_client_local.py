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
            "local_server": {
                "transport": "stdio",
                "command": "python",
                "args": [
                    "src/02-agent-advanced/resources/2.1_mcp_server.py"
                ],
            }
        }
    )

    tools = await client.get_tools()

    resources = await client.get_resources("local_server")

    prompt = await client.get_prompt("local_server", "prompt")
    prompt = prompt[0].content

    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt,
    )

    response = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Tell me about the langchain-mcp-adapters library"
                )
            ]
        }
    )

    pprint(response)


if __name__ == "__main__":
    asyncio.run(main())