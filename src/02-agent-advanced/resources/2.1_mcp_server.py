from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from typing import Dict, Any
from requests import get

load_dotenv()

mcp = FastMCP("mcp_server")

tavily_client = TavilyClient()


@mcp.tool()
def search_web(query: str) -> Dict[str, Any]:
    """Search the web for information."""
    return tavily_client.search(query)


@mcp.resource(
    "github://langchain-ai/langchain-mcp-adapters/main/README.md"
)
def github_file():
    """Access the langchain-mcp-adapters README file."""
    url = "https://raw.githubusercontent.com/langchain-ai/langchain-mcp-adapters/main/README.md"

    try:
        response = get(url)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.prompt()
def prompt():
    """Provide instructions for the assistant."""

    return """
    You are a helpful assistant that answers user questions about
    LangChain, LangGraph and LangSmith.

    You can use the following tools/resources:
    - search_web: Search the web for information
    - github_file: Access the langchain-mcp-adapters README file

    If the user asks a question that is not related to
    LangChain, LangGraph or LangSmith, say:
    "I'm sorry, I can only answer questions about LangChain, LangGraph and LangSmith."

    You may use multiple tool and resource calls when necessary.
    """


if __name__ == "__main__":
    mcp.run(transport="stdio")