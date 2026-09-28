import asyncio
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent
from tavily import TavilyClient

load_dotenv()


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# Path-safe database location: works no matter which folder you run from
DB_PATH = Path(__file__).parent / "resources" / "Chinook.db"

RETRYABLE_MCP_CODES = {-32603}
MAX_WEB_SEARCHES = 12

# Groq free tier has a small tokens-per-minute limit, so keep tool outputs small
MAX_TOOL_CHARS = 5000


def truncate_result(result):
    """Cut very long MCP text results so the next model call stays under the TPM limit."""
    if isinstance(result, CallToolResult):
        for block in result.content:
            if isinstance(block, TextContent) and len(block.text) > MAX_TOOL_CHARS:
                block.text = block.text[:MAX_TOOL_CHARS] + "\n...[truncated]"
    return result


def _retry_hint(attempt: int) -> str:
    if attempt == 0:
        return ""
    return (
        "\n\n(Previous attempt failed because of a malformed tool call. "
        "Call tools only with their exact documented arguments, e.g. web_search needs a 'query' string.)"
    )


def safe_invoke(agent, query: str, retries: int = 3) -> str:
    """Run an agent; retry if the model produces an invalid tool call."""
    last_error = None
    for attempt in range(retries):
        try:
            response = agent.invoke(
                {"messages": [HumanMessage(content=query + _retry_hint(attempt))]}
            )
            return response["messages"][-1].content
        except Exception as exc:
            last_error = exc
            print(f"[safe_invoke] attempt {attempt + 1}/{retries} failed: {exc}")
    return f"Specialist agent failed after {retries} attempts: {last_error}"


async def safe_ainvoke(agent, query: str, retries: int = 3) -> str:
    """Async version of safe_invoke."""
    last_error = None
    for attempt in range(retries):
        try:
            response = await agent.ainvoke(
                {"messages": [HumanMessage(content=query + _retry_hint(attempt))]}
            )
            return response["messages"][-1].content
        except Exception as exc:
            last_error = exc
            print(f"[safe_ainvoke] attempt {attempt + 1}/{retries} failed: {exc}")
            await asyncio.sleep(2 ** attempt)
    return f"Specialist agent failed after {retries} attempts: {last_error}"


class RetryMCPInterceptor:
    """Intercept MCP tool calls and retry transient failures."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return truncate_result(await handler(request))

            except McpError as exc:
                last_error = exc

                print(
                    f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                    f"(code {exc.error.code}, attempt {attempt + 1}/{self.max_retries}): {exc}"
                )

                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Tool call failed (non-retryable): {exc}",
                            )
                        ],
                        isError=False,
                    )

            except Exception as exc:
                last_error = exc

                print(
                    f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                    f"(attempt {attempt + 1}/{self.max_retries}): {exc}"
                )

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        print(
            f"[MCP interceptor] all {self.max_retries} retries exhausted "
            f"for {request.name}"
        )

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Tool call failed after {self.max_retries} attempts: {last_error}",
                )
            ],
            isError=False,
        )


async def main():
    # Connect to the travel MCP server
    client = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com",
            }
        },
        tool_interceptors=[RetryMCPInterceptor()],
    )

    tools = await client.get_tools()
    print("MCP tools:", [t.name for t in tools])

    # Keep only the flight search tool: fewer tool schemas = fewer tokens per request
    flight_tools = [t for t in tools if "search" in t.name and "flight" in t.name]
    tools = flight_tools or tools

    # Create the web search tool (search limit is tracked in Python,
    # so the model doesn't have to count its own searches)
    tavily_client = TavilyClient()
    search_counter = {"count": 0}

    @tool
    def web_search(query: str) -> Dict[str, Any]:
        """Search the web. Always pass a 'query' string, e.g. 'wedding venues Paris 100 guests'."""
        search_counter["count"] += 1

        if search_counter["count"] > MAX_WEB_SEARCHES:
            return {
                "message": (
                    "Search limit reached. Please summarize your findings "
                    "and provide your final answer."
                )
            }

        try:
            raw = tavily_client.search(query, max_results=3)
            # Return only the useful fields: smaller and less confusing for the model
            return {
                "results": [
                    {
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "content": (r.get("content") or "")[:500],
                    }
                    for r in raw.get("results", [])
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    # Connect to the playlist database 
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

    @tool
    def get_db_schema() -> str:
        """Get all table names and their CREATE statements. Call this first."""
        return db.get_table_info()

    @tool
    def query_playlist_db(query: str) -> str:
        """Run a read-only SELECT query on the Chinook SQLite database."""
        if not query.strip().lower().startswith(("select", "with")):
            return "Only SELECT queries are allowed."
        try:
            return db.run(query)
        except Exception as e:
            return f"Error querying database: {e}"

    # Define the wedding state
    class WeddingState(AgentState):
        origin: str
        destination: str
        guest_count: str
        genre: str

    # Create the travel agent
    travel_agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="""
        You are a travel agent. Search for flights to the desired destination wedding location.
        You are not allowed to ask any more follow up questions, you must find the best flight options based on the following criteria:
        - Price (lowest, economy class)
        - Duration (shortest)
        - Date (time of year which you believe is best for a wedding at this location)
        To make things easy, only look for one ticket, one way.
        You may need to make multiple searches to iteratively find the best options.
        You will be given no extra information, only the origin and destination. It is your job to think critically about the best options.
        If the MCP tool fails, returns malformed output, or does not give you usable flight results, try the tool again.
        Once you have found the best options, let the user know your shortlist of options.
        """,
    )

    # Create the venue agent
    venue_agent = create_agent(
        model=model,
        tools=[web_search],
        system_prompt=f"""
        You are a venue specialist. Search for venues in the desired location, and with the desired capacity.
        You are not allowed to ask any more follow up questions, you must find the best venue options based on the following criteria:
        - Price (lowest)
        - Capacity (exact match)
        - Reviews (highest)
        You may need to make multiple searches to iteratively find the best options.
        You have a limit of {MAX_WEB_SEARCHES} web searches. When the tool tells you the
        limit is reached, stop searching and summarize the best options you have found so far.
        """,
    )

    # Create the playlist agent
    playlist_agent = create_agent(
        model=model,
        tools=[get_db_schema, query_playlist_db],
        system_prompt="""
        You are a playlist specialist. Query the sql database and curate the perfect playlist for a wedding given a genre.
        Once you have your playlist, calculate the total duration and cost of the playlist, each song has an associated price.
        If you run into errors when querying the database, try to fix them by making changes to the query.
        Do not come back empty handed, keep trying to query the db until you find a list of songs.

        This is a SQLite database (Chinook music store). Before writing any data queries,
        call get_db_schema first to discover the tables and columns.
        Useful join path: Track -> Genre (GenreId), Track -> Album (AlbumId), Album -> Artist (ArtistId).
        Track.Milliseconds is the duration and Track.UnitPrice is the price.
        """,
    )

    # Define tools for the main coordinator
    @tool
    async def search_flights(runtime: ToolRuntime) -> str:
        """Travel agent searches for flights to the desired destination wedding location."""
        origin = runtime.state["origin"]
        destination = runtime.state["destination"]

        return await safe_ainvoke(
            travel_agent, f"Find flights from {origin} to {destination}"
        )

    @tool
    def search_venues(runtime: ToolRuntime) -> str:
        """Venue agent chooses the best venue for the given location and capacity."""
        destination = runtime.state["destination"]
        capacity = runtime.state["guest_count"]

        query = (
            f"Find wedding venues in {destination} "
            f"for {capacity} guests"
        )

        return safe_invoke(venue_agent, query)

    @tool
    def suggest_playlist(runtime: ToolRuntime) -> str:
        """Playlist agent curates the perfect playlist for the given genre."""
        genre = runtime.state["genre"]

        query = f"Find {genre} tracks for wedding playlist"

        return safe_invoke(playlist_agent, query)

    @tool
    def update_state(
        origin: str,
        destination: str,
        guest_count: str,
        genre: str,
        runtime: ToolRuntime,
    ) -> Command:
        """Update the state with all required wedding information."""
        return Command(
            update={
                "origin": origin,
                "destination": destination,
                "guest_count": guest_count,
                "genre": genre,
                "messages": [
                    ToolMessage(
                        "Successfully updated state",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    # Create the main coordinator
    coordinator = create_agent(
        model=model,
        tools=[
            search_flights,
            search_venues,
            suggest_playlist,
            update_state,
        ],
        state_schema=WeddingState,
        system_prompt="""
        You are a wedding coordinator.

        First find all the information you need to update the state.
        When you have the information, update the state.

        Once that has completed and returned, you can delegate the tasks
        to your specialists for flights, venues, and playlists.

        Once you have received their answers, coordinate the perfect wedding for me.
        """,
    )

    # Test the wedding planner
    response = await coordinator.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "I'm from London and I'd like a wedding in Paris "
                        "for 100 guests, jazz-genre"
                    )
                )
            ]
        },
        config={
            "tags": ["WP"],
            "recursion_limit": 40,
        },
    )

    print("\nFinal Answer:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())