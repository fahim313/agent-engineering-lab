# Agent Engineering Lab

A hands-on repository documenting my journey into **AI Agent Engineering** through practical implementations, experiments, and progressively advanced projects.

This repository focuses on building AI agents using LLMs, tools, memory, MCP, state management, multi-agent systems, and production-oriented agent patterns.

## Learning Roadmap

### 01. Agent Foundations

Learn the core concepts required to build AI agents.

- Foundational models
- Prompt engineering
- Tool calling
- Web search integration
- Memory management
- Multimodal messages
- Personal Chef AI agent

### 02. Advanced Agents

Explore advanced agent architectures and coordination patterns.

- Building MCP servers
- Local and remote MCP clients
- Travel planning agent
- Runtime context
- Agent state management
- Multi-agent systems
- Wedding Planner AI agent

### 03. Production-Ready Agents

Explore techniques for making AI agents more reliable and practical.

- Agent middleware
- Managing long conversations
- Human-in-the-loop workflows
- Dynamic agent configuration

## Project Structure

```text
agent-engineering-lab/
├── src/
│   ├── 01-agent-foundations/
│   │   ├── resources/
│   │   │   └── moon.png
│   │   ├── 1.1_foundational_models.py
│   │   ├── 1.1_prompting.py
│   │   ├── 1.2_tools.py
│   │   ├── 1.2_web_search.py
│   │   ├── 1.3_memory.py
│   │   ├── 1.4_multimodal_messages.py
│   │   └── 1.5_personal_chef.py
│   │
│   ├── 02-agent-advanced/
│   │   ├── resources/
│   │   ├── 2.1_mcp_server.py
│   │   ├── Chinook.db
│   │   ├── 2.1_mcp_client_local.py
│   │   ├── 2.1_mcp_client_online.py
│   │   ├── 2.1_travel_agent.py
│   │   ├── 2.2_runtime_context.py
│   │   ├── 2.2_state.py
│   │   ├── 2.3_multi_agent.py
│   │   └── 2.4_wedding_planner.py
│   │
│   └── 03-agent-production-ready/
│       ├── 3.1_middleware.py
│       ├── 3.2_long_conversations.py
│       ├── 3.3_human_in_the_loop.py
│       └── 3.4_dynamic_agents.py
│
└── README.md
```

## Technologies & Concepts

- **Language:** Python
- **LLM Integration:** LangChain and compatible model providers
- **Agent Framework:** LangChain and LangGraph ecosystem
- **Tool Integration:** Tool calling and web search
- **Agent Communication:** Model Context Protocol (MCP)
- **Agent Architecture:** Memory, state, runtime context, and multi-agent workflows
- **Development Tools:** Git and GitHub

*The technologies used may vary between individual experiments.*

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/fahim313/agent-engineering-lab.git
cd agent-engineering-lab
```

### 2. Set up the Python environment

This project uses Python. If you use `uv`, create and activate a virtual environment:

```bash
uv sync
```

If the repository does not yet contain a `pyproject.toml` and lockfile, initialize the environment and install the dependencies required by the relevant example first.

### 3. Configure environment variables

Create a `.env` file in the project root and configure the API keys required by the examples you want to run.

Depending on the implementation, these may include LLM provider keys and a web search API key.

**Security:** Never commit `.env` files or expose API keys in source code.

### 4. Run an example

Run a Python script from the project root. For example:

```bash
uv run python src/01-agent-foundations/1.1_foundational_models.py
```

Make sure the required dependencies and environment variables are configured for that example.

## Learning Objectives

By working through this repository, I aim to:

- Understand how LLM-powered agents work.
- Build agents that interact with external tools and services.
- Implement memory and context-aware workflows.
- Explore MCP-based integrations.
- Design collaborative multi-agent systems.
- Handle long conversations and human approval workflows.
- Learn patterns for building more reliable AI agent applications.

## Progress

- [x] Agent foundations
- [x] Advanced agent concepts
- [ ] Production-ready agent patterns

*Progress reflects the learning roadmap and will be updated as implementations evolve.*

## Repository

[**agent-engineering-lab — GitHub**](https://github.com/fahim313/agent-engineering-lab)

---

**Learn by building. Experiment by coding. Engineer reliable AI agents.**
