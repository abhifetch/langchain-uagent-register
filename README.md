# langchain-uagent-register

[![Python Versions](https://img.shields.io/pypi/pyversions/langchain-uagent-register.svg)](https://pypi.org/project/langchain-uagent-register/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A LangChain integration tool for converting LangChain agents into [uAgents](https://fetch.ai/products/uagent) and registering them on the [Agentverse](https://agentverse.ai) platform.

## Installation

```bash
pip install langchain-uagent-register
```

## Overview

This package provides the `UAgentRegisterTool` that bridges LangChain agents with the uAgent ecosystem. It allows you to:

1. Convert a LangChain agent into a uAgent
2. Start the uAgent as an HTTP service
3. Register the agent on Agentverse for discovery
4. Update the agent's README with appropriate documentation
5. Enable the agent to communicate with other agents via the Agentverse protocol

## Usage

### Basic Example

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_uagent_register import UAgentRegisterTool

# Create a simple tool for the agent to use
def calculator(expression: str) -> str:
    """Evaluates a math expression."""
    return str(eval(expression))

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for when you need to calculate math expressions"
    )
]

# Create a LangChain agent
llm = ChatOpenAI()
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Create the UAgentRegisterTool
uagent_tool = UAgentRegisterTool()

# Convert your LangChain agent to a uAgent and register it
agent_info = uagent_tool.invoke({
    "agent_obj": agent,
    "name": "Math_Agent",
    "port": 8000,
    "description": "A mathematical agent that can evaluate expressions",
    "api_token": "your-agentverse-api-token"  # Get this from agentverse.ai
})

# Print the agent's address
print(f"Agent address: {agent_info['address']}")
print(f"Agent running on port: {agent_info['port']}")

# The agent is now running and can be accessed at:
# http://localhost:8000/

# To stop the agent when done
from langchain_uagent_register import cleanup_uagent
cleanup_uagent(agent_info["name"])

# Or to stop all agents
from langchain_uagent_register import cleanup_all_uagents
cleanup_all_uagents()
```

## API Reference

### UAgentRegisterTool

The main class that handles converting LangChain agents to uAgents and registering them on Agentverse.

```python
from langchain_uagent_register import UAgentRegisterTool

tool = UAgentRegisterTool()
```

#### Parameters

- `agent_obj` (required): The LangChain agent object to convert
- `name` (required): Name for the uAgent
- `port` (required): Port to run the uAgent on
- `description` (required): Description of the agent
- `api_token` (optional): API token for agentverse.ai

### Cleanup Functions

```python
from langchain_uagent_register import cleanup_uagent, cleanup_all_uagents

# Stop a specific agent
cleanup_uagent("agent_name")

# Stop all agents
cleanup_all_uagents()
```

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/abhifetch/langchain-uagent-register.git
cd langchain-uagent-register

# Install dependencies
poetry install --with test

# Run tests
poetry run pytest tests/unit_tests
poetry run pytest tests/integration_tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [uAgents](https://github.com/fetchai/uAgents) for the agent infrastructure
- [Agentverse](https://agentverse.ai) for the agent discovery platform 