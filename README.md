# Langchain UAgent Register Tool

A Langchain tool for converting Langchain agents into uAgents and registering them on Agentverse.

## Features

- Convert Langchain agents to uAgents
- Automatic port allocation with fallback options
- Register agents on Agentverse
- Support for AI agent message forwarding
- Clean shutdown and resource management

## Installation

```bash
pip install langchain-uagent-register
```

## Usage

### Basic Example

```python
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_uagent_register import UAgentRegisterTool, cleanup_uagent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API tokens
API_TOKEN = os.getenv("AV_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create your Langchain agent
def calculator_tool(expression: str) -> str:
    """Evaluates a basic math expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for evaluating math expressions"
    )
]

llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Convert to uAgent and register
tool = UAgentRegisterTool()
agent_info = tool.invoke({
    "agent_obj": agent,
    "name": "calculator_agent",
    "port": 8080,
    "description": "A calculator agent for testing",
    "api_token": API_TOKEN
})

print(f"Created uAgent '{agent_info['name']}' with address {agent_info['address']} on port {agent_info['port']}")

# Clean up when done
cleanup_uagent("calculator_agent")
```

### Environment Variables

The tool requires the following environment variables:

- `AV_API_KEY`: Your Agentverse API key for registering agents
- `OPENAI_API_KEY`: Your OpenAI API key for the Langchain agent

You can set these in a `.env` file or export them in your environment:

```bash
export AV_API_KEY="your_agentverse_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

### Port Allocation

The tool automatically handles port allocation:

1. First tries to use the specified port
2. If the port is in use, searches for an available port in the range 8000-9000
3. Raises a RuntimeError if no ports are available

You can customize the port range:

```python
agent_info = tool.invoke({
    "agent_obj": agent,
    "name": "calculator_agent",
    "port": 8080,  # Preferred port
    "start_range": 8000,  # Start of port range
    "end_range": 9000,    # End of port range
    "description": "A calculator agent for testing",
    "api_token": API_TOKEN
})
```

### AI Agent Message Forwarding

You can forward messages to an AI agent by specifying its address:

```python
agent_info = tool.invoke({
    "agent_obj": agent,
    "name": "calculator_agent",
    "port": 8080,
    "description": "A calculator agent for testing",
    "api_token": API_TOKEN,
    "ai_agent_address": "agent1test123"  # Forward messages to this agent
})
```

### Error Handling

The tool includes built-in error handling for common scenarios:

1. Port conflicts: Automatically finds an available port
2. API key validation: Checks for required API keys
3. Agent name validation: Ensures valid agent names
4. Resource cleanup: Properly cleans up resources on shutdown

Example error handling:

```python
try:
    agent_info = tool.invoke({
        "agent_obj": agent,
        "name": "calculator_agent",
        "port": 8080,
        "description": "A calculator agent for testing",
        "api_token": API_TOKEN
    })
except ValueError as e:
    print(f"Validation error: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
finally:
    cleanup_uagent("calculator_agent")
```

## Development

### Running Tests

```bash
# Run unit tests
pytest tests/unit_tests

# Run integration tests
pytest tests/integration_tests

# Run all tests with socket disabled
pytest --disable-socket --allow-unix-socket --asyncio-mode=auto tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 