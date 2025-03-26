"""Unit tests for UAgentRegisterTool."""

from typing import Type
from unittest.mock import MagicMock, patch

import pytest
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_uagent_register.tools import UAgentRegisterTool


class TestUAgentRegisterToolUnit(ToolsUnitTests):
    """Unit tests for UAgentRegisterTool."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, request: pytest.FixtureRequest) -> MagicMock:
        """Set up mocks for testing."""
        # Patch the Agent creation to avoid actual networking
        patcher = patch("langchain_uagent_register.tools.Agent")
        mock_agent = patcher.start()
        
        # Mock the agent instances
        mock_agent_instance = MagicMock()
        mock_agent_instance.address = "agent1mock0123456789xxxxxx"
        mock_agent.return_value = mock_agent_instance
        
        # Patch the thread handling
        thread_patcher = patch("langchain_uagent_register.tools.threading.Thread")
        mock_thread = thread_patcher.start()
        
        # Use pytest's cleanup mechanism
        request.addfinalizer(patcher.stop)
        request.addfinalizer(thread_patcher.stop)
        
        # Return the mocks in case we need them in tests
        return {
            "agent": mock_agent,
            "thread": mock_thread
        }

    @property
    def tool_constructor(self) -> Type[UAgentRegisterTool]:
        """Return the constructor for the tool."""
        return UAgentRegisterTool

    @property
    def tool_constructor_params(self) -> dict:
        """Return parameters for initializing the tool."""
        # Parameters for initializing the UAgentRegisterTool
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """Return an example of the parameters to pass to the tool's invoke method."""
        return {
            "agent_obj": "langchain_agent_object",  # Test mode agent
            "name": "test_agent",
            "port": 8080,
            "description": "Test agent for unit testing",
            "api_token": "test_api_token"
        } 