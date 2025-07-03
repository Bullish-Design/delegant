"""
Delegant Agent Base Class
=========================

Base class for creating agents with typed server access and automatic instantiation.
Provides Pydantic validation, context propagation, and dynamic server management.
"""

import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints, get_origin
from pydantic import BaseModel, Field, PrivateAttr
from datetime import datetime

from .server import MCPServer
from .context import ContextMetadata, get_context_extractor
from .config import get_config
from .exceptions import (
    ValidationError, 
    ServerConnectionError, 
    ContextExtractionError,
    ConfigurationError
)

logger = logging.getLogger(__name__)


class Agent(BaseModel):
    """Base class for creating agents with typed server access and context propagation.
    
    Agents automatically instantiate servers based on type annotations and propagate
    contextual information extracted from docstrings and class structure.
    
    Example:
        class DocumentAgent(Agent):
            '''Agent specialized in document analysis and research.'''
            instruction: str = "You analyze documents and provide insights"
            
            files: FileSystemServer
            search: WebSearchServer
        
        agent = DocumentAgent()
        result = await agent.files.read_file("document.pdf")
    """
    
    instruction: str = Field(..., description="Agent instruction/role description")
    
    # Private attributes for internal state management
    _servers: Dict[str, MCPServer] = PrivateAttr(default_factory=dict)
    _context_cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _server_configs: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)
    _is_initialized: bool = PrivateAttr(default=False)
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Extract and instantiate servers from type annotations
        self._extract_and_instantiate_servers()
        
        # Extract context for the agent
        self._extract_agent_context()
        
        self._is_initialized = True
    
    def __init_subclass__(cls, **kwargs):
        """Automatically process server annotations when subclass is created."""
        super().__init_subclass__(**kwargs)
        cls._validate_server_annotations()
    
    @classmethod
    def _validate_server_annotations(cls) -> None:
        """Validate that server type annotations are correct."""
        try:
            type_hints = get_type_hints(cls)
            
            for attr_name, type_hint in type_hints.items():
                # Skip standard fields like 'instruction'
                if attr_name in ['instruction']:
                    continue
                
                # Check if the type hint is an MCPServer subclass
                origin = get_origin(type_hint)
                if origin is None:  # Direct type, not generic
                    if (inspect.isclass(type_hint) and 
                        issubclass(type_hint, MCPServer)):
                        # Valid server annotation
                        continue
                
                # If we get here, it might not be a server type
                # Log a warning but don't fail
                logger.debug(f"Type annotation {attr_name}: {type_hint} may not be an MCPServer")
                
        except Exception as e:
            logger.warning(f"Failed to validate server annotations for {cls.__name__}: {e}")
    
    def _extract_and_instantiate_servers(self) -> None:
        """Extract server types from annotations and instantiate them."""
        try:
            type_hints = get_type_hints(self.__class__)
            
            for attr_name, type_hint in type_hints.items():
                # Skip non-server attributes
                if attr_name in ['instruction']:
                    continue
                
                # Check if this is an MCPServer type
                if (inspect.isclass(type_hint) and 
                    issubclass(type_hint, MCPServer)):
                    
                    # Get any configuration for this server
                    server_config = self._server_configs.get(attr_name, {})
                    
                    # Instantiate the server
                    try:
                        server_instance = type_hint(**server_config)
                        
                        # Store reference to attribute name for context
                        server_instance._server_attribute_name = attr_name
                        
                        # Add to servers dict
                        self._servers[attr_name] = server_instance
                        
                        # Set as attribute on the agent for direct access
                        setattr(self, attr_name, server_instance)
                        
                        logger.debug(f"Instantiated server {attr_name}: {type_hint.__name__}")
                        
                    except Exception as e:
                        logger.error(f"Failed to instantiate server {attr_name}: {e}")
                        raise ConfigurationError(
                            config_source=f"server_{attr_name}",
                            suggested_fix=f"Check {type_hint.__name__} configuration parameters",
                            original_error=e
                        )
        
        except Exception as e:
            if not isinstance(e, ConfigurationError):
                raise ConfigurationError(
                    config_source="agent_initialization",
                    suggested_fix="Check server type annotations and configurations",
                    original_error=e
                )
            raise
    
    def _extract_agent_context(self) -> None:
        """Extract context for the agent and propagate to servers."""
        try:
            config = get_config()
            if not config.context_extraction:
                return
            
            # Extract context for this agent class
            extractor = get_context_extractor()
            agent_context = extractor.extract_from_class(self.__class__)
            
            # Store in context cache
            self._context_cache['agent'] = agent_context.model_dump()
            
            # Propagate context to all servers
            for server_name, server in self._servers.items():
                if server.context_extraction:
                    # Create server-specific context
                    server_context = extractor.extract_from_class(
                        server.__class__, 
                        server_attribute_name=server_name
                    )
                    
                    # Merge with agent context
                    merged_context = extractor.merge_contexts(agent_context, server_context)
                    
                    # Update server's context
                    server._context_metadata = merged_context
                    
        except Exception as e:
            logger.warning(f"Context extraction failed for agent {self.__class__.__name__}: {e}")
            # Don't fail initialization for context extraction errors
    
    def get_server(self, name: str) -> MCPServer:
        """Get server instance by name.
        
        Args:
            name: Server attribute name
            
        Returns:
            Server instance
            
        Raises:
            ValidationError: If server doesn't exist
        """
        if name not in self._servers:
            raise ValidationError(
                validation_target=f"server_{name}",
                field_errors={"server_name": f"Server '{name}' not found"},
                invalid_value=name
            )
        
        return self._servers[name]
    
    def list_servers(self) -> Dict[str, str]:
        """List all available servers with their types.
        
        Returns:
            Dictionary mapping server names to their class names
        """
        return {
            name: server.__class__.__name__ 
            for name, server in self._servers.items()
        }
    
    async def add_server(self, name: str, server: MCPServer, connect: bool = True) -> None:
        """Add server dynamically and optionally connect.
        
        Args:
            name: Name for the server
            server: Server instance to add
            connect: Whether to connect immediately
            
        Raises:
            ValidationError: If server name conflicts or server is invalid
        """
        if name in self._servers:
            raise ValidationError(
                validation_target=f"server_{name}",
                field_errors={"server_name": f"Server '{name}' already exists"},
                invalid_value=name
            )
        
        if not isinstance(server, MCPServer):
            raise ValidationError(
                validation_target=f"server_{name}",
                field_errors={"server_type": "Must be an MCPServer instance"},
                invalid_value=type(server).__name__
            )
        
        # Store server
        self._servers[name] = server
        setattr(self, name, server)
        
        # Store attribute name for context
        server._server_attribute_name = name
        
        # Connect if requested
        if connect and not server.lazy_connect:
            try:
                await server.connect()
            except Exception as e:
                # Remove server if connection fails
                del self._servers[name]
                if hasattr(self, name):
                    delattr(self, name)
                raise ServerConnectionError(
                    server_name=name,
                    server_type=server.__class__.__name__,
                    original_error=e
                )
        
        logger.info(f"Added server '{name}' of type {server.__class__.__name__}")
    
    async def remove_server(self, name: str, disconnect: bool = True) -> None:
        """Remove server and optionally disconnect.
        
        Args:
            name: Server name to remove
            disconnect: Whether to disconnect before removal
            
        Raises:
            ValidationError: If server doesn't exist
        """
        if name not in self._servers:
            raise ValidationError(
                validation_target=f"server_{name}",
                field_errors={"server_name": f"Server '{name}' not found"},
                invalid_value=name
            )
        
        server = self._servers[name]
        
        # Disconnect if requested
        if disconnect:
            try:
                await server.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting server '{name}': {e}")
        
        # Remove server
        del self._servers[name]
        if hasattr(self, name):
            delattr(self, name)
        
        logger.info(f"Removed server '{name}'")
    
    async def connect_all_servers(self) -> Dict[str, bool]:
        """Connect all servers that aren't already connected.
        
        Returns:
            Dictionary mapping server names to connection success status
        """
        results = {}
        
        for name, server in self._servers.items():
            try:
                await server.connect()
                results[name] = True
                logger.debug(f"Connected server '{name}'")
            except Exception as e:
                results[name] = False
                logger.error(f"Failed to connect server '{name}': {e}")
        
        return results
    
    async def disconnect_all_servers(self) -> Dict[str, bool]:
        """Disconnect all connected servers.
        
        Returns:
            Dictionary mapping server names to disconnection success status
        """
        results = {}
        
        for name, server in self._servers.items():
            try:
                await server.disconnect()
                results[name] = True
                logger.debug(f"Disconnected server '{name}'")
            except Exception as e:
                results[name] = False
                logger.error(f"Failed to disconnect server '{name}': {e}")
        
        return results
    
    async def health_check_all_servers(self) -> Dict[str, bool]:
        """Perform health checks on all servers.
        
        Returns:
            Dictionary mapping server names to health status
        """
        results = {}
        
        for name, server in self._servers.items():
            try:
                is_healthy = await server.health_check()
                results[name] = is_healthy
            except Exception as e:
                results[name] = False
                logger.warning(f"Health check failed for server '{name}': {e}")
        
        return results
    
    def get_context(self) -> Dict[str, Any]:
        """Get the full context for this agent and its servers.
        
        Returns:
            Complete context information including agent and server contexts
        """
        full_context = {
            'agent': self._context_cache.get('agent', {}),
            'servers': {}
        }
        
        for name, server in self._servers.items():
            full_context['servers'][name] = server.get_context()
        
        return full_context
    
    def configure_server(self, server_name: str, **config) -> None:
        """Configure a server before it's instantiated (for use in __init__).
        
        Args:
            server_name: Name of the server attribute
            **config: Configuration parameters for the server
        """
        self._server_configs[server_name] = config
    
    async def execute_on_server(self, server_name: str, tool_name: str, **parameters) -> Any:
        """Execute a tool on a specific server with error handling.
        
        Args:
            server_name: Name of the server
            tool_name: Name of the tool to execute
            **parameters: Parameters to pass to the tool
            
        Returns:
            Tool execution result
        """
        server = self.get_server(server_name)
        return await server.call_tool(tool_name, parameters)
    
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all servers.
        
        Returns:
            Dictionary mapping server names to their status information
        """
        status = {}
        
        for name, server in self._servers.items():
            status[name] = server.get_connection_status()
        
        return status
    
    async def __aenter__(self):
        """Async context manager entry - connect all servers."""
        await self.connect_all_servers()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - disconnect all servers."""
        await self.disconnect_all_servers()


# Agent configuration utility
class AgentBuilder:
    """Builder pattern for creating agents with complex configurations."""
    
    def __init__(self, agent_class: Type[Agent]):
        self.agent_class = agent_class
        self.server_configs = {}
        self.agent_config = {}
    
    def configure_server(self, server_name: str, **config) -> 'AgentBuilder':
        """Configure a server."""
        self.server_configs[server_name] = config
        return self
    
    def configure_agent(self, **config) -> 'AgentBuilder':
        """Configure the agent."""
        self.agent_config = config
        return self
    
    def build(self) -> Agent:
        """Build the configured agent."""
        # Create agent instance
        agent = self.agent_class(**self.agent_config)
        
        # Apply server configurations
        for server_name, config in self.server_configs.items():
            if server_name in agent._servers:
                # Update server configuration
                server = agent._servers[server_name]
                for key, value in config.items():
                    if hasattr(server, key):
                        setattr(server, key, value)
        
        return agent


# Decorator for agent classes to add automatic configuration
def agent_config(**default_configs):
    """Decorator to add default configurations to agent classes.
    
    Args:
        **default_configs: Default configuration for servers
        
    Example:
        @agent_config(files={'root_dir': '/tmp'}, search={'api_key': 'key'})
        class MyAgent(Agent):
            files: FileSystemServer
            search: WebSearchServer
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, **kwargs):
            # Apply default server configurations
            for server_name, config in default_configs.items():
                self.configure_server(server_name, **config)
            
            # Call original init
            original_init(self, **kwargs)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


# Example usage and testing
if __name__ == "__main__":
    from .servers.filesystem import FileSystemServer
    from .servers.websearch import WebSearchServer
    
    class DocumentAgent(Agent):
        """Agent specialized in document analysis and research.
        
        This agent combines file operations with web search to provide
        comprehensive document analysis capabilities. It can read documents,
        analyze their content, and search for related information online.
        """
        instruction: str = "You analyze documents and provide insights"
        
        # Servers are automatically instantiated with extracted context
        files: FileSystemServer
        search: WebSearchServer
    
    @agent_config(files={'root_dir': '/tmp'})
    class ConfiguredAgent(Agent):
        """Agent with default configuration."""
        instruction: str = "Configured agent example"
        files: FileSystemServer
    
    async def test_agent():
        # Test basic agent creation
        agent = DocumentAgent(instruction="Analyze documents")
        
        print(f"Agent servers: {agent.list_servers()}")
        print(f"Agent context: {agent.get_context()}")
        
        # Test server access
        print(f"Files server: {agent.files}")
        print(f"Search server: {agent.search}")
        
        # Test agent builder
        builder = AgentBuilder(DocumentAgent)
        builder.configure_server('files', root_dir='/custom/path')
        builder.configure_agent(instruction="Custom instruction")
        
        custom_agent = builder.build()
        print(f"Custom agent: {custom_agent.instruction}")
        
        # Test async context manager
        async with agent:
            status = agent.get_server_status()
            print(f"Server status: {status}")
    
    # Run test
    # asyncio.run(test_agent())
