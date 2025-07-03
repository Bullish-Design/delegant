"""
Delegant MCPServer Base Class
=============================

Base class for all MCP server connections with context extraction, connection management,
and automatic tool discovery. Provides the foundation for type-safe server wrappers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, PrivateAttr
import httpx
import json

from .exceptions import (
    ServerConnectionError, 
    ToolDiscoveryError, 
    ToolExecutionError,
    ContextExtractionError
)
from .config import get_config
from .context import ContextMetadata, get_context_extractor

logger = logging.getLogger(__name__)


@runtime_checkable
class MCPToolInterface(Protocol):
    """Interface for MCP tool implementations."""
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        ...
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's parameter schema."""
        ...


class MCPTool:
    """Standard implementation of an MCP tool."""
    
    def __init__(
        self, 
        name: str, 
        description: str,
        parameters_schema: Dict[str, Any],
        execution_func: callable
    ):
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema
        self.execution_func = execution_func
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with validation."""
        # Basic parameter validation could be added here
        if asyncio.iscoroutinefunction(self.execution_func):
            return await self.execution_func(**kwargs)
        else:
            return self.execution_func(**kwargs)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema
        }


class ConnectionStatus:
    """Connection status tracking."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class ServerConnection(BaseModel):
    """Internal model for managing server connections."""
    
    server_id: str = Field(..., description="Unique server identifier")
    connection_type: str = Field(..., description="HTTP, WebSocket, or process")
    status: str = Field(default=ConnectionStatus.DISCONNECTED, description="Connection status")
    last_activity: Optional[datetime] = Field(None, description="Last communication timestamp")
    error_count: int = Field(default=0, description="Consecutive error count")
    connection_url: Optional[str] = Field(None, description="Connection URL/endpoint")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional connection info")


class MCPServer(BaseModel, ABC):
    """Base class for all MCP server connections with context extraction.
    
    Provides connection management, tool discovery, context integration,
    and error handling for MCP server implementations.
    """
    
    # Configuration fields
    auto_discover_tools: bool = Field(
        default=False, 
        description="Enable automatic tool discovery"
    )
    connection_timeout: Optional[int] = Field(
        default=None, 
        description="Connection timeout in seconds (uses global default if None)"
    )
    context_extraction: bool = Field(
        default=True, 
        description="Extract context from docstrings/annotations"
    )
    lazy_connect: bool = Field(
        default=True, 
        description="Connect on first use instead of initialization"
    )
    
    # Private attributes for internal state
    _context_metadata: ContextMetadata = PrivateAttr(default_factory=ContextMetadata)
    _connection: Optional[ServerConnection] = PrivateAttr(default=None)
    _tools: Dict[str, MCPTool] = PrivateAttr(default_factory=dict)
    _client: Optional[httpx.AsyncClient] = PrivateAttr(default=None)
    _is_initialized: bool = PrivateAttr(default=False)
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Initialize connection info
        self._connection = ServerConnection(
            server_id=self._generate_server_id(),
            connection_type=self._get_connection_type()
        )
        
        # Extract context if enabled
        if self.context_extraction:
            try:
                extractor = get_context_extractor()
                self._context_metadata = extractor.extract_from_class(
                    self.__class__, 
                    server_attribute_name=getattr(self, '_server_attribute_name', None)
                )
            except Exception as e:
                logger.warning(f"Context extraction failed for {self.__class__.__name__}: {e}")
                # Don't fail initialization for context extraction errors
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to MCP server with context metadata.
        
        Must be implemented by each server type to handle specific connection logic.
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close server connection and cleanup resources.
        
        Must be implemented by each server type for proper cleanup.
        """
        pass
    
    @abstractmethod
    def _get_connection_type(self) -> str:
        """Return the connection type for this server (HTTP, WebSocket, process)."""
        pass
    
    @abstractmethod
    def _generate_server_id(self) -> str:
        """Generate unique identifier for this server instance."""
        pass
    
    async def ensure_connected(self) -> None:
        """Ensure server is connected, connecting if necessary."""
        if self._connection is None or self._connection.status != ConnectionStatus.CONNECTED:
            await self.connect()
    
    async def call_tool(self, tool_name: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Call a tool on the MCP server with error handling and retries.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ToolExecutionError: If tool execution fails
        """
        if self.lazy_connect:
            await self.ensure_connected()
        
        parameters = parameters or {}
        config = get_config()
        
        # Add context to parameters if supported
        if self._context_metadata and 'context' not in parameters:
            context_summary = self._create_context_summary()
            if context_summary:
                parameters['_delegant_context'] = context_summary
        
        start_time = datetime.now()
        
        for attempt in range(config.max_retries + 1):
            try:
                result = await self._execute_tool(tool_name, parameters)
                
                # Update connection status on success
                if self._connection:
                    self._connection.last_activity = datetime.now()
                    self._connection.error_count = 0
                    self._connection.status = ConnectionStatus.CONNECTED
                
                return result
                
            except Exception as e:
                if self._connection:
                    self._connection.error_count += 1
                
                if attempt < config.max_retries:
                    # Calculate backoff delay
                    delay = config.retry_backoff ** attempt
                    logger.warning(f"Tool execution failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    execution_time = (datetime.now() - start_time).total_seconds()
                    raise ToolExecutionError(
                        tool_name=tool_name,
                        server_name=self._connection.server_id if self._connection else "unknown",
                        parameters=parameters,
                        execution_time=execution_time,
                        original_error=e
                    )
    
    @abstractmethod
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute tool on the specific MCP server implementation.
        
        Must be implemented by each server type.
        """
        pass
    
    async def discover_tools(self) -> Dict[str, MCPTool]:
        """Discover available tools if auto_discover_tools=True.
        
        Returns:
            Dictionary of available tools by name
        """
        if not self.auto_discover_tools:
            return self._tools.copy()
        
        try:
            await self.ensure_connected()
            discovered_tools = await self._discover_tools_impl()
            self._tools.update(discovered_tools)
            return self._tools.copy()
            
        except Exception as e:
            raise ToolDiscoveryError(
                server_name=self._connection.server_id if self._connection else "unknown",
                discovery_method="auto_discovery",
                available_tools=list(self._tools.keys()),
                original_error=e
            )
    
    @abstractmethod
    async def _discover_tools_impl(self) -> Dict[str, MCPTool]:
        """Implementation-specific tool discovery.
        
        Must be implemented by each server type.
        """
        pass
    
    def get_context(self) -> Dict[str, Any]:
        """Returns extracted context from docstrings and type hints."""
        if self._context_metadata:
            return self._context_metadata.model_dump()
        return {}
    
    def add_context(self, key: str, value: Any) -> None:
        """Add additional context information."""
        if not self._context_metadata:
            self._context_metadata = ContextMetadata()
        
        # Store in relationships for now, could expand context model
        if 'custom' not in self._context_metadata.relationships:
            self._context_metadata.relationships['custom'] = []
        self._context_metadata.relationships['custom'].append(f"{key}: {value}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status and statistics."""
        if not self._connection:
            return {"status": ConnectionStatus.DISCONNECTED}
        
        return {
            "status": self._connection.status,
            "server_id": self._connection.server_id,
            "connection_type": self._connection.connection_type,
            "last_activity": self._connection.last_activity,
            "error_count": self._connection.error_count,
            "available_tools": list(self._tools.keys())
        }
    
    def register_tool(self, tool: MCPTool) -> None:
        """Manually register a tool with the server."""
        self._tools[tool.name] = tool
    
    def _create_context_summary(self) -> Optional[Dict[str, Any]]:
        """Create a summary of context for tool execution."""
        if not self._context_metadata:
            return None
        
        summary = {}
        
        # Add purpose and domain if available
        if self._context_metadata.purpose:
            summary['purpose'] = self._context_metadata.purpose
        if self._context_metadata.domain:
            summary['domain'] = self._context_metadata.domain
        
        # Add key variable names as semantic hints
        if self._context_metadata.variable_names:
            summary['semantic_context'] = self._context_metadata.variable_names[:5]  # Limit to first 5
        
        # Add class description if available
        if self._context_metadata.class_docstring:
            # Take first sentence as brief description
            first_sentence = self._context_metadata.class_docstring.split('.')[0]
            if len(first_sentence) < 200:  # Keep it brief
                summary['description'] = first_sentence
        
        return summary if summary else None
    
    async def health_check(self) -> bool:
        """Perform a health check on the server connection."""
        try:
            if self._connection and self._connection.status == ConnectionStatus.CONNECTED:
                # Try a simple operation to verify connection
                await self._health_check_impl()
                return True
        except Exception as e:
            logger.warning(f"Health check failed for {self._connection.server_id}: {e}")
            if self._connection:
                self._connection.status = ConnectionStatus.ERROR
        
        return False
    
    async def _health_check_impl(self) -> None:
        """Implementation-specific health check.
        
        Default implementation does nothing. Servers can override for specific checks.
        """
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.lazy_connect:
            await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# Base implementations for common server types

class HTTPMCPServer(MCPServer):
    """Base class for HTTP-based MCP servers."""
    
    base_url: str = Field(..., description="Base URL for the MCP server")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional headers")
    
    def _get_connection_type(self) -> str:
        return "HTTP"
    
    def _generate_server_id(self) -> str:
        return f"http_{self.__class__.__name__.lower()}_{id(self)}"
    
    async def connect(self) -> None:
        """Establish HTTP connection."""
        if self._connection:
            self._connection.status = ConnectionStatus.CONNECTING
        
        try:
            # Create HTTP client
            timeout = self.connection_timeout or get_config().connection_timeout
            headers = self.headers.copy()
            
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            self._client = httpx.AsyncClient(
                timeout=timeout,
                headers=headers
            )
            
            # Test connection
            response = await self._client.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            if self._connection:
                self._connection.status = ConnectionStatus.CONNECTED
                self._connection.connection_url = self.base_url
                self._connection.last_activity = datetime.now()
            
            logger.info(f"Connected to HTTP MCP server: {self.base_url}")
            
        except Exception as e:
            if self._connection:
                self._connection.status = ConnectionStatus.ERROR
            raise ServerConnectionError(
                server_name=self._connection.server_id if self._connection else "unknown",
                server_type="HTTP",
                connection_url=self.base_url,
                original_error=e
            )
    
    async def disconnect(self) -> None:
        """Close HTTP connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
        
        if self._connection:
            self._connection.status = ConnectionStatus.DISCONNECTED
    
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute tool via HTTP request."""
        if not self._client:
            raise ServerConnectionError(
                server_name=self._connection.server_id if self._connection else "unknown",
                server_type="HTTP",
                connection_url=self.base_url
            )
        
        try:
            response = await self._client.post(
                f"{self.base_url}/tools/{tool_name}",
                json=parameters
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            raise ToolExecutionError(
                tool_name=tool_name,
                server_name=self._connection.server_id if self._connection else "unknown",
                parameters=parameters,
                original_error=e
            )
    
    async def _discover_tools_impl(self) -> Dict[str, MCPTool]:
        """Discover tools via HTTP endpoint."""
        if not self._client:
            raise ToolDiscoveryError(
                server_name=self._connection.server_id if self._connection else "unknown",
                discovery_method="http_endpoint"
            )
        
        try:
            response = await self._client.get(f"{self.base_url}/tools")
            response.raise_for_status()
            tools_data = response.json()
            
            discovered_tools = {}
            for tool_data in tools_data.get('tools', []):
                tool = MCPTool(
                    name=tool_data['name'],
                    description=tool_data.get('description', ''),
                    parameters_schema=tool_data.get('parameters', {}),
                    execution_func=lambda **kwargs, tn=tool_data['name']: self.call_tool(tn, kwargs)
                )
                discovered_tools[tool.name] = tool
            
            return discovered_tools
            
        except Exception as e:
            raise ToolDiscoveryError(
                server_name=self._connection.server_id if self._connection else "unknown",
                discovery_method="http_endpoint",
                original_error=e
            )
    
    async def _health_check_impl(self) -> None:
        """HTTP-specific health check."""
        if self._client:
            response = await self._client.get(f"{self.base_url}/health")
            response.raise_for_status()


class ProcessMCPServer(MCPServer):
    """Base class for process-based MCP servers."""
    
    command: List[str] = Field(..., description="Command to start the MCP server process")
    working_directory: Optional[str] = Field(None, description="Working directory for the process")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    _process: Optional[asyncio.subprocess.Process] = PrivateAttr(default=None)
    
    def _get_connection_type(self) -> str:
        return "process"
    
    def _generate_server_id(self) -> str:
        return f"process_{self.__class__.__name__.lower()}_{id(self)}"
    
    async def connect(self) -> None:
        """Start the MCP server process."""
        if self._connection:
            self._connection.status = ConnectionStatus.CONNECTING
        
        try:
            # Start the process
            env = {**os.environ, **self.environment} if self.environment else None
            
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory,
                env=env
            )
            
            if self._connection:
                self._connection.status = ConnectionStatus.CONNECTED
                self._connection.connection_url = ' '.join(self.command)
                self._connection.last_activity = datetime.now()
            
            logger.info(f"Started MCP server process: {' '.join(self.command)}")
            
        except Exception as e:
            if self._connection:
                self._connection.status = ConnectionStatus.ERROR
            raise ServerConnectionError(
                server_name=self._connection.server_id if self._connection else "unknown",
                server_type="process",
                connection_url=' '.join(self.command),
                original_error=e
            )
    
    async def disconnect(self) -> None:
        """Terminate the MCP server process."""
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            
            self._process = None
        
        if self._connection:
            self._connection.status = ConnectionStatus.DISCONNECTED
    
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute tool via process communication."""
        if not self._process:
            raise ServerConnectionError(
                server_name=self._connection.server_id if self._connection else "unknown",
                server_type="process"
            )
        
        try:
            # Send JSON-RPC request to process
            request = {
                "jsonrpc": "2.0",
                "method": f"tools/{tool_name}",
                "params": parameters,
                "id": 1
            }
            
            request_data = json.dumps(request).encode() + b'\n'
            self._process.stdin.write(request_data)
            await self._process.stdin.drain()
            
            # Read response
            response_line = await self._process.stdout.readline()
            response_data = json.loads(response_line.decode())
            
            if 'error' in response_data:
                raise Exception(response_data['error'])
            
            return response_data.get('result')
            
        except Exception as e:
            raise ToolExecutionError(
                tool_name=tool_name,
                server_name=self._connection.server_id if self._connection else "unknown",
                parameters=parameters,
                original_error=e
            )
    
    async def _discover_tools_impl(self) -> Dict[str, MCPTool]:
        """Discover tools via process communication."""
        # Implementation would depend on the specific MCP process protocol
        return {}
    
    async def _health_check_impl(self) -> None:
        """Process-specific health check."""
        if self._process and self._process.returncode is not None:
            raise Exception("Process has terminated")


# Example usage
if __name__ == "__main__":
    class ExampleHTTPServer(HTTPMCPServer):
        """Example HTTP MCP server implementation."""
        
        def __init__(self, **data):
            super().__init__(base_url="http://localhost:8000", **data)
    
    async def test_server():
        async with ExampleHTTPServer(auto_discover_tools=True) as server:
            tools = await server.discover_tools()
            print(f"Discovered tools: {list(tools.keys())}")
            
            context = server.get_context()
            print(f"Server context: {context}")
    
    # Run test
    # asyncio.run(test_server())
