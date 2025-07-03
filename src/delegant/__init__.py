"""
Delegant - Type-safe Pydantic wrapper for FastAgent with dynamic MCP server management
======================================================================================

Delegant transforms FastAgent's YAML configuration approach into runtime-configurable 
Pydantic models, enabling dynamic server management and type-safe agent creation 
without configuration files.

Key Features:
- 🔒 Type-safe MCP server wrappers with automatic validation
- 📝 Context extraction from docstrings and type annotations  
- 🚀 Dynamic server management without config files
- 🔄 Workflow decorators for agent orchestration
- 🏗️ Production-ready with comprehensive error handling

Quick Start:
-----------

```python
from delegant import Agent, FileSystemServer, WebSearchServer

class DocumentAgent(Agent):
    '''Agent specialized in document analysis and research.'''
    instruction: str = "You analyze documents and provide insights"
    
    # Servers are automatically instantiated with extracted context
    files: FileSystemServer
    search: WebSearchServer

# Context is automatically provided to servers from docstrings
agent = DocumentAgent()
result = await agent.files.read_file("document.pdf")
```

Workflow Decorators:
-------------------

```python
from delegant import chain, parallel, router, orchestrator

@chain(SearchAgent, AnalysisAgent)
class ResearchPipeline(Agent):
    async def research(self, topic: str) -> dict:
        return await self.execute_chain("search", topic)

@parallel(DataAgent, NewsAgent, SocialAgent)
class ParallelGathering(Agent):
    async def gather_all(self, topic: str) -> dict:
        return await self.execute_parallel("collect", topic)
```

Configuration:
--------------

```python
from delegant import configure

configure(
    max_retries=5,
    connection_timeout=60,
    debug_mode=True
)
```
"""

__version__ = "1.0.0"
__author__ = "Delegant Team"
__license__ = "MIT"

# Core components
from .core.agent import Agent, AgentBuilder, agent_config
from .core.server import MCPServer, MCPTool, HTTPMCPServer, ProcessMCPServer
from .core.config import configure, get_config, set_config, reset_config, ConfigContext
from .core.context import ContextMetadata, get_context_extractor

# Server implementations
from .servers.filesystem import FileSystemServer
from .servers.websearch import WebSearchServer, SearchResult
from .servers.terminal import TerminalServer, CommandResult
from .servers.atuin import AtuinServer, AtuinHistoryEntry

# Workflow decorators
from .workflows.decorators import (
    chain, router, parallel, orchestrator,
    WorkflowResult, ChainExecutor, RouterExecutor, 
    ParallelExecutor, OrchestratorExecutor
)

# Exceptions
from .exceptions import (
    DelegantException,
    ServerConnectionError,
    ToolDiscoveryError, 
    ToolExecutionError,
    ContextExtractionError,
    WorkflowExecutionError,
    ValidationError,
    ConfigurationError,
    RetryExhaustedError
)

# Utility functions
from .utils.connection import ConnectionPool
from .utils.retry import retry_with_backoff
from .utils.validation import validate_server_config

# Public API exports
__all__ = [
    # Core classes
    "Agent",
    "MCPServer", 
    "MCPTool",
    "HTTPMCPServer",
    "ProcessMCPServer",
    
    # Agent utilities
    "AgentBuilder",
    "agent_config",
    
    # Server implementations
    "FileSystemServer",
    "WebSearchServer", 
    "TerminalServer",
    "AtuinServer",
    
    # Data models
    "SearchResult",
    "CommandResult", 
    "AtuinHistoryEntry",
    "ContextMetadata",
    "WorkflowResult",
    
    # Workflow decorators
    "chain",
    "router", 
    "parallel",
    "orchestrator",
    
    # Workflow executors (for advanced usage)
    "ChainExecutor",
    "RouterExecutor",
    "ParallelExecutor", 
    "OrchestratorExecutor",
    
    # Configuration
    "configure",
    "get_config",
    "set_config", 
    "reset_config",
    "ConfigContext",
    
    # Context extraction
    "get_context_extractor",
    
    # Exceptions
    "DelegantException",
    "ServerConnectionError",
    "ToolDiscoveryError",
    "ToolExecutionError", 
    "ContextExtractionError",
    "WorkflowExecutionError",
    "ValidationError",
    "ConfigurationError",
    "RetryExhaustedError",
    
    # Utilities
    "ConnectionPool",
    "retry_with_backoff",
    "validate_server_config",
    
    # Meta
    "__version__",
]


class Delegant:
    """Global configuration and utility class.
    
    Provides a central point for configuring Delegant behavior and accessing
    library-wide functionality.
    """
    
    @classmethod
    def configure(
        cls,
        auto_retry: bool = True,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        connection_timeout: int = 30,
        lazy_connect: bool = True,
        context_extraction: bool = True,
        debug_mode: bool = False,
        **kwargs
    ) -> None:
        """Configure global Delegant behavior.
        
        This is the main public API for configuration, as mentioned in the spec.
        
        Args:
            auto_retry: Enable automatic retries for failed operations
            max_retries: Maximum retry attempts
            retry_backoff: Backoff multiplier for retries
            connection_timeout: Default connection timeout in seconds
            lazy_connect: Enable lazy connection (connect on first use)
            context_extraction: Enable context extraction from docstrings/annotations
            debug_mode: Enable debug mode with additional logging
            **kwargs: Additional configuration parameters
        """
        configure(
            auto_retry=auto_retry,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            connection_timeout=connection_timeout,
            lazy_connect=lazy_connect,
            context_extraction=context_extraction,
            debug_mode=debug_mode,
            **kwargs
        )
    
    @classmethod
    def get_version(cls) -> str:
        """Get the current Delegant version."""
        return __version__
    
    @classmethod
    def get_server_types(cls) -> list:
        """Get list of available server types."""
        return [
            "FileSystemServer",
            "WebSearchServer", 
            "TerminalServer",
            "AtuinServer"
        ]
    
    @classmethod
    def get_workflow_types(cls) -> list:
        """Get list of available workflow decorator types."""
        return ["chain", "router", "parallel", "orchestrator"]
    
    @classmethod
    def create_agent(cls, instruction: str, **server_configs) -> Agent:
        """Create a basic agent with specified servers.
        
        Args:
            instruction: Agent instruction/role description
            **server_configs: Server configurations by name
            
        Returns:
            Configured agent instance
            
        Example:
            agent = Delegant.create_agent(
                "File analysis agent",
                files={"root_dir": "/data"},
                search={"provider": "duckduckgo"}
            )
        """
        class DynamicAgent(Agent):
            pass
        
        # Set instruction
        DynamicAgent.instruction = instruction
        
        # Add servers based on configs
        for server_name, server_config in server_configs.items():
            if server_name == "files":
                DynamicAgent.__annotations__[server_name] = FileSystemServer
            elif server_name == "search":
                DynamicAgent.__annotations__[server_name] = WebSearchServer
            elif server_name == "terminal":
                DynamicAgent.__annotations__[server_name] = TerminalServer
            elif server_name == "atuin":
                DynamicAgent.__annotations__[server_name] = AtuinServer
        
        # Create and configure agent
        agent = DynamicAgent(instruction=instruction)
        
        # Apply server configurations
        for server_name, config in server_configs.items():
            if hasattr(agent, server_name):
                server = getattr(agent, server_name)
                for key, value in config.items():
                    if hasattr(server, key):
                        setattr(server, key, value)
        
        return agent


# Convenience functions for common patterns

def create_file_agent(root_dir: str = None, **kwargs) -> Agent:
    """Create an agent specialized for file operations.
    
    Args:
        root_dir: Root directory for file operations
        **kwargs: Additional FileSystemServer configuration
        
    Returns:
        Configured file agent
    """
    config = {"root_dir": root_dir} if root_dir else {}
    config.update(kwargs)
    
    return Delegant.create_agent(
        "File operations agent specialized in reading, writing, and managing files",
        files=config
    )


def create_search_agent(provider: str = "duckduckgo", **kwargs) -> Agent:
    """Create an agent specialized for web search.
    
    Args:
        provider: Search provider (duckduckgo, google, bing)
        **kwargs: Additional WebSearchServer configuration
        
    Returns:
        Configured search agent
    """
    config = {"provider": provider}
    config.update(kwargs)
    
    return Delegant.create_agent(
        "Web search agent specialized in finding and retrieving online information",
        search=config
    )


def create_terminal_agent(shell: str = "/bin/bash", **kwargs) -> Agent:
    """Create an agent specialized for terminal operations.
    
    Args:
        shell: Shell to use for command execution
        **kwargs: Additional TerminalServer configuration
        
    Returns:
        Configured terminal agent
    """
    config = {"shell": shell}
    config.update(kwargs)
    
    return Delegant.create_agent(
        "Terminal operations agent specialized in executing system commands safely",
        terminal=config
    )


def create_research_agent(
    search_provider: str = "duckduckgo",
    file_root: str = None,
    **kwargs
) -> Agent:
    """Create an agent specialized for research (search + file operations).
    
    Args:
        search_provider: Search provider to use
        file_root: Root directory for file operations
        **kwargs: Additional configuration
        
    Returns:
        Configured research agent
    """
    search_config = {"provider": search_provider}
    file_config = {"root_dir": file_root} if file_root else {}
    
    return Delegant.create_agent(
        "Research agent specialized in gathering information from web sources and managing research files",
        search=search_config,
        files=file_config
    )


# Library initialization
def _initialize_library():
    """Initialize the Delegant library with default settings."""
    try:
        # Set up default configuration from environment
        config = get_config()
        config.configure_logging()
        
        # Log library initialization
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Delegant v{__version__} initialized")
        
    except Exception as e:
        # Don't fail library import on initialization errors
        import warnings
        warnings.warn(f"Failed to initialize Delegant library: {e}")


# Initialize on import
_initialize_library()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        """Demonstrate basic Delegant usage."""
        print(f"Delegant v{Delegant.get_version()}")
        print(f"Available servers: {', '.join(Delegant.get_server_types())}")
        print(f"Available workflows: {', '.join(Delegant.get_workflow_types())}")
        
        # Create a simple research agent
        agent = create_research_agent(file_root="/tmp")
        print(f"Created agent: {agent.instruction}")
        print(f"Agent servers: {agent.list_servers()}")
        
        # Configure library globally
        Delegant.configure(
            max_retries=5,
            debug_mode=True
        )
        print("Library configured successfully")
    
    # Run demo
    # asyncio.run(demo())
