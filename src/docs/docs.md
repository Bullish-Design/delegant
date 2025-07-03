# Delegant Library - Complete Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [API Reference](#api-reference)
6. [Server Implementations](#server-implementations)
7. [Workflow Decorators](#workflow-decorators)
8. [Terminal Command Agent Demo](#terminal-command-agent-demo)
9. [Advanced Usage](#advanced-usage)
10. [Configuration](#configuration)
11. [Error Handling](#error-handling)
12. [Performance](#performance)
13. [Examples](#examples)
14. [Contributing](#contributing)

## Introduction

**Delegant** is a type-safe Pydantic wrapper for FastAgent that provides dynamic MCP server management and agent creation without configuration files. It transforms FastAgent's YAML configuration approach into runtime-configurable Pydantic models.

### Key Features

- 🔒 **Type-safe MCP server wrappers** with automatic validation
- 📝 **Context extraction** from docstrings and type annotations  
- 🚀 **Dynamic server management** without config files
- 🔄 **Workflow decorators** for agent orchestration
- 🏗️ **Production-ready** with comprehensive error handling
- 🤖 **Natural language terminal interface** (demo application)

### Why Delegant?

Traditional MCP setups require complex YAML configurations and lack type safety. Delegant provides:

- **Runtime Configuration**: No static config files needed
- **Type Safety**: Full Pydantic validation for all operations  
- **Context Awareness**: Automatically extracts semantic information from your code
- **Production Ready**: Comprehensive error handling, retry logic, and connection pooling

## Installation

```bash
# Basic installation
pip install delegant

# With all optional dependencies
pip install delegant[all]

# Specific feature sets
pip install delegant[web]      # Web search capabilities
pip install delegant[git]      # Git operations
pip install delegant[github]   # GitHub API integration
pip install delegant[monitoring] # Metrics collection
```

### Development Installation

```bash
git clone https://github.com/delegant/delegant.git
cd delegant
pip install -e .[dev,docs]
```

## Quick Start

### Basic Agent Creation

```python
from delegant import Agent, FileSystemServer, WebSearchServer

class DocumentAgent(Agent):
    """Agent specialized in document analysis and research.
    
    This agent combines file operations with web search to provide
    comprehensive document analysis capabilities.
    """
    instruction: str = "You analyze documents and provide insights"
    
    # Servers are automatically instantiated with extracted context
    files: FileSystemServer
    search: WebSearchServer

# Create and use the agent
async def main():
    async with DocumentAgent() as agent:
        # Read a document
        content = await agent.files.read_file("report.pdf")
        
        # Search for related information
        results = await agent.search.search("machine learning trends")
        
        print(f"Document content: {len(content)} characters")
        print(f"Search results: {len(results)} found")

# Context is automatically provided to servers from docstrings
asyncio.run(main())
```

### Terminal Command Agent (Demo)

```python
from delegant.examples.terminal_agent import TerminalCommandAgent

async def demo():
    agent = TerminalCommandAgent(
        use_devenv=False,  # Set to True if devenv.sh is available
        working_directory="/tmp/demo"
    )
    
    async with agent:
        # Process natural language commands
        result = await agent.process_natural_language(
            "list all Python files in the current directory",
            auto_execute=True,
            build_help_tree=True
        )
        
        print(f"Command: {result['best_command']}")
        print(f"Success: {result['execution_result']['success']}")

asyncio.run(demo())
```

## Core Concepts

### Agents

Agents are the main interface for interacting with MCP servers. They automatically instantiate servers based on type annotations and provide context-aware operations.

```python
class MyAgent(Agent):
    instruction: str = "Agent description"
    
    # Type annotations automatically create server instances
    files: FileSystemServer
    search: WebSearchServer
    terminal: TerminalServer
```

### Servers

Servers wrap MCP server functionality with type safety and context extraction:

- **FileSystemServer**: File operations with security restrictions
- **WebSearchServer**: Web search with multiple providers
- **TerminalServer**: Safe command execution with logging  
- **AtuinServer**: Command history management

### Context Extraction

Delegant automatically extracts context from:

- **Docstrings**: Class and method documentation
- **Type Annotations**: Parameter and return types
- **Variable Names**: Semantic identifiers
- **Comments**: Additional context information

This context is provided to MCP servers to enhance their understanding and responses.

### Workflow Decorators

Compose complex agent behaviors using decorators:

- `@chain`: Sequential execution
- `@router`: Intelligent routing  
- `@parallel`: Concurrent execution
- `@orchestrator`: Complex orchestration

## API Reference

### Core Classes

#### Agent

```python
class Agent(BaseModel):
    instruction: str  # Agent's role description
    
    # Methods
    async def add_server(name: str, server: MCPServer) -> None
    async def remove_server(name: str) -> None
    def get_server(name: str) -> MCPServer
    def list_servers() -> Dict[str, str]
    async def connect_all_servers() -> Dict[str, bool]
    async def disconnect_all_servers() -> Dict[str, bool]
```

#### MCPServer

```python
class MCPServer(BaseModel, ABC):
    auto_discover_tools: bool = False
    connection_timeout: Optional[int] = None
    context_extraction: bool = True
    lazy_connect: bool = True
    
    # Abstract methods
    async def connect() -> None
    async def disconnect() -> None
    async def call_tool(tool_name: str, parameters: Dict[str, Any]) -> Any
```

### Configuration

```python
from delegant import configure

configure(
    auto_retry=True,
    max_retries=3,
    retry_backoff=2.0,
    connection_timeout=30,
    lazy_connect=True,
    context_extraction=True,
    debug_mode=False
)
```

### Utilities

```python
from delegant import (
    retry_with_backoff,
    ConnectionPool,
    validate_server_config,
    get_context_extractor
)

# Retry decorator
@retry_with_backoff(max_attempts=5)
async def unreliable_operation():
    pass

# Connection pooling
async with ConnectionPool(max_size=50) as pool:
    server = await pool.get_connection(FileSystemServer, root_dir="/data")

# Validation
result = validate_server_config(WebSearchServer, {"provider": "google"})
```

## Server Implementations

### FileSystemServer

Secure file operations with configurable restrictions:

```python
server = FileSystemServer(
    root_dir="/safe/directory",
    allowed_extensions=[".txt", ".md", ".py"],
    max_file_size=10 * 1024 * 1024,  # 10MB
    readonly=False,
    enable_search=True
)

# Operations
content = await server.read_file("document.txt")
await server.write_file("output.txt", "content", create_dirs=True)
files = await server.list_directory(".", recursive=True)
results = await server.search_files("search term", search_content=True)
```

### WebSearchServer  

Multi-provider web search with caching:

```python
server = WebSearchServer(
    provider="duckduckgo",  # or "google", "bing"
    max_results=20,
    safe_search=True,
    enable_caching=True,
    # For Google/Bing:
    # api_key="your_api_key",
    # custom_search_engine_id="your_cse_id"
)

# Operations
results = await server.search("AI trends 2024")
images = await server.search_images("data visualization")
news = await server.search_news("technology", time_range="week")
```

### TerminalServer

Safe command execution with comprehensive logging:

```python
server = TerminalServer(
    shell="/bin/bash",
    working_directory="/project",
    allowed_commands=["git", "ls", "cat", "grep"],  # None allows all
    log_all_commands=True,
    use_devenv=True,  # devenv.sh integration
    timeout_seconds=300
)

# Operations
result = await server.execute_command("ls -la")
history = await server.get_command_history(limit=20)
parsed = await server.parse_command("git status", check_safety=True)
help_info = await server.get_command_help("git")
```

### AtuinServer

Command history management and analysis:

```python
server = AtuinServer(
    atuin_db_path="~/.local/share/atuin/history.db",
    enable_sync=False,
    max_results=100
)

# Operations
history = await server.search_history("git", limit=50)
stats = await server.get_statistics(period="week")
await server.import_command("ls -la", "/home/user", exit_code=0)
top_cmds = await server.get_top_commands(limit=10)
```

## Workflow Decorators

### Chain Decorator

Execute agents sequentially, passing output to the next agent:

```python
@chain(SearchAgent, AnalysisAgent, ReportAgent)
class ResearchPipeline(Agent):
    """Research pipeline with sequential processing."""
    
    async def research(self, topic: str) -> dict:
        return await self.execute_chain("process_topic", topic)

# Usage
pipeline = ResearchPipeline()
result = await pipeline.research("AI ethics")
```

### Router Decorator

Route requests to appropriate agents based on content:

```python
@router({
    "search": SearchAgent,
    "analysis": AnalysisAgent,
    "report": ReportAgent
})
class SmartRouter(Agent):
    """Intelligent request routing."""
    
    async def process(self, request: str, route_to: str) -> dict:
        return await self.execute_route("handle_request", request, route_to=route_to)

# Usage
router = SmartRouter()
result = await router.process("Find AI papers", route_to="search")
```

### Parallel Decorator

Execute multiple agents concurrently:

```python
@parallel(DataAgent, NewsAgent, SocialAgent)
class ParallelGathering(Agent):
    """Parallel data collection."""
    
    async def gather_all(self, topic: str) -> dict:
        return await self.execute_parallel("collect_data", topic)

# Usage
gatherer = ParallelGathering()
result = await gatherer.gather_all("climate change")
```

### Orchestrator Decorator

Complex multi-agent workflows with dependencies:

```python
@orchestrator({
    "search": SearchAgent,
    "analysis": AnalysisAgent,
    "report": ReportAgent
})
class ComplexWorkflow(Agent):
    """Complex orchestrated workflow."""
    
    def setup_orchestration(self):
        # Define workflow steps with dependencies
        self.add_orchestration_step("search", "search_data")
        self.add_orchestration_step("analysis", "analyze", depends_on=["step_0"])
        self.add_orchestration_step("report", "generate", depends_on=["step_1"])
    
    async def execute_workflow(self, topic: str) -> dict:
        self.setup_orchestration()
        return await self.execute_orchestration(topic)

# Usage
workflow = ComplexWorkflow()
result = await workflow.execute_workflow("market analysis")
```

## Terminal Command Agent Demo

The Terminal Command Agent demonstrates Delegant's capabilities with natural language command processing:

### Features

- **Natural Language Processing**: Converts English to terminal commands
- **Safety Analysis**: Validates commands before execution
- **Help Tree Building**: Recursive help command analysis
- **Command History**: Atuin integration for persistent history
- **Comprehensive Logging**: All stdio/stderr captured
- **devenv.sh Support**: Container-based command execution

### Running the Demo

```python
from delegant.examples.terminal_agent import TerminalCommandAgent

async def run_demo():
    agent = TerminalCommandAgent(
        instruction="Natural language terminal interface",
        use_devenv=False,  # Enable if devenv.sh available
        enable_atuin_sync=False,  # Enable if Atuin configured
        working_directory="/tmp/delegant_demo"
    )
    
    async with agent:
        # Interactive mode
        await agent.interactive_mode()

# Or run directly
if __name__ == "__main__":
    asyncio.run(run_demo())
```

### Example Interactions

```
🤖 What would you like me to do? list all Python files

📝 Parsed Commands:
   1. find . -name "*.py" (90% confidence, ✅ Safe)

❓ Execute 'find . -name "*.py"'? [y/N/h for help tree]: y

⚡ Executing: find . -name "*.py"
✅ Command completed (exit code: 0)

📤 Output:
./src/main.py
./tests/test_agent.py
./examples/demo.py

📚 Command saved to Atuin history
```

### Natural Language Examples

The agent understands various natural language patterns:

- "list files in current directory" → `ls -la`
- "find Python files" → `find . -name "*.py"`
- "show git status" → `git status`
- "count lines in README" → `wc -l README.md`
- "compress the project folder" → `tar -czf project.tar.gz project/`

## Advanced Usage

### Custom Server Implementation

```python
class CustomServer(MCPServer):
    """Custom MCP server implementation."""
    
    custom_param: str = Field(..., description="Custom parameter")
    
    def _get_connection_type(self) -> str:
        return "custom"
    
    def _generate_server_id(self) -> str:
        return f"custom_{self.custom_param}_{id(self)}"
    
    async def connect(self) -> None:
        # Custom connection logic
        if self._connection:
            self._connection.status = "connected"
    
    async def disconnect(self) -> None:
        # Custom disconnection logic
        if self._connection:
            self._connection.status = "disconnected"
    
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        # Custom tool execution
        return f"custom_result_{tool_name}"
    
    async def _discover_tools_impl(self) -> Dict[str, MCPTool]:
        # Custom tool discovery
        return {}

# Usage
class MyAgent(Agent):
    instruction: str = "Agent with custom server"
    custom: CustomServer
```

### Dynamic Agent Creation

```python
from delegant import Delegant

# Create agent dynamically
agent = Delegant.create_agent(
    "File analysis agent",
    files={"root_dir": "/data", "readonly": True},
    search={"provider": "duckduckgo", "max_results": 20}
)

# Or use convenience functions
file_agent = create_file_agent(root_dir="/documents")
search_agent = create_search_agent(provider="google", api_key="key")
research_agent = create_research_agent(
    search_provider="bing",
    file_root="/research"
)
```

### Connection Pool Usage

```python
from delegant import ConnectionPool, pooled_connection

# Global pool configuration
async with ConnectionPool(max_size=100, cleanup_interval=300) as pool:
    # Get connections manually
    server1 = await pool.get_connection(FileSystemServer, root_dir="/data")
    await pool.return_connection(server1)
    
    # Or use context manager
    async with pooled_connection(WebSearchServer, provider="duckduckgo") as server:
        results = await server.search("AI research")
```

### Advanced Retry Configuration

```python
from delegant import retry_with_backoff, RetryStrategy

# Custom retry strategies
@retry_with_backoff(
    max_attempts=10,
    base_delay=0.5,
    strategy=RetryStrategy.FIBONACCI,
    retryable_exceptions=[ConnectionError, TimeoutError]
)
async def robust_operation():
    # Operation with advanced retry logic
    pass

# Retry context manager
from delegant import RetryContext

async with RetryContext(max_attempts=5, timeout=30.0) as retry:
    result = await retry.execute(unreliable_operation, arg1, arg2)
```

## Configuration

### Global Configuration

```python
from delegant import configure, get_config, ConfigContext

# Set global configuration
configure(
    auto_retry=True,
    max_retries=5,
    retry_backoff=2.0,
    connection_timeout=60,
    lazy_connect=True,
    context_extraction=True,
    debug_mode=True,
    connection_pool_size=50,
    enable_metrics=False
)

# Get current configuration
config = get_config()
print(f"Max retries: {config.max_retries}")

# Temporary configuration changes
with ConfigContext(debug_mode=False, max_retries=1):
    # Configuration changed temporarily
    pass
# Configuration restored
```

### Environment Variables

```bash
# Override configuration via environment variables
export DELEGANT_MAX_RETRIES=10
export DELEGANT_CONNECTION_TIMEOUT=120
export DELEGANT_DEBUG_MODE=true
export DELEGANT_API_KEY_GOOGLE=your_google_api_key
export DELEGANT_API_KEY_BING=your_bing_api_key
```

### Server-Specific Configuration

```python
class ConfiguredAgent(Agent):
    instruction: str = "Pre-configured agent"
    
    def __init__(self, **kwargs):
        # Configure servers before instantiation
        self.configure_server("files", 
            root_dir="/custom/path",
            max_file_size=50 * 1024 * 1024,
            readonly=True
        )
        
        self.configure_server("search",
            provider="google",
            api_key=get_config().get_api_key("google"),
            max_results=50
        )
        
        super().__init__(**kwargs)
    
    files: FileSystemServer
    search: WebSearchServer
```

## Error Handling

### Exception Hierarchy

```python
from delegant import (
    DelegantException,          # Base exception
    ServerConnectionError,      # Connection failures
    ToolDiscoveryError,        # Tool discovery issues
    ToolExecutionError,        # Tool execution failures
    ContextExtractionError,    # Context extraction problems
    WorkflowExecutionError,    # Workflow failures
    ValidationError,           # Input validation errors
    ConfigurationError,        # Configuration issues
    RetryExhaustedError       # Retry exhaustion
)

try:
    result = await agent.files.read_file("nonexistent.txt")
except ToolExecutionError as e:
    print(f"Tool execution failed: {e}")
    print(f"Error context: {e.context}")
    print(f"Original error: {e.original_error}")
except ServerConnectionError as e:
    print(f"Server connection failed: {e}")
except DelegantException as e:
    print(f"Delegant error: {e}")
```

### Error Context and Debugging

All Delegant exceptions include rich context information:

```python
try:
    await problematic_operation()
except DelegantException as e:
    # Rich error information
    print(f"Message: {e.message}")
    print(f"Context: {e.context}")
    print(f"Original error: {e.original_error}")
    
    # Error severity for logging/monitoring
    if hasattr(e, 'severity'):
        print(f"Severity: {e.severity}")
```

## Performance

### Benchmarks

Delegant is designed for production use with performance targets:

- **Agent Creation**: <10ms per agent
- **Context Extraction**: <5ms per class
- **Connection Pool**: <1ms per connection request
- **Tool Execution**: <100ms overhead
- **Memory Usage**: <50MB for typical applications

### Optimization Tips

1. **Use Connection Pooling**: Reuse server connections
2. **Enable Lazy Connection**: Connect only when needed
3. **Configure Cache Settings**: Cache search results and context
4. **Limit Context Size**: Use reasonable context extraction limits
5. **Batch Operations**: Group multiple tool calls when possible

```python
# Optimized configuration
configure(
    lazy_connect=True,
    connection_pool_size=100,
    max_context_size=32768,  # 32KB limit
    enable_caching=True
)

# Use connection pooling
async with pooled_connection(FileSystemServer, root_dir="/data") as server:
    # Multiple operations on same connection
    files = await server.list_directory(".")
    for file_info in files:
        if file_info["type"] == "file":
            content = await server.read_file(file_info["path"])
```

## Examples

### Research Assistant

```python
@chain(WebSearchServer, FileSystemServer)
class ResearchAssistant(Agent):
    """AI research assistant with search and file capabilities."""
    
    instruction = """
    I help with research by searching for information online and 
    managing research documents locally.
    """
    
    async def research_topic(self, topic: str, save_results: bool = True) -> dict:
        """Research a topic and optionally save results."""
        # Execute search and file chain
        result = await self.execute_chain("search_and_save", topic, save_results)
        return result

# Usage
assistant = ResearchAssistant()
research = await assistant.research_topic("quantum computing", save_results=True)
```

### Development Helper

```python
class DevAgent(Agent):
    """Development helper agent."""
    
    instruction = "I help with development tasks"
    
    files: FileSystemServer
    terminal: TerminalServer
    
    async def run_tests(self, test_pattern: str = "test_*.py") -> dict:
        """Run tests matching pattern."""
        # Find test files
        test_files = await self.files.search_files(test_pattern)
        
        # Run pytest
        result = await self.terminal.execute_command(f"pytest {test_pattern}")
        
        return {
            "test_files": len(test_files),
            "success": result["success"],
            "output": result["stdout"]
        }
    
    async def check_code_quality(self, path: str = ".") -> dict:
        """Check code quality with multiple tools."""
        results = {}
        
        # Run multiple checks in parallel
        commands = [
            "flake8 .",
            "mypy .", 
            "bandit -r .",
            "black --check ."
        ]
        
        for cmd in commands:
            try:
                result = await self.terminal.execute_command(cmd)
                results[cmd.split()[0]] = {
                    "success": result["success"],
                    "issues": len(result["stderr"].split("\n")) if result["stderr"] else 0
                }
            except Exception as e:
                results[cmd.split()[0]] = {"error": str(e)}
        
        return results

# Usage
dev = DevAgent()
test_results = await dev.run_tests()
quality_check = await dev.check_code_quality()
```

### Data Analysis Pipeline

```python
@orchestrator({
    "search": WebSearchServer,
    "files": FileSystemServer, 
    "terminal": TerminalServer
})
class DataPipeline(Agent):
    """Data analysis pipeline with orchestration."""
    
    def setup_pipeline(self, output_dir: str):
        """Setup the data pipeline workflow."""
        # Step 1: Search for data sources
        self.add_orchestration_step(
            "search", "search_data_sources",
            depends_on=[]
        )
        
        # Step 2: Download and prepare data
        self.add_orchestration_step(
            "files", "prepare_data", 
            depends_on=["step_0"]
        )
        
        # Step 3: Run analysis
        self.add_orchestration_step(
            "terminal", "run_analysis",
            depends_on=["step_1"]
        )
        
        # Step 4: Generate report
        self.add_orchestration_step(
            "files", "generate_report",
            depends_on=["step_2"]
        )
    
    async def analyze_dataset(self, topic: str, output_dir: str) -> dict:
        """Run complete data analysis pipeline."""
        self.setup_pipeline(output_dir)
        return await self.execute_orchestration(topic, output_dir)

# Usage
pipeline = DataPipeline()
analysis = await pipeline.analyze_dataset("stock prices", "/analysis/output")
```

## Contributing

### Development Setup

```bash
git clone https://github.com/delegant/delegant.git
cd delegant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -e .[dev,docs,all]

# Run tests
python -m pytest tests/ -v --cov=delegant

# Run linting
ruff check delegant/
mypy delegant/

# Build documentation
mkdocs serve
```

### Code Style

- Use **Google-style docstrings** for all classes and methods
- **Type hints required** for all public APIs
- **Async/await patterns** throughout for consistency
- Follow **PEP 8** with line length of 88 characters
- Use **Ruff** for linting and **MyPy** for type checking

### Testing

- **90%+ test coverage** required for all new code
- Write **unit tests** for individual components
- Include **integration tests** for complete workflows
- Add **performance tests** for critical paths
- Use **pytest** with async support

### Documentation

- Update **API documentation** for any public interface changes
- Add **examples** for new features
- Update **CHANGELOG.md** with version history
- Include **type stubs** for better IDE support

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: https://delegant.readthedocs.io
- **Issues**: https://github.com/delegant/delegant/issues
- **Discussions**: https://github.com/delegant/delegant/discussions
- **Discord**: https://discord.gg/delegant

---

*Built with ❤️ by the Delegant team*