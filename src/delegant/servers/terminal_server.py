"""
Delegant Terminal Server Implementation
======================================

MCP server implementation for terminal command execution with comprehensive logging,
devenv.sh container support, and security features for safe command execution.
"""

import asyncio
import json
import os
import shlex
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import Field, field_validator
import logging

from ..server import ProcessMCPServer, MCPTool
from ..exceptions import ToolExecutionError, ConfigurationError
from ..config import get_config

logger = logging.getLogger(__name__)


class CommandResult:
    """Structured result from command execution."""
    
    def __init__(
        self,
        command: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        execution_time: float,
        working_directory: str,
        environment: Dict[str, str],
        timestamp: datetime
    ):
        self.command = command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.execution_time = execution_time
        self.working_directory = working_directory
        self.environment = environment
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time": self.execution_time,
            "working_directory": self.working_directory,
            "environment": dict(self.environment),
            "timestamp": self.timestamp.isoformat(),
            "success": self.exit_code == 0
        }
    
    def __str__(self) -> str:
        """String representation of the command result."""
        status = "SUCCESS" if self.exit_code == 0 else f"FAILED (exit code: {self.exit_code})"
        return f"Command: {self.command}\nStatus: {status}\nTime: {self.execution_time:.2f}s\nOutput: {self.stdout[:200]}..."


class TerminalServer(ProcessMCPServer):
    """Server for terminal command execution with context-aware functionality.
    
    Provides secure command execution with configurable restrictions, comprehensive
    logging, and support for devenv.sh containers and other shell environments.
    
    Example:
        server = TerminalServer(
            shell="/bin/bash",
            working_directory="/project",
            allowed_commands=["git", "ls", "cat", "grep"],
            log_all_commands=True
        )
        
        result = await server.execute_command("ls -la")
        history = await server.get_command_history()
    """
    
    shell: str = Field(
        default="/bin/bash",
        description="Shell to use for command execution"
    )
    working_directory: Optional[str] = Field(
        None,
        description="Default working directory for commands"
    )
    allowed_commands: Optional[List[str]] = Field(
        None,
        description="Allowed command prefixes (None allows all)"
    )
    blocked_commands: List[str] = Field(
        default_factory=lambda: ["rm -rf", "sudo rm", "format", "fdisk"],
        description="Blocked dangerous commands"
    )
    timeout_seconds: int = Field(
        default=300,
        description="Command timeout in seconds",
        ge=1,
        le=3600
    )
    log_all_commands: bool = Field(
        default=True,
        description="Log all command executions"
    )
    log_directory: Optional[str] = Field(
        None,
        description="Directory to store command logs"
    )
    use_devenv: bool = Field(
        default=False,
        description="Execute commands in devenv.sh container"
    )
    devenv_config: Optional[str] = Field(
        None,
        description="Path to devenv configuration file"
    )
    max_output_size: int = Field(
        default=1024 * 1024,  # 1MB
        description="Maximum output size in bytes",
        ge=1024
    )
    preserve_environment: bool = Field(
        default=True,
        description="Preserve current environment variables"
    )
    
    # Private attributes
    _command_history: List[CommandResult] = []
    _log_file: Optional[Path] = None
    
    @field_validator('shell')
    @classmethod
    def validate_shell(cls, v: str) -> str:
        """Validate shell exists and is executable."""
        if not Path(v).exists():
            raise ValueError(f"Shell not found: {v}")
        if not os.access(v, os.X_OK):
            raise ValueError(f"Shell not executable: {v}")
        return v
    
    def __init__(self, **data):
        # Set up command for process-based server (we'll override the execution)
        if 'command' not in data:
            data['command'] = [data.get('shell', '/bin/bash')]
        
        super().__init__(**data)
        
        # Set up working directory
        if self.working_directory is None:
            self.working_directory = os.getcwd()
        
        # Set up logging
        self._setup_logging()
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def _setup_logging(self) -> None:
        """Set up command logging."""
        if self.log_all_commands:
            if self.log_directory:
                log_dir = Path(self.log_directory)
                log_dir.mkdir(parents=True, exist_ok=True)
                self._log_file = log_dir / f"terminal_commands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            else:
                # Use a default log location
                log_dir = Path.home() / ".delegant" / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                self._log_file = log_dir / f"terminal_commands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    def _get_connection_type(self) -> str:
        return "terminal"
    
    def _generate_server_id(self) -> str:
        return f"terminal_{os.getpid()}_{id(self)}"
    
    async def connect(self) -> None:
        """Connect to terminal (validate shell and environment)."""
        if self._connection:
            self._connection.status = "connecting"
        
        try:
            # Validate shell
            if not Path(self.shell).exists():
                raise FileNotFoundError(f"Shell not found: {self.shell}")
            
            # Validate working directory
            if self.working_directory and not Path(self.working_directory).exists():
                raise FileNotFoundError(f"Working directory not found: {self.working_directory}")
            
            # Test shell execution
            test_result = await self._execute_raw_command("echo 'test'")
            if test_result.exit_code != 0:
                raise Exception(f"Shell test failed: {test_result.stderr}")
            
            # Initialize devenv if requested
            if self.use_devenv:
                await self._initialize_devenv()
            
            if self._connection:
                self._connection.status = "connected"
                self._connection.connection_url = f"{self.shell}@{self.working_directory}"
                self._connection.last_activity = datetime.now()
            
            logger.info(f"Connected to terminal: {self.shell}")
            
        except Exception as e:
            if self._connection:
                self._connection.status = "error"
            raise e
    
    async def disconnect(self) -> None:
        """Disconnect from terminal."""
        if self._connection:
            self._connection.status = "disconnected"
        logger.info("Disconnected from terminal")
    
    async def _initialize_devenv(self) -> None:
        """Initialize devenv.sh environment."""
        try:
            # Check if devenv is available
            devenv_check = await self._execute_raw_command("which devenv")
            if devenv_check.exit_code != 0:
                raise ConfigurationError(
                    config_source="devenv_setup",
                    suggested_fix="Install devenv.sh: https://devenv.sh/getting-started/"
                )
            
            # Initialize devenv if config exists
            if self.devenv_config and Path(self.devenv_config).exists():
                init_result = await self._execute_raw_command(f"devenv shell --config {self.devenv_config}")
                if init_result.exit_code != 0:
                    logger.warning(f"Failed to initialize devenv: {init_result.stderr}")
            
            logger.info("Devenv environment initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize devenv: {e}")
            raise
    
    def _register_builtin_tools(self) -> None:
        """Register built-in terminal tools."""
        
        # Execute command tool
        self.register_tool(MCPTool(
            name="execute_command",
            description="Execute a terminal command",
            parameters_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute"
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Working directory for the command",
                        "default": self.working_directory
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Command timeout in seconds",
                        "default": self.timeout_seconds,
                        "minimum": 1,
                        "maximum": 3600
                    },
                    "capture_output": {
                        "type": "boolean",
                        "description": "Capture and return command output",
                        "default": True
                    }
                },
                "required": ["command"]
            },
            execution_func=self.execute_command
        ))
        
        # Get command history tool
        self.register_tool(MCPTool(
            name="get_command_history",
            description="Get command execution history",
            parameters_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of commands to return",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 1000
                    },
                    "filter_command": {
                        "type": "string",
                        "description": "Filter commands containing this text"
                    },
                    "only_successful": {
                        "type": "boolean",
                        "description": "Only return successful commands",
                        "default": False
                    }
                }
            },
            execution_func=self.get_command_history
        ))
        
        # Parse command tool
        self.register_tool(MCPTool(
            name="parse_command",
            description="Parse and validate a command before execution",
            parameters_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to parse and validate"
                    },
                    "check_safety": {
                        "type": "boolean",
                        "description": "Check if command is safe to execute",
                        "default": True
                    }
                },
                "required": ["command"]
            },
            execution_func=self.parse_command
        ))
        
        # Get help for command tool
        self.register_tool(MCPTool(
            name="get_command_help",
            description="Get help/manual for a command",
            parameters_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to get help for"
                    },
                    "help_type": {
                        "type": "string",
                        "description": "Type of help (man, help, info)",
                        "default": "man"
                    }
                },
                "required": ["command"]
            },
            execution_func=self.get_command_help
        ))
        
        # Environment info tool
        self.register_tool(MCPTool(
            name="get_environment_info",
            description="Get information about the current environment",
            parameters_schema={
                "type": "object",
                "properties": {
                    "include_variables": {
                        "type": "boolean",
                        "description": "Include environment variables",
                        "default": False
                    }
                }
            },
            execution_func=self.get_environment_info
        ))
    
    def _validate_command_safety(self, command: str) -> None:
        """Validate command is safe to execute."""
        command_lower = command.lower().strip()
        
        # Check blocked commands
        for blocked in self.blocked_commands:
            if blocked.lower() in command_lower:
                raise PermissionError(f"Blocked dangerous command pattern: {blocked}")
        
        # Check allowed commands if specified
        if self.allowed_commands:
            command_parts = shlex.split(command)
            if command_parts:
                base_command = command_parts[0]
                
                # Check if base command or any prefix is allowed
                allowed = False
                for allowed_cmd in self.allowed_commands:
                    if base_command.startswith(allowed_cmd) or allowed_cmd in base_command:
                        allowed = True
                        break
                
                if not allowed:
                    raise PermissionError(f"Command not in allowed list: {base_command}")
    
    async def _execute_raw_command(
        self,
        command: str,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
        env_override: Optional[Dict[str, str]] = None
    ) -> CommandResult:
        """Execute a raw command and return structured result."""
        start_time = time.time()
        timestamp = datetime.now()
        timeout = timeout or self.timeout_seconds
        work_dir = working_directory or self.working_directory
        
        # Prepare environment
        env = dict(os.environ) if self.preserve_environment else {}
        if env_override:
            env.update(env_override)
        
        # Prepare devenv command if needed
        if self.use_devenv:
            if self.devenv_config:
                command = f"devenv shell --config {self.devenv_config} -c '{command}'"
            else:
                command = f"devenv shell -c '{command}'"
        
        try:
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env=env
            )
            
            # Wait for completion with timeout
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Command timed out after {timeout} seconds")
            
            execution_time = time.time() - start_time
            
            # Decode output
            stdout = stdout_data.decode('utf-8', errors='replace')
            stderr = stderr_data.decode('utf-8', errors='replace')
            
            # Truncate output if too large
            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n... [output truncated]"
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n... [output truncated]"
            
            result = CommandResult(
                command=command,
                exit_code=process.returncode or 0,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                working_directory=work_dir,
                environment=env,
                timestamp=timestamp
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                working_directory=work_dir,
                environment=env,
                timestamp=timestamp
            )
            
            return result
    
    def _log_command_result(self, result: CommandResult) -> None:
        """Log command result to file."""
        if not self.log_all_commands or not self._log_file:
            return
        
        try:
            log_entry = result.to_dict()
            with open(self._log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.warning(f"Failed to log command result: {e}")
    
    async def execute_command(
        self,
        command: str,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
        capture_output: bool = True
    ) -> Dict[str, Any]:
        """Execute a terminal command with comprehensive logging.
        
        Args:
            command: Command to execute
            working_directory: Working directory for the command
            timeout: Command timeout in seconds
            capture_output: Whether to capture and return output
            
        Returns:
            Command execution result
            
        Raises:
            PermissionError: If command is not allowed
            TimeoutError: If command times out
        """
        try:
            # Validate command safety
            self._validate_command_safety(command)
            
            # Execute command
            result = await self._execute_raw_command(
                command,
                working_directory,
                timeout
            )
            
            # Add to history
            self._command_history.append(result)
            
            # Log result
            self._log_command_result(result)
            
            # Prepare return data
            response = result.to_dict()
            
            if not capture_output:
                # Remove output to save bandwidth
                response.pop('stdout', None)
                response.pop('stderr', None)
            
            logger.info(f"Executed command: {command} (exit code: {result.exit_code})")
            return response
            
        except Exception as e:
            logger.error(f"Command execution failed: {command} - {e}")
            raise ToolExecutionError(
                tool_name="execute_command",
                server_name=self._connection.server_id if self._connection else "terminal",
                parameters={"command": command},
                original_error=e
            )
    
    async def get_command_history(
        self,
        limit: int = 50,
        filter_command: Optional[str] = None,
        only_successful: bool = False
    ) -> List[Dict[str, Any]]:
        """Get command execution history with filtering.
        
        Args:
            limit: Maximum number of commands to return
            filter_command: Filter commands containing this text
            only_successful: Only return successful commands
            
        Returns:
            List of command history entries
        """
        try:
            filtered_history = self._command_history.copy()
            
            # Apply filters
            if filter_command:
                filtered_history = [
                    cmd for cmd in filtered_history
                    if filter_command.lower() in cmd.command.lower()
                ]
            
            if only_successful:
                filtered_history = [
                    cmd for cmd in filtered_history
                    if cmd.exit_code == 0
                ]
            
            # Sort by timestamp (most recent first)
            filtered_history.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply limit
            filtered_history = filtered_history[:limit]
            
            # Convert to dict format
            return [cmd.to_dict() for cmd in filtered_history]
            
        except Exception as e:
            logger.error(f"Failed to get command history: {e}")
            raise ToolExecutionError(
                tool_name="get_command_history",
                server_name=self._connection.server_id if self._connection else "terminal",
                original_error=e
            )
    
    async def parse_command(
        self,
        command: str,
        check_safety: bool = True
    ) -> Dict[str, Any]:
        """Parse and validate a command before execution.
        
        Args:
            command: Command to parse
            check_safety: Whether to check command safety
            
        Returns:
            Command parsing information
        """
        try:
            # Parse command into parts
            try:
                command_parts = shlex.split(command)
            except ValueError as e:
                return {
                    "valid": False,
                    "error": f"Invalid command syntax: {e}",
                    "command": command
                }
            
            if not command_parts:
                return {
                    "valid": False,
                    "error": "Empty command",
                    "command": command
                }
            
            base_command = command_parts[0]
            arguments = command_parts[1:] if len(command_parts) > 1 else []
            
            # Check if command exists
            which_result = await self._execute_raw_command(f"which {base_command}")
            command_exists = which_result.exit_code == 0
            
            result = {
                "valid": True,
                "command": command,
                "base_command": base_command,
                "arguments": arguments,
                "command_exists": command_exists,
                "command_path": which_result.stdout.strip() if command_exists else None
            }
            
            # Safety check if requested
            if check_safety:
                try:
                    self._validate_command_safety(command)
                    result["safe"] = True
                    result["safety_warnings"] = []
                except PermissionError as e:
                    result["safe"] = False
                    result["safety_warnings"] = [str(e)]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse command: {command} - {e}")
            raise ToolExecutionError(
                tool_name="parse_command",
                server_name=self._connection.server_id if self._connection else "terminal",
                parameters={"command": command},
                original_error=e
            )
    
    async def get_command_help(
        self,
        command: str,
        help_type: str = "man"
    ) -> Dict[str, Any]:
        """Get help/manual for a command.
        
        Args:
            command: Command to get help for
            help_type: Type of help (man, help, info)
            
        Returns:
            Help information for the command
        """
        try:
            help_commands = {
                "man": f"man {command}",
                "help": f"{command} --help",
                "info": f"info {command}"
            }
            
            if help_type not in help_commands:
                help_type = "man"
            
            help_command = help_commands[help_type]
            result = await self._execute_raw_command(help_command)
            
            return {
                "command": command,
                "help_type": help_type,
                "help_command": help_command,
                "success": result.exit_code == 0,
                "help_text": result.stdout if result.exit_code == 0 else result.stderr,
                "error": result.stderr if result.exit_code != 0 else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get help for command: {command} - {e}")
            raise ToolExecutionError(
                tool_name="get_command_help",
                server_name=self._connection.server_id if self._connection else "terminal",
                parameters={"command": command},
                original_error=e
            )
    
    async def get_environment_info(
        self,
        include_variables: bool = False
    ) -> Dict[str, Any]:
        """Get information about the current environment.
        
        Args:
            include_variables: Whether to include environment variables
            
        Returns:
            Environment information
        """
        try:
            info = {
                "shell": self.shell,
                "working_directory": self.working_directory,
                "use_devenv": self.use_devenv,
                "devenv_config": self.devenv_config
            }
            
            # Get system information
            uname_result = await self._execute_raw_command("uname -a")
            if uname_result.exit_code == 0:
                info["system_info"] = uname_result.stdout.strip()
            
            # Get shell version
            shell_version = await self._execute_raw_command(f"{self.shell} --version")
            if shell_version.exit_code == 0:
                info["shell_version"] = shell_version.stdout.strip().split('\n')[0]
            
            # Get current user
            whoami_result = await self._execute_raw_command("whoami")
            if whoami_result.exit_code == 0:
                info["current_user"] = whoami_result.stdout.strip()
            
            # Get current directory
            pwd_result = await self._execute_raw_command("pwd")
            if pwd_result.exit_code == 0:
                info["current_directory"] = pwd_result.stdout.strip()
            
            # Include environment variables if requested
            if include_variables:
                env_result = await self._execute_raw_command("env")
                if env_result.exit_code == 0:
                    env_vars = {}
                    for line in env_result.stdout.strip().split('\n'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key] = value
                    info["environment_variables"] = env_vars
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get environment info: {e}")
            raise ToolExecutionError(
                tool_name="get_environment_info",
                server_name=self._connection.server_id if self._connection else "terminal",
                original_error=e
            )
    
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute terminal tool."""
        if tool_name not in self._tools:
            raise ToolExecutionError(
                tool_name=tool_name,
                server_name=self._connection.server_id if self._connection else "terminal",
                parameters=parameters,
                original_error=ValueError(f"Unknown tool: {tool_name}")
            )
        
        tool = self._tools[tool_name]
        return await tool.execute(**parameters)
    
    async def _discover_tools_impl(self) -> Dict[str, MCPTool]:
        """Discover terminal tools."""
        return self._tools.copy()


# Example usage and testing
if __name__ == "__main__":
    async def test_terminal_server():
        # Create test server
        server = TerminalServer(
            shell="/bin/bash",
            working_directory="/tmp",
            allowed_commands=["ls", "echo", "cat", "grep", "find"],
            log_all_commands=True,
            timeout_seconds=30
        )
        
        async with server:
            # Test command execution
            result = await server.execute_command("echo 'Hello, World!'")
            print(f"Command result: {result}")
            
            # Test command parsing
            parse_result = await server.parse_command("ls -la")
            print(f"Parse result: {parse_result}")
            
            # Test command help
            help_result = await server.get_command_help("ls")
            print(f"Help result: {help_result['success']}")
            
            # Test environment info
            env_info = await server.get_environment_info()
            print(f"Environment: {env_info.get('system_info', 'Unknown')}")
            
            # Test command history
            history = await server.get_command_history(limit=5)
            print(f"Command history: {len(history)} entries")
    
    # Run test
    # asyncio.run(test_terminal_server())
