"""
Delegant FileSystem Server Implementation
========================================

MCP server implementation for file system operations with context-aware functionality.
Provides secure file operations with configurable restrictions and comprehensive logging.
"""

import os
import asyncio
import aiofiles
import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import Field, field_validator
import logging

from ..server import MCPServer, MCPTool
from ..exceptions import ToolExecutionError, ConfigurationError
from ..config import get_config

logger = logging.getLogger(__name__)


class FileSystemServer(MCPServer):
    """Server for file system operations with context-aware functionality.
    
    Provides secure file operations including read, write, list, and search
    with configurable restrictions and comprehensive error handling.
    
    Example:
        server = FileSystemServer(
            root_dir="/project/files",
            allowed_extensions=[".txt", ".md", ".py"],
            max_file_size=10 * 1024 * 1024  # 10MB
        )
        
        content = await server.read_file("document.txt")
        await server.write_file("output.txt", "Hello, World!")
    """
    
    root_dir: Optional[str] = Field(
        None, 
        description="Root directory for operations (uses current dir if None)"
    )
    allowed_extensions: Optional[List[str]] = Field(
        None, 
        description="Allowed file extensions (allows all if None)"
    )
    max_file_size: Optional[int] = Field(
        None, 
        description="Maximum file size in bytes (no limit if None)"
    )
    readonly: bool = Field(
        default=False, 
        description="Enable read-only mode (no write operations)"
    )
    enable_search: bool = Field(
        default=True, 
        description="Enable file content search functionality"
    )
    
    @field_validator('root_dir')
    @classmethod
    def validate_root_dir(cls, v: Optional[str]) -> Optional[str]:
        """Validate root directory exists and is accessible."""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Root directory does not exist: {v}")
            if not path.is_dir():
                raise ValueError(f"Root path is not a directory: {v}")
            if not os.access(path, os.R_OK):
                raise ValueError(f"Root directory is not readable: {v}")
        return v
    
    @field_validator('allowed_extensions')
    @classmethod
    def validate_extensions(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate file extensions format."""
        if v is not None:
            for ext in v:
                if not ext.startswith('.'):
                    raise ValueError(f"Extension must start with dot: {ext}")
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Set default root directory if not provided
        if self.root_dir is None:
            self.root_dir = os.getcwd()
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def _get_connection_type(self) -> str:
        return "filesystem"
    
    def _generate_server_id(self) -> str:
        return f"fs_{hashlib.md5(str(self.root_dir).encode()).hexdigest()[:8]}"
    
    async def connect(self) -> None:
        """Establish filesystem connection (verify access)."""
        if self._connection:
            self._connection.status = "connecting"
        
        try:
            # Verify root directory access
            root_path = Path(self.root_dir)
            if not root_path.exists():
                raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
            
            if not root_path.is_dir():
                raise NotADirectoryError(f"Root path is not a directory: {self.root_dir}")
            
            # Test read access
            if not os.access(root_path, os.R_OK):
                raise PermissionError(f"No read access to root directory: {self.root_dir}")
            
            # Test write access if not readonly
            if not self.readonly and not os.access(root_path, os.W_OK):
                raise PermissionError(f"No write access to root directory: {self.root_dir}")
            
            if self._connection:
                self._connection.status = "connected"
                self._connection.connection_url = str(root_path.absolute())
                self._connection.last_activity = datetime.now()
            
            logger.info(f"Connected to filesystem: {self.root_dir}")
            
        except Exception as e:
            if self._connection:
                self._connection.status = "error"
            raise e
    
    async def disconnect(self) -> None:
        """Disconnect from filesystem (cleanup)."""
        if self._connection:
            self._connection.status = "disconnected"
        logger.info("Disconnected from filesystem")
    
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute filesystem tool."""
        if tool_name not in self._tools:
            raise ToolExecutionError(
                tool_name=tool_name,
                server_name=self._connection.server_id if self._connection else "filesystem",
                parameters=parameters,
                original_error=ValueError(f"Unknown tool: {tool_name}")
            )
        
        tool = self._tools[tool_name]
        return await tool.execute(**parameters)
    
    async def _discover_tools_impl(self) -> Dict[str, MCPTool]:
        """Discover filesystem tools."""
        return self._tools.copy()
    
    def _register_builtin_tools(self) -> None:
        """Register built-in filesystem tools."""
        
        # Read file tool
        self.register_tool(MCPTool(
            name="read_file",
            description="Read the contents of a file",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8)",
                        "default": "utf-8"
                    }
                },
                "required": ["path"]
            },
            execution_func=self.read_file
        ))
        
        # Write file tool
        if not self.readonly:
            self.register_tool(MCPTool(
                name="write_file",
                description="Write content to a file",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        },
                        "encoding": {
                            "type": "string",
                            "description": "File encoding (default: utf-8)",
                            "default": "utf-8"
                        },
                        "create_dirs": {
                            "type": "boolean",
                            "description": "Create parent directories if they don't exist",
                            "default": False
                        }
                    },
                    "required": ["path", "content"]
                },
                execution_func=self.write_file
            ))
        
        # List directory tool
        self.register_tool(MCPTool(
            name="list_directory",
            description="List contents of a directory",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to list",
                        "default": "."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively",
                        "default": False
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files/directories",
                        "default": False
                    }
                }
            },
            execution_func=self.list_directory
        ))
        
        # File info tool
        self.register_tool(MCPTool(
            name="get_file_info",
            description="Get information about a file or directory",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory"
                    }
                },
                "required": ["path"]
            },
            execution_func=self.get_file_info
        ))
        
        # Search files tool
        if self.enable_search:
            self.register_tool(MCPTool(
                name="search_files",
                description="Search for files and content",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "search_content": {
                            "type": "boolean",
                            "description": "Search file contents (not just names)",
                            "default": False
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search in",
                            "default": "."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 50
                        }
                    },
                    "required": ["query"]
                },
                execution_func=self.search_files
            ))
        
        # Delete file tool (if not readonly)
        if not self.readonly:
            self.register_tool(MCPTool(
                name="delete_file",
                description="Delete a file or directory",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file or directory to delete"
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Delete directories recursively",
                            "default": False
                        }
                    },
                    "required": ["path"]
                },
                execution_func=self.delete_file
            ))
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve and validate file path within root directory."""
        # Convert to Path object
        path_obj = Path(path)
        
        # If absolute path, make sure it's within root_dir
        if path_obj.is_absolute():
            root_path = Path(self.root_dir).resolve()
            resolved_path = path_obj.resolve()
            
            # Check if the path is within root directory
            try:
                resolved_path.relative_to(root_path)
            except ValueError:
                raise PermissionError(f"Access denied: path outside root directory: {path}")
        else:
            # Relative path - resolve relative to root_dir
            root_path = Path(self.root_dir).resolve()
            resolved_path = (root_path / path_obj).resolve()
            
            # Check if resolved path is still within root directory
            try:
                resolved_path.relative_to(root_path)
            except ValueError:
                raise PermissionError(f"Access denied: path outside root directory: {path}")
        
        return resolved_path
    
    def _check_file_extension(self, path: Path) -> None:
        """Check if file extension is allowed."""
        if self.allowed_extensions is not None:
            if path.suffix.lower() not in [ext.lower() for ext in self.allowed_extensions]:
                raise PermissionError(f"File extension not allowed: {path.suffix}")
    
    def _check_file_size(self, path: Path) -> None:
        """Check if file size is within limits."""
        if self.max_file_size is not None and path.exists():
            size = path.stat().st_size
            if size > self.max_file_size:
                raise ValueError(f"File size ({size} bytes) exceeds limit ({self.max_file_size} bytes)")
    
    async def read_file(self, path: str, encoding: str = "utf-8") -> str:
        """Read file content with context about operation purpose.
        
        Args:
            path: Path to the file to read
            encoding: File encoding (default: utf-8)
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file access is denied
            UnicodeDecodeError: If file can't be decoded with specified encoding
        """
        try:
            resolved_path = self._resolve_path(path)
            self._check_file_extension(resolved_path)
            self._check_file_size(resolved_path)
            
            if not resolved_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            if not resolved_path.is_file():
                raise IsADirectoryError(f"Path is a directory, not a file: {path}")
            
            async with aiofiles.open(resolved_path, 'r', encoding=encoding) as f:
                content = await f.read()
            
            logger.info(f"Read file: {path} ({len(content)} characters)")
            return content
            
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            raise
    
    async def write_file(
        self, 
        path: str, 
        content: str, 
        encoding: str = "utf-8",
        create_dirs: bool = False
    ) -> bool:
        """Write file content with contextual metadata.
        
        Args:
            path: Path to the file to write
            content: Content to write
            encoding: File encoding (default: utf-8)
            create_dirs: Create parent directories if they don't exist
            
        Returns:
            True if successful
            
        Raises:
            PermissionError: If readonly mode or access denied
            ValueError: If content size exceeds limits
        """
        if self.readonly:
            raise PermissionError("Server is in readonly mode")
        
        try:
            resolved_path = self._resolve_path(path)
            self._check_file_extension(resolved_path)
            
            # Check content size
            content_bytes = content.encode(encoding)
            if self.max_file_size is not None and len(content_bytes) > self.max_file_size:
                raise ValueError(f"Content size ({len(content_bytes)} bytes) exceeds limit ({self.max_file_size} bytes)")
            
            # Create parent directories if requested
            if create_dirs:
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(resolved_path, 'w', encoding=encoding) as f:
                await f.write(content)
            
            logger.info(f"Wrote file: {path} ({len(content)} characters)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            raise
    
    async def list_directory(
        self, 
        path: str = ".", 
        recursive: bool = False,
        include_hidden: bool = False
    ) -> List[Dict[str, Any]]:
        """List directory contents with file metadata.
        
        Args:
            path: Directory path to list
            recursive: List recursively
            include_hidden: Include hidden files/directories
            
        Returns:
            List of file/directory information dictionaries
        """
        try:
            resolved_path = self._resolve_path(path)
            
            if not resolved_path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")
            
            if not resolved_path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {path}")
            
            results = []
            
            if recursive:
                # Recursive listing
                for item in resolved_path.rglob("*"):
                    if not include_hidden and item.name.startswith('.'):
                        continue
                    
                    results.append(await self._get_file_info_dict(item))
            else:
                # Non-recursive listing
                for item in resolved_path.iterdir():
                    if not include_hidden and item.name.startswith('.'):
                        continue
                    
                    results.append(await self._get_file_info_dict(item))
            
            # Sort by name
            results.sort(key=lambda x: x['name'])
            
            logger.info(f"Listed directory: {path} ({len(results)} items)")
            return results
            
        except Exception as e:
            logger.error(f"Failed to list directory {path}: {e}")
            raise
    
    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get detailed information about a file or directory.
        
        Args:
            path: Path to the file or directory
            
        Returns:
            Dictionary with file information
        """
        try:
            resolved_path = self._resolve_path(path)
            
            if not resolved_path.exists():
                raise FileNotFoundError(f"Path not found: {path}")
            
            return await self._get_file_info_dict(resolved_path)
            
        except Exception as e:
            logger.error(f"Failed to get file info for {path}: {e}")
            raise
    
    async def _get_file_info_dict(self, path: Path) -> Dict[str, Any]:
        """Get file information as dictionary."""
        stat = path.stat()
        
        info = {
            "name": path.name,
            "path": str(path.relative_to(self.root_dir)),
            "absolute_path": str(path),
            "type": "directory" if path.is_dir() else "file",
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
            "is_hidden": path.name.startswith('.')
        }
        
        if path.is_file():
            # Add file-specific information
            info["extension"] = path.suffix
            info["mime_type"] = mimetypes.guess_type(str(path))[0]
            
            # Add hash for small files
            if stat.st_size < 1024 * 1024:  # Less than 1MB
                try:
                    with open(path, 'rb') as f:
                        content = f.read()
                        info["md5_hash"] = hashlib.md5(content).hexdigest()
                except:
                    pass  # Skip hash if can't read file
        
        return info
    
    async def search_files(
        self, 
        query: str,
        search_content: bool = False,
        path: str = ".",
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """Search for files and optionally their content.
        
        Args:
            query: Search query
            search_content: Search file contents (not just names)
            path: Directory to search in
            max_results: Maximum number of results
            
        Returns:
            List of matching files with context
        """
        if not self.enable_search:
            raise PermissionError("Search functionality is disabled")
        
        try:
            resolved_path = self._resolve_path(path)
            
            if not resolved_path.exists():
                raise FileNotFoundError(f"Search directory not found: {path}")
            
            if not resolved_path.is_dir():
                raise NotADirectoryError(f"Search path is not a directory: {path}")
            
            results = []
            query_lower = query.lower()
            
            for item in resolved_path.rglob("*"):
                if len(results) >= max_results:
                    break
                
                # Skip hidden files unless specifically searching for them
                if item.name.startswith('.') and not query.startswith('.'):
                    continue
                
                match_info = {
                    "file": await self._get_file_info_dict(item),
                    "match_type": None,
                    "match_context": None
                }
                
                # Search filename
                if query_lower in item.name.lower():
                    match_info["match_type"] = "filename"
                    match_info["match_context"] = f"Filename contains: {query}"
                    results.append(match_info)
                    continue
                
                # Search file content if requested and it's a text file
                if search_content and item.is_file():
                    try:
                        # Only search text files
                        mime_type = mimetypes.guess_type(str(item))[0]
                        if mime_type and mime_type.startswith('text/'):
                            # Check file size limit for content search
                            if item.stat().st_size > 1024 * 1024:  # Skip files > 1MB
                                continue
                            
                            async with aiofiles.open(item, 'r', encoding='utf-8', errors='ignore') as f:
                                content = await f.read()
                                
                            if query_lower in content.lower():
                                # Find context around the match
                                match_index = content.lower().find(query_lower)
                                start = max(0, match_index - 50)
                                end = min(len(content), match_index + len(query) + 50)
                                context = content[start:end].strip()
                                
                                match_info["match_type"] = "content"
                                match_info["match_context"] = f"...{context}..."
                                results.append(match_info)
                    except:
                        # Skip files that can't be read as text
                        continue
            
            logger.info(f"Search completed: {query} in {path} ({len(results)} results)")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}' in {path}: {e}")
            raise
    
    async def delete_file(self, path: str, recursive: bool = False) -> bool:
        """Delete a file or directory.
        
        Args:
            path: Path to delete
            recursive: Delete directories recursively
            
        Returns:
            True if successful
            
        Raises:
            PermissionError: If readonly mode or access denied
        """
        if self.readonly:
            raise PermissionError("Server is in readonly mode")
        
        try:
            resolved_path = self._resolve_path(path)
            
            if not resolved_path.exists():
                raise FileNotFoundError(f"Path not found: {path}")
            
            if resolved_path.is_dir():
                if recursive:
                    import shutil
                    shutil.rmtree(resolved_path)
                else:
                    resolved_path.rmdir()  # Only works if directory is empty
            else:
                resolved_path.unlink()
            
            logger.info(f"Deleted: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    async def test_filesystem_server():
        # Create test server
        server = FileSystemServer(
            root_dir="/tmp/delegant_test",
            allowed_extensions=[".txt", ".md", ".py"],
            max_file_size=1024 * 1024,  # 1MB
            enable_search=True
        )
        
        async with server:
            # Test write
            await server.write_file("test.txt", "Hello, World!", create_dirs=True)
            
            # Test read
            content = await server.read_file("test.txt")
            print(f"Read content: {content}")
            
            # Test list directory
            files = await server.list_directory(".")
            print(f"Directory contents: {len(files)} items")
            
            # Test search
            results = await server.search_files("Hello", search_content=True)
            print(f"Search results: {len(results)} matches")
            
            # Test file info
            info = await server.get_file_info("test.txt")
            print(f"File info: {info}")
    
    # Run test
    # asyncio.run(test_filesystem_server())
