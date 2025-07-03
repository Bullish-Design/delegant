"""
Delegant Atuin Server Implementation
===================================

MCP server implementation for Atuin command history management with context-aware
functionality. Provides rich command history search, statistics, and integration.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import Field, field_validator
import sqlite3
import logging

from ..server import MCPServer, MCPTool
from ..exceptions import ToolExecutionError, ConfigurationError
from ..config import get_config

logger = logging.getLogger(__name__)


class AtuinHistoryEntry:
    """Structured Atuin history entry."""
    
    def __init__(self, **data):
        self.id = data.get('id')
        self.timestamp = data.get('timestamp')
        self.duration = data.get('duration', 0)
        self.exit = data.get('exit', 0)
        self.command = data.get('command', '')
        self.cwd = data.get('cwd', '')
        self.session = data.get('session', '')
        self.hostname = data.get('hostname', '')
        self.deleted_at = data.get('deleted_at')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'exit': self.exit,
            'command': self.command,
            'cwd': self.cwd,
            'session': self.session,
            'hostname': self.hostname,
            'deleted_at': self.deleted_at,
            'success': self.exit == 0,
            'formatted_timestamp': datetime.fromtimestamp(self.timestamp / 1_000_000_000).isoformat() if self.timestamp else None,
            'duration_ms': self.duration / 1_000_000 if self.duration else 0
        }


class AtuinServer(MCPServer):
    """Server for Atuin command history management with context awareness.
    
    Provides comprehensive command history search, statistics, and management
    through the Atuin command history tool integration.
    
    Example:
        server = AtuinServer(
            atuin_db_path="~/.local/share/atuin/history.db",
            enable_sync=True
        )
        
        history = await server.search_history("git")
        stats = await server.get_statistics()
        await server.import_command("ls -la", "/home/user", 0)
    """
    
    atuin_db_path: Optional[str] = Field(
        None,
        description="Path to Atuin database file (auto-detected if None)"
    )
    atuin_config_path: Optional[str] = Field(
        None,
        description="Path to Atuin config file (auto-detected if None)"
    )
    enable_sync: bool = Field(
        default=False,
        description="Enable Atuin sync functionality"
    )
    max_results: int = Field(
        default=100,
        description="Maximum results to return in searches",
        ge=1,
        le=10000
    )
    
    # Private attributes
    _db_connection: Optional[sqlite3.Connection] = None
    _atuin_available: bool = False
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Auto-detect Atuin paths if not provided
        if self.atuin_db_path is None:
            self.atuin_db_path = self._find_atuin_db()
        
        if self.atuin_config_path is None:
            self.atuin_config_path = self._find_atuin_config()
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def _find_atuin_db(self) -> Optional[str]:
        """Find Atuin database file in standard locations."""
        possible_paths = [
            Path.home() / ".local" / "share" / "atuin" / "history.db",
            Path.home() / ".config" / "atuin" / "history.db",
            Path.home() / "Library" / "Application Support" / "atuin" / "history.db",  # macOS
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _find_atuin_config(self) -> Optional[str]:
        """Find Atuin config file in standard locations."""
        possible_paths = [
            Path.home() / ".config" / "atuin" / "config.toml",
            Path.home() / ".atuin" / "config.toml",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _get_connection_type(self) -> str:
        return "atuin"
    
    def _generate_server_id(self) -> str:
        return f"atuin_{id(self)}"
    
    async def connect(self) -> None:
        """Connect to Atuin database and validate installation."""
        if self._connection:
            self._connection.status = "connecting"
        
        try:
            # Check if Atuin CLI is available
            import subprocess
            try:
                result = subprocess.run(['atuin', '--version'], 
                                     capture_output=True, text=True, timeout=10)
                self._atuin_available = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self._atuin_available = False
            
            # Connect to database if available
            if self.atuin_db_path and Path(self.atuin_db_path).exists():
                self._db_connection = sqlite3.connect(self.atuin_db_path)
                self._db_connection.row_factory = sqlite3.Row  # Enable column access by name
                
                # Test database connection
                cursor = self._db_connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM history LIMIT 1")
                cursor.fetchone()
                
                logger.info(f"Connected to Atuin database: {self.atuin_db_path}")
            else:
                logger.warning("Atuin database not found, some features will be limited")
            
            if self._connection:
                self._connection.status = "connected"
                self._connection.connection_url = self.atuin_db_path
                self._connection.last_activity = datetime.now()
            
        except Exception as e:
            if self._connection:
                self._connection.status = "error"
            raise e
    
    async def disconnect(self) -> None:
        """Disconnect from Atuin database."""
        if self._db_connection:
            self._db_connection.close()
            self._db_connection = None
        
        if self._connection:
            self._connection.status = "disconnected"
        
        logger.info("Disconnected from Atuin")
    
    def _register_builtin_tools(self) -> None:
        """Register built-in Atuin tools."""
        
        # Search history tool
        self.register_tool(MCPTool(
            name="search_history",
            description="Search command history with filters",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (supports regex)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 1000
                    },
                    "directory": {
                        "type": "string",
                        "description": "Filter by directory"
                    },
                    "hostname": {
                        "type": "string",
                        "description": "Filter by hostname"
                    },
                    "exit_code": {
                        "type": "integer",
                        "description": "Filter by exit code"
                    },
                    "successful_only": {
                        "type": "boolean",
                        "description": "Only return successful commands",
                        "default": False
                    },
                    "before": {
                        "type": "string",
                        "description": "Filter commands before this date (ISO format)"
                    },
                    "after": {
                        "type": "string",
                        "description": "Filter commands after this date (ISO format)"
                    }
                }
            },
            execution_func=self.search_history
        ))
        
        # Get statistics tool
        self.register_tool(MCPTool(
            name="get_statistics",
            description="Get command history statistics",
            parameters_schema={
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Time period for stats (day, week, month, year, all)",
                        "default": "all"
                    },
                    "include_details": {
                        "type": "boolean",
                        "description": "Include detailed breakdown",
                        "default": True
                    }
                }
            },
            execution_func=self.get_statistics
        ))
        
        # Import command tool
        self.register_tool(MCPTool(
            name="import_command",
            description="Import a command into Atuin history",
            parameters_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to import"
                    },
                    "directory": {
                        "type": "string",
                        "description": "Working directory where command was executed"
                    },
                    "exit_code": {
                        "type": "integer",
                        "description": "Command exit code",
                        "default": 0
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Command duration in milliseconds",
                        "default": 0
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "Command timestamp (ISO format, defaults to now)"
                    }
                },
                "required": ["command", "directory"]
            },
            execution_func=self.import_command
        ))
        
        # Get recent commands tool
        self.register_tool(MCPTool(
            name="get_recent_commands",
            description="Get most recent commands",
            parameters_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent commands to return",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "unique_only": {
                        "type": "boolean",
                        "description": "Only return unique commands",
                        "default": False
                    }
                }
            },
            execution_func=self.get_recent_commands
        ))
        
        # Get top commands tool
        self.register_tool(MCPTool(
            name="get_top_commands",
            description="Get most frequently used commands",
            parameters_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of top commands to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period (day, week, month, year, all)",
                        "default": "all"
                    }
                }
            },
            execution_func=self.get_top_commands
        ))
        
        # Sync history tool (if sync enabled)
        if self.enable_sync:
            self.register_tool(MCPTool(
                name="sync_history",
                description="Sync command history with Atuin server",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "force": {
                            "type": "boolean",
                            "description": "Force sync even if up to date",
                            "default": False
                        }
                    }
                },
                execution_func=self.sync_history
            ))
    
    async def search_history(
        self,
        query: Optional[str] = None,
        limit: int = 50,
        directory: Optional[str] = None,
        hostname: Optional[str] = None,
        exit_code: Optional[int] = None,
        successful_only: bool = False,
        before: Optional[str] = None,
        after: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search command history with comprehensive filters.
        
        Args:
            query: Search query (supports regex)
            limit: Maximum number of results
            directory: Filter by directory
            hostname: Filter by hostname
            exit_code: Filter by exit code
            successful_only: Only return successful commands
            before: Filter commands before this date
            after: Filter commands after this date
            
        Returns:
            List of matching history entries
        """
        if not self._db_connection:
            raise ToolExecutionError(
                tool_name="search_history",
                server_name=self._connection.server_id if self._connection else "atuin",
                original_error=Exception("No database connection available")
            )
        
        try:
            # Build SQL query
            sql_conditions = []
            params = []
            
            if query:
                sql_conditions.append("command LIKE ?")
                params.append(f"%{query}%")
            
            if directory:
                sql_conditions.append("cwd = ?")
                params.append(directory)
            
            if hostname:
                sql_conditions.append("hostname = ?")
                params.append(hostname)
            
            if exit_code is not None:
                sql_conditions.append("exit = ?")
                params.append(exit_code)
            
            if successful_only:
                sql_conditions.append("exit = 0")
            
            if before:
                before_ts = int(datetime.fromisoformat(before.replace('Z', '+00:00')).timestamp() * 1_000_000_000)
                sql_conditions.append("timestamp < ?")
                params.append(before_ts)
            
            if after:
                after_ts = int(datetime.fromisoformat(after.replace('Z', '+00:00')).timestamp() * 1_000_000_000)
                sql_conditions.append("timestamp > ?")
                params.append(after_ts)
            
            # Exclude deleted entries
            sql_conditions.append("deleted_at IS NULL")
            
            where_clause = "WHERE " + " AND ".join(sql_conditions) if sql_conditions else ""
            
            sql = f"""
                SELECT id, timestamp, duration, exit, command, cwd, session, hostname, deleted_at
                FROM history
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params.append(limit)
            
            cursor = self._db_connection.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                entry = AtuinHistoryEntry(**dict(row))
                results.append(entry.to_dict())
            
            logger.info(f"History search completed: {len(results)} results for query '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"History search failed: {e}")
            raise ToolExecutionError(
                tool_name="search_history",
                server_name=self._connection.server_id if self._connection else "atuin",
                parameters={"query": query},
                original_error=e
            )
    
    async def get_statistics(
        self,
        period: str = "all",
        include_details: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive command history statistics.
        
        Args:
            period: Time period for statistics
            include_details: Include detailed breakdown
            
        Returns:
            Statistics about command history
        """
        if not self._db_connection:
            raise ToolExecutionError(
                tool_name="get_statistics",
                server_name=self._connection.server_id if self._connection else "atuin",
                original_error=Exception("No database connection available")
            )
        
        try:
            cursor = self._db_connection.cursor()
            
            # Calculate time filter
            time_filter = ""
            if period != "all":
                now = datetime.now()
                if period == "day":
                    start_time = now - timedelta(days=1)
                elif period == "week":
                    start_time = now - timedelta(weeks=1)
                elif period == "month":
                    start_time = now - timedelta(days=30)
                elif period == "year":
                    start_time = now - timedelta(days=365)
                else:
                    start_time = None
                
                if start_time:
                    start_ts = int(start_time.timestamp() * 1_000_000_000)
                    time_filter = f"AND timestamp >= {start_ts}"
            
            # Basic statistics
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_commands,
                    COUNT(CASE WHEN exit = 0 THEN 1 END) as successful_commands,
                    COUNT(CASE WHEN exit != 0 THEN 1 END) as failed_commands,
                    AVG(duration) as avg_duration,
                    MIN(timestamp) as first_command,
                    MAX(timestamp) as last_command,
                    COUNT(DISTINCT session) as unique_sessions,
                    COUNT(DISTINCT hostname) as unique_hosts,
                    COUNT(DISTINCT cwd) as unique_directories
                FROM history 
                WHERE deleted_at IS NULL {time_filter}
            """)
            
            basic_stats = dict(cursor.fetchone())
            
            # Convert timestamps and durations
            if basic_stats['first_command']:
                basic_stats['first_command'] = datetime.fromtimestamp(
                    basic_stats['first_command'] / 1_000_000_000
                ).isoformat()
            
            if basic_stats['last_command']:
                basic_stats['last_command'] = datetime.fromtimestamp(
                    basic_stats['last_command'] / 1_000_000_000
                ).isoformat()
            
            if basic_stats['avg_duration']:
                basic_stats['avg_duration_ms'] = basic_stats['avg_duration'] / 1_000_000
            
            # Calculate success rate
            total = basic_stats['total_commands']
            successful = basic_stats['successful_commands']
            basic_stats['success_rate'] = (successful / total * 100) if total > 0 else 0
            
            stats = {
                "period": period,
                "basic_stats": basic_stats
            }
            
            if include_details:
                # Top commands
                cursor.execute(f"""
                    SELECT command, COUNT(*) as count
                    FROM history 
                    WHERE deleted_at IS NULL {time_filter}
                    GROUP BY command 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                stats["top_commands"] = [dict(row) for row in cursor.fetchall()]
                
                # Top directories
                cursor.execute(f"""
                    SELECT cwd, COUNT(*) as count
                    FROM history 
                    WHERE deleted_at IS NULL {time_filter}
                    GROUP BY cwd 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                stats["top_directories"] = [dict(row) for row in cursor.fetchall()]
                
                # Commands by exit code
                cursor.execute(f"""
                    SELECT exit, COUNT(*) as count
                    FROM history 
                    WHERE deleted_at IS NULL {time_filter}
                    GROUP BY exit 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                stats["exit_codes"] = [dict(row) for row in cursor.fetchall()]
                
                # Activity by hour (for recent period)
                if period in ["day", "week"]:
                    cursor.execute(f"""
                        SELECT 
                            strftime('%H', datetime(timestamp/1000000000, 'unixepoch')) as hour,
                            COUNT(*) as count
                        FROM history 
                        WHERE deleted_at IS NULL {time_filter}
                        GROUP BY hour
                        ORDER BY hour
                    """)
                    stats["activity_by_hour"] = [dict(row) for row in cursor.fetchall()]
            
            logger.info(f"Statistics generated for period: {period}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise ToolExecutionError(
                tool_name="get_statistics",
                server_name=self._connection.server_id if self._connection else "atuin",
                parameters={"period": period},
                original_error=e
            )
    
    async def import_command(
        self,
        command: str,
        directory: str,
        exit_code: int = 0,
        duration: int = 0,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Import a command into Atuin history.
        
        Args:
            command: Command to import
            directory: Working directory
            exit_code: Command exit code
            duration: Duration in milliseconds
            timestamp: Command timestamp (ISO format)
            
        Returns:
            Import result information
        """
        try:
            if not self._atuin_available:
                logger.warning("Atuin CLI not available, cannot import command")
                return {
                    "success": False,
                    "error": "Atuin CLI not available"
                }
            
            # Use atuin import command
            import subprocess
            
            # Convert timestamp if provided
            if timestamp:
                ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                ts = datetime.now()
            
            # Format command for atuin import
            import_cmd = [
                'atuin', 'import', 'auto',
                '--cmd', command,
                '--cwd', directory,
                '--exit', str(exit_code),
                '--duration', str(duration * 1_000_000),  # Convert to nanoseconds
                '--time', ts.strftime('%Y-%m-%d %H:%M:%S')
            ]
            
            result = subprocess.run(
                import_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            success = result.returncode == 0
            
            response = {
                "success": success,
                "command": command,
                "directory": directory,
                "exit_code": exit_code,
                "timestamp": ts.isoformat()
            }
            
            if not success:
                response["error"] = result.stderr.strip()
            
            logger.info(f"Command import {'succeeded' if success else 'failed'}: {command}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to import command: {e}")
            raise ToolExecutionError(
                tool_name="import_command",
                server_name=self._connection.server_id if self._connection else "atuin",
                parameters={"command": command},
                original_error=e
            )
    
    async def get_recent_commands(
        self,
        limit: int = 20,
        unique_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get most recent commands from history.
        
        Args:
            limit: Number of commands to return
            unique_only: Only return unique commands
            
        Returns:
            List of recent commands
        """
        if not self._db_connection:
            return []
        
        try:
            cursor = self._db_connection.cursor()
            
            if unique_only:
                sql = """
                    SELECT DISTINCT command, MAX(timestamp) as latest_timestamp, cwd, exit
                    FROM history 
                    WHERE deleted_at IS NULL
                    GROUP BY command
                    ORDER BY latest_timestamp DESC 
                    LIMIT ?
                """
            else:
                sql = """
                    SELECT id, timestamp, duration, exit, command, cwd, session, hostname
                    FROM history 
                    WHERE deleted_at IS NULL
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
            
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                if unique_only:
                    results.append({
                        "command": row["command"],
                        "latest_timestamp": datetime.fromtimestamp(
                            row["latest_timestamp"] / 1_000_000_000
                        ).isoformat(),
                        "cwd": row["cwd"],
                        "exit": row["exit"]
                    })
                else:
                    entry = AtuinHistoryEntry(**dict(row))
                    results.append(entry.to_dict())
            
            logger.info(f"Retrieved {len(results)} recent commands")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get recent commands: {e}")
            raise ToolExecutionError(
                tool_name="get_recent_commands",
                server_name=self._connection.server_id if self._connection else "atuin",
                original_error=e
            )
    
    async def get_top_commands(
        self,
        limit: int = 10,
        period: str = "all"
    ) -> List[Dict[str, Any]]:
        """Get most frequently used commands.
        
        Args:
            limit: Number of top commands to return
            period: Time period for analysis
            
        Returns:
            List of top commands with usage counts
        """
        if not self._db_connection:
            return []
        
        try:
            # Calculate time filter
            time_filter = ""
            if period != "all":
                now = datetime.now()
                if period == "day":
                    start_time = now - timedelta(days=1)
                elif period == "week":
                    start_time = now - timedelta(weeks=1)
                elif period == "month":
                    start_time = now - timedelta(days=30)
                elif period == "year":
                    start_time = now - timedelta(days=365)
                else:
                    start_time = None
                
                if start_time:
                    start_ts = int(start_time.timestamp() * 1_000_000_000)
                    time_filter = f"AND timestamp >= {start_ts}"
            
            cursor = self._db_connection.cursor()
            cursor.execute(f"""
                SELECT 
                    command,
                    COUNT(*) as count,
                    COUNT(CASE WHEN exit = 0 THEN 1 END) as successful_count,
                    AVG(duration) as avg_duration,
                    MAX(timestamp) as last_used
                FROM history 
                WHERE deleted_at IS NULL {time_filter}
                GROUP BY command 
                ORDER BY count DESC 
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                
                # Convert timestamp and duration
                if row_dict['last_used']:
                    row_dict['last_used'] = datetime.fromtimestamp(
                        row_dict['last_used'] / 1_000_000_000
                    ).isoformat()
                
                if row_dict['avg_duration']:
                    row_dict['avg_duration_ms'] = row_dict['avg_duration'] / 1_000_000
                
                # Calculate success rate
                total = row_dict['count']
                successful = row_dict['successful_count']
                row_dict['success_rate'] = (successful / total * 100) if total > 0 else 0
                
                results.append(row_dict)
            
            logger.info(f"Retrieved {len(results)} top commands for period: {period}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get top commands: {e}")
            raise ToolExecutionError(
                tool_name="get_top_commands",
                server_name=self._connection.server_id if self._connection else "atuin",
                original_error=e
            )
    
    async def sync_history(self, force: bool = False) -> Dict[str, Any]:
        """Sync command history with Atuin server.
        
        Args:
            force: Force sync even if up to date
            
        Returns:
            Sync result information
        """
        if not self.enable_sync or not self._atuin_available:
            return {
                "success": False,
                "error": "Sync not enabled or Atuin CLI not available"
            }
        
        try:
            import subprocess
            
            sync_cmd = ['atuin', 'sync']
            if force:
                sync_cmd.append('--force')
            
            result = subprocess.run(
                sync_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for sync
            )
            
            success = result.returncode == 0
            
            response = {
                "success": success,
                "force": force,
                "output": result.stdout.strip() if success else result.stderr.strip()
            }
            
            logger.info(f"History sync {'succeeded' if success else 'failed'}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to sync history: {e}")
            raise ToolExecutionError(
                tool_name="sync_history",
                server_name=self._connection.server_id if self._connection else "atuin",
                original_error=e
            )
    
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute Atuin tool."""
        if tool_name not in self._tools:
            raise ToolExecutionError(
                tool_name=tool_name,
                server_name=self._connection.server_id if self._connection else "atuin",
                parameters=parameters,
                original_error=ValueError(f"Unknown tool: {tool_name}")
            )
        
        tool = self._tools[tool_name]
        return await tool.execute(**parameters)
    
    async def _discover_tools_impl(self) -> Dict[str, MCPTool]:
        """Discover Atuin tools."""
        return self._tools.copy()


# Example usage and testing
if __name__ == "__main__":
    async def test_atuin_server():
        # Create test server
        server = AtuinServer(
            enable_sync=False  # Disable sync for testing
        )
        
        async with server:
            # Test search
            recent_commands = await server.search_history(limit=10)
            print(f"Recent commands: {len(recent_commands)} found")
            
            # Test statistics
            stats = await server.get_statistics(period="week")
            print(f"Statistics: {stats['basic_stats']['total_commands']} total commands")
            
            # Test top commands
            top_commands = await server.get_top_commands(limit=5)
            print(f"Top commands: {len(top_commands)} found")
            if top_commands:
                print(f"  Most used: {top_commands[0]['command']} ({top_commands[0]['count']} times)")
    
    # Run test
    # asyncio.run(test_atuin_server())
