"""
Delegant Connection Pool Utilities
==================================

Connection pooling and lifecycle management utilities for efficient
MCP server connection handling and resource optimization.
"""

import asyncio
import logging
import weakref
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Type
from collections import defaultdict
from dataclasses import dataclass
import threading

from ..server import MCPServer
from ..exceptions import ServerConnectionError, ConfigurationError
from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a pooled connection."""
    server: MCPServer
    created_at: datetime
    last_used: datetime
    usage_count: int
    is_active: bool
    pool_id: str


class ConnectionPool:
    """Thread-safe connection pool for MCP servers.
    
    Manages server connections with automatic cleanup, health checking,
    and efficient resource utilization.
    
    Example:
        pool = ConnectionPool(max_size=50, cleanup_interval=300)
        
        # Get or create connection
        server = await pool.get_connection(FileSystemServer, root_dir="/data")
        
        # Use server...
        result = await server.read_file("document.txt")
        
        # Return to pool (automatic via context manager)
        await pool.return_connection(server)
    """
    
    def __init__(
        self,
        max_size: Optional[int] = None,
        cleanup_interval: int = 300,
        max_idle_time: int = 3600,
        health_check_interval: int = 600
    ):
        self.max_size = max_size or get_config().connection_pool_size
        self.cleanup_interval = cleanup_interval
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        
        # Connection storage
        self._connections: Dict[str, ConnectionInfo] = {}
        self._connections_by_type: Dict[Type[MCPServer], List[str]] = defaultdict(list)
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self._stats = {
            "connections_created": 0,
            "connections_reused": 0,
            "connections_cleaned": 0,
            "health_checks_performed": 0,
            "health_check_failures": 0
        }
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self) -> None:
        """Start background cleanup and health check tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up idle connections."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.cleanup_interval
                )
            except asyncio.TimeoutError:
                await self._cleanup_idle_connections()
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _health_check_loop(self) -> None:
        """Background task for health checking connections."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.health_check_interval
                )
            except asyncio.TimeoutError:
                await self._perform_health_checks()
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    def _generate_connection_key(
        self, 
        server_type: Type[MCPServer], 
        **config
    ) -> str:
        """Generate unique key for connection based on type and config."""
        # Create a stable hash of the configuration
        import hashlib
        import json
        
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{server_type.__name__}_{config_hash}"
    
    async def get_connection(
        self, 
        server_type: Type[MCPServer],
        **config
    ) -> MCPServer:
        """Get connection from pool or create new one.
        
        Args:
            server_type: Type of MCP server to get
            **config: Configuration parameters for the server
            
        Returns:
            Connected MCP server instance
            
        Raises:
            ServerConnectionError: If connection fails
            ConfigurationError: If pool is full
        """
        connection_key = self._generate_connection_key(server_type, **config)
        
        async with self._global_lock:
            # Check if connection exists and is healthy
            if connection_key in self._connections:
                conn_info = self._connections[connection_key]
                
                if conn_info.is_active:
                    # Update usage statistics
                    conn_info.last_used = datetime.now()
                    conn_info.usage_count += 1
                    self._stats["connections_reused"] += 1
                    
                    logger.debug(f"Reusing connection: {connection_key}")
                    return conn_info.server
                else:
                    # Remove inactive connection
                    await self._remove_connection(connection_key)
            
            # Check pool size limit
            if len(self._connections) >= self.max_size:
                # Try to clean up idle connections first
                await self._cleanup_idle_connections()
                
                if len(self._connections) >= self.max_size:
                    raise ConfigurationError(
                        config_source="connection_pool",
                        suggested_fix=f"Connection pool full ({self.max_size}). Increase pool size or clean up connections."
                    )
            
            # Create new connection
            try:
                server = server_type(**config)
                await server.connect()
                
                # Add to pool
                conn_info = ConnectionInfo(
                    server=server,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    usage_count=1,
                    is_active=True,
                    pool_id=connection_key
                )
                
                self._connections[connection_key] = conn_info
                self._connections_by_type[server_type].append(connection_key)
                self._locks[connection_key] = asyncio.Lock()
                
                self._stats["connections_created"] += 1
                
                logger.info(f"Created new connection: {connection_key}")
                return server
                
            except Exception as e:
                logger.error(f"Failed to create connection {connection_key}: {e}")
                raise ServerConnectionError(
                    server_name=connection_key,
                    server_type=server_type.__name__,
                    original_error=e
                )
    
    async def return_connection(self, server: MCPServer) -> None:
        """Return connection to pool.
        
        Args:
            server: Server instance to return
        """
        # Find connection in pool
        connection_key = None
        for key, conn_info in self._connections.items():
            if conn_info.server is server:
                connection_key = key
                break
        
        if connection_key:
            # Update last used time
            self._connections[connection_key].last_used = datetime.now()
            logger.debug(f"Returned connection to pool: {connection_key}")
        else:
            # Connection not in pool, disconnect it
            try:
                await server.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting untracked server: {e}")
    
    async def remove_connection(self, server: MCPServer) -> None:
        """Remove connection from pool and disconnect.
        
        Args:
            server: Server instance to remove
        """
        connection_key = None
        for key, conn_info in self._connections.items():
            if conn_info.server is server:
                connection_key = key
                break
        
        if connection_key:
            await self._remove_connection(connection_key)
    
    async def _remove_connection(self, connection_key: str) -> None:
        """Remove connection from pool by key."""
        if connection_key not in self._connections:
            return
        
        conn_info = self._connections[connection_key]
        
        try:
            await conn_info.server.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting server {connection_key}: {e}")
        
        # Remove from all tracking structures
        del self._connections[connection_key]
        
        # Remove from type index
        server_type = type(conn_info.server)
        if connection_key in self._connections_by_type[server_type]:
            self._connections_by_type[server_type].remove(connection_key)
        
        # Remove lock
        if connection_key in self._locks:
            del self._locks[connection_key]
        
        logger.info(f"Removed connection from pool: {connection_key}")
    
    async def _cleanup_idle_connections(self) -> None:
        """Clean up idle connections that haven't been used recently."""
        now = datetime.now()
        idle_threshold = now - timedelta(seconds=self.max_idle_time)
        
        keys_to_remove = []
        
        for key, conn_info in self._connections.items():
            if conn_info.last_used < idle_threshold:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            await self._remove_connection(key)
            self._stats["connections_cleaned"] += 1
        
        if keys_to_remove:
            logger.info(f"Cleaned up {len(keys_to_remove)} idle connections")
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all active connections."""
        failed_connections = []
        
        for key, conn_info in self._connections.items():
            if not conn_info.is_active:
                continue
            
            try:
                # Use the server's health check if available
                is_healthy = await conn_info.server.health_check()
                
                if not is_healthy:
                    conn_info.is_active = False
                    failed_connections.append(key)
                else:
                    self._stats["health_checks_performed"] += 1
                
            except Exception as e:
                logger.warning(f"Health check failed for {key}: {e}")
                conn_info.is_active = False
                failed_connections.append(key)
                self._stats["health_check_failures"] += 1
        
        # Remove failed connections
        for key in failed_connections:
            await self._remove_connection(key)
        
        if failed_connections:
            logger.info(f"Removed {len(failed_connections)} unhealthy connections")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connection pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        active_connections = sum(1 for c in self._connections.values() if c.is_active)
        
        # Calculate connection age statistics
        now = datetime.now()
        ages = [(now - c.created_at).total_seconds() for c in self._connections.values()]
        
        return {
            "total_connections": len(self._connections),
            "active_connections": active_connections,
            "inactive_connections": len(self._connections) - active_connections,
            "max_size": self.max_size,
            "usage_statistics": self._stats.copy(),
            "connection_ages": {
                "min": min(ages) if ages else 0,
                "max": max(ages) if ages else 0,
                "avg": sum(ages) / len(ages) if ages else 0
            },
            "connections_by_type": {
                server_type.__name__: len(keys) 
                for server_type, keys in self._connections_by_type.items()
            }
        }
    
    def list_connections(self) -> List[Dict[str, Any]]:
        """List all connections in the pool.
        
        Returns:
            List of connection information
        """
        connections = []
        
        for key, conn_info in self._connections.items():
            connections.append({
                "key": key,
                "server_type": type(conn_info.server).__name__,
                "created_at": conn_info.created_at.isoformat(),
                "last_used": conn_info.last_used.isoformat(),
                "usage_count": conn_info.usage_count,
                "is_active": conn_info.is_active,
                "age_seconds": (datetime.now() - conn_info.created_at).total_seconds()
            })
        
        return connections
    
    async def clear_all(self) -> None:
        """Clear all connections from the pool."""
        async with self._global_lock:
            keys = list(self._connections.keys())
            
            for key in keys:
                await self._remove_connection(key)
            
            logger.info(f"Cleared all {len(keys)} connections from pool")
    
    async def shutdown(self) -> None:
        """Shutdown the connection pool and all background tasks."""
        logger.info("Shutting down connection pool...")
        
        # Signal background tasks to stop
        self._shutdown_event.set()
        
        # Wait for background tasks to complete
        if self._cleanup_task and not self._cleanup_task.done():
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._cleanup_task.cancel()
        
        if self._health_check_task and not self._health_check_task.done():
            try:
                await asyncio.wait_for(self._health_check_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._health_check_task.cancel()
        
        # Clear all connections
        await self.clear_all()
        
        logger.info("Connection pool shutdown complete")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()


class PooledConnection:
    """Context manager for pooled connections.
    
    Automatically returns connections to the pool when done.
    
    Example:
        async with PooledConnection(pool, FileSystemServer, root_dir="/data") as server:
            content = await server.read_file("document.txt")
        # Connection automatically returned to pool
    """
    
    def __init__(
        self, 
        pool: ConnectionPool, 
        server_type: Type[MCPServer],
        **config
    ):
        self.pool = pool
        self.server_type = server_type
        self.config = config
        self.server: Optional[MCPServer] = None
    
    async def __aenter__(self) -> MCPServer:
        """Get connection from pool."""
        self.server = await self.pool.get_connection(self.server_type, **self.config)
        return self.server
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Return connection to pool."""
        if self.server:
            await self.pool.return_connection(self.server)


# Global connection pool instance
_global_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_global_pool() -> ConnectionPool:
    """Get the global connection pool instance."""
    global _global_pool
    
    if _global_pool is None:
        with _pool_lock:
            if _global_pool is None:
                config = get_config()
                _global_pool = ConnectionPool(
                    max_size=config.connection_pool_size,
                    cleanup_interval=300,
                    max_idle_time=3600,
                    health_check_interval=600
                )
    
    return _global_pool


async def set_global_pool(pool: ConnectionPool) -> None:
    """Set the global connection pool instance."""
    global _global_pool
    
    # Shutdown existing pool if it exists
    if _global_pool:
        await _global_pool.shutdown()
    
    _global_pool = pool


async def shutdown_global_pool() -> None:
    """Shutdown the global connection pool."""
    global _global_pool
    
    if _global_pool:
        await _global_pool.shutdown()
        _global_pool = None


# Convenience functions for common pooling operations

async def get_pooled_connection(
    server_type: Type[MCPServer],
    **config
) -> MCPServer:
    """Get a connection from the global pool.
    
    Args:
        server_type: Type of server to get
        **config: Server configuration
        
    Returns:
        Connected server instance
    """
    pool = get_global_pool()
    return await pool.get_connection(server_type, **config)


async def return_pooled_connection(server: MCPServer) -> None:
    """Return a connection to the global pool.
    
    Args:
        server: Server instance to return
    """
    pool = get_global_pool()
    await pool.return_connection(server)


def pooled_connection(server_type: Type[MCPServer], **config) -> PooledConnection:
    """Create a pooled connection context manager.
    
    Args:
        server_type: Type of server to get
        **config: Server configuration
        
    Returns:
        PooledConnection context manager
        
    Example:
        async with pooled_connection(FileSystemServer, root_dir="/data") as server:
            content = await server.read_file("document.txt")
    """
    pool = get_global_pool()
    return PooledConnection(pool, server_type, **config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from ..servers.filesystem import FileSystemServer
    
    async def test_connection_pool():
        """Test connection pool functionality."""
        
        # Create a connection pool
        async with ConnectionPool(max_size=10, cleanup_interval=60) as pool:
            print(f"Created connection pool with max size: 10")
            
            # Test getting connections
            server1 = await pool.get_connection(FileSystemServer, root_dir="/tmp")
            server2 = await pool.get_connection(FileSystemServer, root_dir="/tmp")
            server3 = await pool.get_connection(FileSystemServer, root_dir="/home")
            
            print(f"Created 3 connections")
            
            # Check statistics
            stats = pool.get_statistics()
            print(f"Pool stats: {stats['total_connections']} total, {stats['active_connections']} active")
            
            # Test connection reuse (should get same instance for same config)
            server4 = await pool.get_connection(FileSystemServer, root_dir="/tmp")
            print(f"Connection reused: {server1 is server4}")
            
            # Test pooled connection context manager
            async with PooledConnection(pool, FileSystemServer, root_dir="/var") as server:
                print(f"Using pooled connection: {type(server).__name__}")
            
            # List all connections
            connections = pool.list_connections()
            print(f"Pool contains {len(connections)} connections:")
            for conn in connections:
                print(f"  - {conn['server_type']} (used {conn['usage_count']} times)")
    
    # Run test
    # asyncio.run(test_connection_pool())
