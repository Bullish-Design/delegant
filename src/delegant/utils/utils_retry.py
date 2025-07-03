"""
Delegant Retry Logic Implementation
===================================

Robust retry mechanisms with exponential backoff, jitter, and comprehensive
error handling for MCP server operations and tool executions.
"""

import asyncio
import functools
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from ..exceptions import RetryExhaustedError, DelegantException
from ..config import get_config

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Available retry strategies."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    delay: float
    error: Exception
    timestamp: datetime
    elapsed_time: float


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    result: Any
    attempts: List[RetryAttempt]
    total_time: float
    final_error: Optional[Exception] = None


class RetryConfig:
    """Configuration for retry operations."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        jitter_range: float = 0.1,
        timeout: Optional[float] = None,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        non_retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.jitter_range = jitter_range
        self.timeout = timeout
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError
        ]
        self.non_retryable_exceptions = non_retryable_exceptions or [
            KeyboardInterrupt,
            SystemExit,
            asyncio.CancelledError
        ]
    
    @classmethod
    def from_global_config(cls) -> 'RetryConfig':
        """Create retry config from global Delegant configuration."""
        config = get_config()
        return cls(
            max_attempts=config.max_retries,
            base_delay=1.0,
            backoff_factor=config.retry_backoff,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=True
        )
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception that occurred
            attempt: Current attempt number (1-based)
            
        Returns:
            True if should retry, False otherwise
        """
        # Check attempt limit
        if attempt >= self.max_attempts:
            return False
        
        # Check non-retryable exceptions first
        for exc_type in self.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # Check retryable exceptions
        for exc_type in self.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        # Default: retry for any DelegantException
        return isinstance(exception, DelegantException)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number.
        
        Args:
            attempt: Attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * attempt
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        elif self.strategy == RetryStrategy.FIBONACCI:
            delay = self.base_delay * self._fibonacci(attempt)
        else:
            delay = self.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.max_delay)
        
        # Apply jitter to avoid thundering herd
        if self.jitter:
            jitter_amount = delay * self.jitter_range
            jitter_offset = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay + jitter_offset)
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number for fibonacci backoff strategy."""
        if n <= 1:
            return 1
        elif n == 2:
            return 1
        else:
            a, b = 1, 1
            for _ in range(3, n + 1):
                a, b = b, a + b
            return b


class RetryHandler:
    """Handles retry logic for operations."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig.from_global_config()
        self.retry_history: List[RetryResult] = []
    
    async def execute(
        self,
        operation: Callable,
        *args,
        operation_name: str = "operation",
        **kwargs
    ) -> RetryResult:
        """Execute an operation with retry logic.
        
        Args:
            operation: Async or sync operation to execute
            *args: Arguments for the operation
            operation_name: Name for logging/debugging
            **kwargs: Keyword arguments for the operation
            
        Returns:
            RetryResult with execution details
            
        Raises:
            RetryExhaustedError: If all retry attempts fail
        """
        start_time = time.time()
        attempts: List[RetryAttempt] = []
        
        # Handle timeout
        if self.config.timeout:
            deadline = start_time + self.config.timeout
        else:
            deadline = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            attempt_start = time.time()
            
            try:
                # Check timeout before attempt
                if deadline and time.time() > deadline:
                    raise TimeoutError(f"Operation '{operation_name}' timed out")
                
                # Execute the operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Success! Create result and return
                total_time = time.time() - start_time
                
                retry_result = RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_time=total_time
                )
                
                self.retry_history.append(retry_result)
                
                if attempt > 1:
                    logger.info(f"Operation '{operation_name}' succeeded on attempt {attempt}")
                
                return retry_result
                
            except Exception as e:
                elapsed_time = time.time() - attempt_start
                
                # Record the attempt
                retry_attempt = RetryAttempt(
                    attempt_number=attempt,
                    delay=0.0,  # Will be updated if we retry
                    error=e,
                    timestamp=datetime.now(),
                    elapsed_time=elapsed_time
                )
                attempts.append(retry_attempt)
                
                # Check if we should retry
                if not self.config.should_retry(e, attempt):
                    logger.error(f"Operation '{operation_name}' failed with non-retryable error: {e}")
                    break
                
                if attempt == self.config.max_attempts:
                    logger.error(f"Operation '{operation_name}' failed after {attempt} attempts")
                    break
                
                # Calculate delay for next attempt
                delay = self.config.calculate_delay(attempt)
                retry_attempt.delay = delay
                
                logger.warning(f"Operation '{operation_name}' failed on attempt {attempt}, retrying in {delay:.2f}s: {e}")
                
                # Check timeout before sleeping
                if deadline and time.time() + delay > deadline:
                    logger.warning(f"Operation '{operation_name}' would exceed timeout, not retrying")
                    break
                
                # Sleep before retry
                await asyncio.sleep(delay)
        
        # All attempts failed
        total_time = time.time() - start_time
        final_error = attempts[-1].error if attempts else Exception("Unknown error")
        
        retry_result = RetryResult(
            success=False,
            result=None,
            attempts=attempts,
            total_time=total_time,
            final_error=final_error
        )
        
        self.retry_history.append(retry_result)
        
        # Create comprehensive error with retry history
        raise RetryExhaustedError(
            operation=operation_name,
            max_retries=self.config.max_attempts,
            retry_history=[
                {
                    "attempt": a.attempt_number,
                    "error": str(a.error),
                    "delay": a.delay,
                    "elapsed_time": a.elapsed_time
                }
                for a in attempts
            ],
            final_error=final_error
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retry statistics.
        
        Returns:
            Dictionary with retry statistics
        """
        if not self.retry_history:
            return {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "average_attempts": 0.0,
                "total_retry_time": 0.0
            }
        
        successful = sum(1 for r in self.retry_history if r.success)
        failed = len(self.retry_history) - successful
        total_attempts = sum(len(r.attempts) for r in self.retry_history)
        total_time = sum(r.total_time for r in self.retry_history)
        
        return {
            "total_operations": len(self.retry_history),
            "successful_operations": successful,
            "failed_operations": failed,
            "success_rate": successful / len(self.retry_history) * 100,
            "average_attempts": total_attempts / len(self.retry_history),
            "total_retry_time": total_time,
            "average_operation_time": total_time / len(self.retry_history)
        }


# Decorator for automatic retry functionality

def retry_with_backoff(
    max_attempts: int = None,
    base_delay: float = None,
    strategy: RetryStrategy = None,
    retryable_exceptions: List[Type[Exception]] = None,
    **retry_kwargs
) -> Callable:
    """Decorator that adds retry logic to functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between attempts
        strategy: Retry strategy to use
        retryable_exceptions: List of exceptions that should trigger retries
        **retry_kwargs: Additional retry configuration
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_attempts=5, base_delay=2.0)
        async def unreliable_operation():
            # Operation that might fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create retry config with overrides
            config_kwargs = {}
            if max_attempts is not None:
                config_kwargs['max_attempts'] = max_attempts
            if base_delay is not None:
                config_kwargs['base_delay'] = base_delay
            if strategy is not None:
                config_kwargs['strategy'] = strategy
            if retryable_exceptions is not None:
                config_kwargs['retryable_exceptions'] = retryable_exceptions
            
            config_kwargs.update(retry_kwargs)
            
            # Use global config as base
            base_config = RetryConfig.from_global_config()
            
            # Override with decorator parameters
            for key, value in config_kwargs.items():
                if hasattr(base_config, key):
                    setattr(base_config, key, value)
            
            # Execute with retry
            handler = RetryHandler(base_config)
            result = await handler.execute(
                func, 
                *args, 
                operation_name=func.__name__,
                **kwargs
            )
            
            return result.result
        
        return wrapper
    
    return decorator


# Specific retry decorators for common patterns

def retry_on_connection_error(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator that retries on connection-related errors."""
    return retry_with_backoff(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=RetryStrategy.EXPONENTIAL,
        retryable_exceptions=[
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError
        ]
    )


def retry_on_server_error(max_attempts: int = 5, base_delay: float = 2.0):
    """Decorator that retries on server-related errors."""
    from ..exceptions import ServerConnectionError, ToolExecutionError
    
    return retry_with_backoff(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=RetryStrategy.EXPONENTIAL,
        retryable_exceptions=[
            ServerConnectionError,
            ToolExecutionError,
            ConnectionError,
            TimeoutError
        ]
    )


def retry_with_fibonacci_backoff(max_attempts: int = 8, base_delay: float = 0.5):
    """Decorator that uses fibonacci backoff strategy."""
    return retry_with_backoff(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=RetryStrategy.FIBONACCI
    )


# Context manager for retry operations

class RetryContext:
    """Context manager for retry operations.
    
    Example:
        async with RetryContext(max_attempts=5) as retry:
            result = await retry.execute(unreliable_operation, arg1, arg2)
    """
    
    def __init__(self, **config_kwargs):
        config = RetryConfig.from_global_config()
        
        # Override with context parameters
        for key, value in config_kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.handler = RetryHandler(config)
    
    async def __aenter__(self):
        return self.handler
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Log statistics if any retries were performed
        stats = self.handler.get_statistics()
        if stats["total_operations"] > 0:
            logger.debug(f"Retry context stats: {stats}")


# Utility functions for common retry patterns

async def retry_operation(
    operation: Callable,
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    **kwargs
) -> Any:
    """Simple function to retry an operation.
    
    Args:
        operation: Operation to retry
        *args: Arguments for the operation
        max_attempts: Maximum retry attempts
        base_delay: Base delay between attempts
        **kwargs: Keyword arguments for the operation
        
    Returns:
        Result of the operation
        
    Raises:
        RetryExhaustedError: If all attempts fail
    """
    config = RetryConfig(max_attempts=max_attempts, base_delay=base_delay)
    handler = RetryHandler(config)
    
    result = await handler.execute(operation, *args, operation_name=operation.__name__, **kwargs)
    return result.result


async def retry_until_success(
    operation: Callable,
    *args,
    timeout: float = 300.0,
    base_delay: float = 1.0,
    **kwargs
) -> Any:
    """Retry operation until success or timeout.
    
    Args:
        operation: Operation to retry
        *args: Arguments for the operation
        timeout: Maximum time to keep trying
        base_delay: Base delay between attempts
        **kwargs: Keyword arguments for the operation
        
    Returns:
        Result of the operation
        
    Raises:
        TimeoutError: If timeout is reached
    """
    config = RetryConfig(
        max_attempts=999,  # Effectively unlimited
        base_delay=base_delay,
        timeout=timeout,
        strategy=RetryStrategy.EXPONENTIAL,
        max_delay=30.0  # Cap at 30 seconds
    )
    handler = RetryHandler(config)
    
    result = await handler.execute(operation, *args, operation_name=operation.__name__, **kwargs)
    return result.result


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import random
    
    # Example operations for testing
    
    async def unreliable_operation(fail_probability: float = 0.7) -> str:
        """Operation that fails randomly."""
        await asyncio.sleep(0.1)  # Simulate work
        
        if random.random() < fail_probability:
            raise ConnectionError("Simulated connection failure")
        
        return "Success!"
    
    @retry_with_backoff(max_attempts=5, base_delay=0.5)
    async def decorated_operation(fail_probability: float = 0.8) -> str:
        """Operation with retry decorator."""
        return await unreliable_operation(fail_probability)
    
    async def test_retry_functionality():
        """Test retry functionality."""
        
        print("Testing retry functionality...")
        
        # Test 1: Manual retry with handler
        print("\n1. Testing manual retry with handler:")
        config = RetryConfig(max_attempts=3, base_delay=0.5, strategy=RetryStrategy.EXPONENTIAL)
        handler = RetryHandler(config)
        
        try:
            result = await handler.execute(
                unreliable_operation,
                fail_probability=0.6,
                operation_name="test_operation"
            )
            print(f"Success: {result.result} (took {result.total_time:.2f}s, {len(result.attempts)} attempts)")
        except RetryExhaustedError as e:
            print(f"Failed after all retries: {e}")
        
        # Test 2: Decorator usage
        print("\n2. Testing retry decorator:")
        try:
            result = await decorated_operation(fail_probability=0.5)
            print(f"Decorated operation succeeded: {result}")
        except RetryExhaustedError as e:
            print(f"Decorated operation failed: {e}")
        
        # Test 3: Context manager
        print("\n3. Testing retry context manager:")
        async with RetryContext(max_attempts=4, strategy=RetryStrategy.FIBONACCI) as retry:
            try:
                result = await retry.execute(
                    unreliable_operation,
                    fail_probability=0.4,
                    operation_name="context_operation"
                )
                print(f"Context operation succeeded: {result.result}")
            except RetryExhaustedError as e:
                print(f"Context operation failed: {e}")
        
        # Test 4: Utility function
        print("\n4. Testing utility function:")
        try:
            result = await retry_operation(
                unreliable_operation,
                fail_probability=0.3,
                max_attempts=3
            )
            print(f"Utility function succeeded: {result}")
        except RetryExhaustedError as e:
            print(f"Utility function failed: {e}")
        
        # Test 5: Statistics
        print("\n5. Retry statistics:")
        stats = handler.get_statistics()
        print(f"Handler stats: {stats}")
    
    # Run test
    # asyncio.run(test_retry_functionality())
