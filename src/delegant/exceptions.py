"""
Delegant Exception Hierarchy
============================

Custom exception classes for comprehensive error handling throughout the library.
All exceptions provide clear, actionable error messages with debugging context.
"""

from typing import Any, Dict, Optional


class DelegantException(Exception):
    """Base exception for all Delegant errors.
    
    Provides common functionality for error context and debugging information.
    """
    
    def __init__(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.original_error = original_error
    
    def __str__(self) -> str:
        """Return detailed error message with context."""
        error_parts = [self.message]
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            error_parts.append(f"Context: {context_str}")
        
        if self.original_error:
            error_parts.append(f"Original error: {self.original_error}")
        
        return " | ".join(error_parts)


class ServerConnectionError(DelegantException):
    """Raised when MCP server connection fails.
    
    Includes server details and connection parameters for debugging.
    """
    
    def __init__(
        self,
        server_name: str,
        server_type: str,
        connection_url: Optional[str] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        message = f"Failed to connect to MCP server '{server_name}' of type '{server_type}'"
        context = {
            "server_name": server_name,
            "server_type": server_type,
        }
        if connection_url:
            context["connection_url"] = connection_url
            
        super().__init__(message, context, original_error)


class ToolDiscoveryError(DelegantException):
    """Raised when tool discovery fails on an MCP server.
    
    Provides details about which server and what discovery method failed.
    """
    
    def __init__(
        self,
        server_name: str,
        discovery_method: str,
        available_tools: Optional[list] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        message = f"Tool discovery failed for server '{server_name}' using method '{discovery_method}'"
        context = {
            "server_name": server_name,
            "discovery_method": discovery_method,
        }
        if available_tools is not None:
            context["available_tools"] = available_tools
            
        super().__init__(message, context, original_error)


class ContextExtractionError(DelegantException):
    """Raised when context extraction from docstrings/annotations fails.
    
    Non-blocking error that allows degraded functionality.
    """
    
    def __init__(
        self,
        target_name: str,
        extraction_type: str,
        partial_context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        message = f"Context extraction failed for '{target_name}' using '{extraction_type}'"
        context = {
            "target_name": target_name,
            "extraction_type": extraction_type,
        }
        if partial_context:
            context["partial_context"] = partial_context
            
        super().__init__(message, context, original_error)


class WorkflowExecutionError(DelegantException):
    """Raised during workflow execution failures.
    
    Includes workflow state and partial results for recovery.
    """
    
    def __init__(
        self,
        workflow_type: str,
        failed_step: str,
        partial_results: Optional[Dict[str, Any]] = None,
        agent_states: Optional[Dict[str, str]] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        message = f"Workflow '{workflow_type}' failed at step '{failed_step}'"
        context = {
            "workflow_type": workflow_type,
            "failed_step": failed_step,
        }
        if partial_results:
            context["partial_results"] = partial_results
        if agent_states:
            context["agent_states"] = agent_states
            
        super().__init__(message, context, original_error)


class ValidationError(DelegantException):
    """Raised when input validation fails.
    
    Provides detailed field-level validation errors.
    """
    
    def __init__(
        self,
        validation_target: str,
        field_errors: Optional[Dict[str, str]] = None,
        invalid_value: Optional[Any] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        message = f"Validation failed for '{validation_target}'"
        context = {
            "validation_target": validation_target,
        }
        if field_errors:
            context["field_errors"] = field_errors
        if invalid_value is not None:
            context["invalid_value"] = str(invalid_value)
            
        super().__init__(message, context, original_error)


class ConfigurationError(DelegantException):
    """Raised for configuration-related errors.
    
    Includes configuration source and suggested fixes.
    """
    
    def __init__(
        self,
        config_source: str,
        config_key: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        suggested_fix: Optional[str] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        message = f"Configuration error in '{config_source}'"
        if config_key:
            message += f" for key '{config_key}'"
            
        context = {
            "config_source": config_source,
        }
        if config_key:
            context["config_key"] = config_key
        if invalid_value is not None:
            context["invalid_value"] = str(invalid_value)
        if suggested_fix:
            context["suggested_fix"] = suggested_fix
            
        super().__init__(message, context, original_error)


class ToolExecutionError(DelegantException):
    """Raised when MCP tool execution fails.
    
    Includes tool details and execution context.
    """
    
    def __init__(
        self,
        tool_name: str,
        server_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        message = f"Tool '{tool_name}' execution failed on server '{server_name}'"
        context = {
            "tool_name": tool_name,
            "server_name": server_name,
        }
        if parameters:
            context["parameters"] = parameters
        if execution_time is not None:
            context["execution_time"] = execution_time
            
        super().__init__(message, context, original_error)


class RetryExhaustedError(DelegantException):
    """Raised when all retry attempts are exhausted.
    
    Includes retry history and final failure reason.
    """
    
    def __init__(
        self,
        operation: str,
        max_retries: int,
        retry_history: Optional[list] = None,
        final_error: Optional[Exception] = None
    ) -> None:
        message = f"All {max_retries} retry attempts exhausted for operation '{operation}'"
        context = {
            "operation": operation,
            "max_retries": max_retries,
        }
        if retry_history:
            context["retry_history"] = retry_history
            
        super().__init__(message, context, final_error)


# Convenience function for error context
def create_error_context(**kwargs: Any) -> Dict[str, Any]:
    """Create error context dictionary with non-None values."""
    return {k: v for k, v in kwargs.items() if v is not None}


# Error severity levels for logging
class ErrorSeverity:
    """Error severity constants for logging and monitoring."""
    LOW = "low"           # Warnings, non-blocking issues
    MEDIUM = "medium"     # Recoverable errors with fallbacks  
    HIGH = "high"         # Blocking errors requiring user intervention
    CRITICAL = "critical" # System-level failures
