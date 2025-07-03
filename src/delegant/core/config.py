"""
Delegant Configuration Management
=================================

Global configuration system with Pydantic validation and environment variable support.
Manages default settings for connections, retries, context extraction, and monitoring.
"""

import os
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path
import json
import logging

from .exceptions import ConfigurationError


class DelegantConfig(BaseModel):
    """Global configuration schema for Delegant library.
    
    All settings can be overridden via environment variables with DELEGANT_ prefix.
    """
    
    # Connection settings
    auto_retry: bool = Field(
        default=True, 
        description="Enable automatic retries for failed operations"
    )
    max_retries: int = Field(
        default=3, 
        description="Maximum retry attempts",
        ge=0,
        le=10
    )
    retry_backoff: float = Field(
        default=2.0, 
        description="Backoff multiplier for retries",
        ge=1.0,
        le=10.0
    )
    connection_timeout: int = Field(
        default=30, 
        description="Default connection timeout in seconds",
        ge=1,
        le=300
    )
    
    # Server management
    lazy_connect: bool = Field(
        default=True, 
        description="Enable lazy connection (connect on first use)"
    )
    connection_pool_size: int = Field(
        default=100, 
        description="Maximum concurrent connections",
        ge=1,
        le=1000
    )
    
    # Context extraction
    context_extraction: bool = Field(
        default=True, 
        description="Enable context extraction from docstrings/annotations"
    )
    max_context_size: int = Field(
        default=65536, 
        description="Maximum context size in bytes",
        ge=1024,
        le=1048576  # 1MB
    )
    
    # Logging and monitoring
    log_level: str = Field(
        default="INFO", 
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    enable_metrics: bool = Field(
        default=False, 
        description="Enable performance metrics collection"
    )
    metrics_port: Optional[int] = Field(
        default=None, 
        description="Port for metrics endpoint",
        ge=1024,
        le=65535
    )
    
    # Security settings
    api_key_env_prefix: str = Field(
        default="DELEGANT_API_KEY_", 
        description="Environment variable prefix for API keys"
    )
    
    # Development settings
    debug_mode: bool = Field(
        default=False, 
        description="Enable debug mode with additional logging"
    )
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is supported."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @model_validator(mode='after')
    def validate_metrics_config(self) -> 'DelegantConfig':
        """Validate metrics configuration consistency."""
        if self.enable_metrics and self.metrics_port is None:
            raise ValueError("metrics_port must be set when enable_metrics=True")
        return self
    
    @classmethod
    def from_env(cls) -> 'DelegantConfig':
        """Create configuration from environment variables.
        
        Environment variables should use DELEGANT_ prefix:
        - DELEGANT_AUTO_RETRY=false
        - DELEGANT_MAX_RETRIES=5
        - DELEGANT_CONNECTION_TIMEOUT=60
        """
        env_values = {}
        
        for field_name in cls.model_fields:
            env_key = f"DELEGANT_{field_name.upper()}"
            env_value = os.getenv(env_key)
            
            if env_value is not None:
                # Convert string values to appropriate types
                field_info = cls.model_fields[field_name]
                field_type = field_info.annotation
                
                try:
                    if field_type == bool:
                        env_values[field_name] = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif field_type == int or field_type == Optional[int]:
                        env_values[field_name] = int(env_value)
                    elif field_type == float:
                        env_values[field_name] = float(env_value)
                    else:
                        env_values[field_name] = env_value
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(
                        config_source="environment",
                        config_key=env_key,
                        invalid_value=env_value,
                        suggested_fix=f"Ensure {env_key} is a valid {field_type}",
                        original_error=e
                    )
        
        return cls(**env_values)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'DelegantConfig':
        """Load configuration from JSON file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(
                config_source=str(file_path),
                suggested_fix="Create the configuration file or use from_env()"
            )
        
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ConfigurationError(
                config_source=str(file_path),
                suggested_fix="Ensure file exists and contains valid JSON",
                original_error=e
            )
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    def get_api_key(self, service_name: str) -> Optional[str]:
        """Get API key for a service from environment variables.
        
        Looks for environment variable: {api_key_env_prefix}{SERVICE_NAME}
        Example: DELEGANT_API_KEY_GITHUB for GitHub service
        """
        env_key = f"{self.api_key_env_prefix}{service_name.upper()}"
        return os.getenv(env_key)
    
    def configure_logging(self) -> None:
        """Configure Python logging based on current settings."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if self.debug_mode:
            # Enable debug logging for delegant modules
            delegant_logger = logging.getLogger('delegant')
            delegant_logger.setLevel(logging.DEBUG)


class ServerConfig(BaseModel):
    """Configuration schema for individual MCP server instances."""
    
    server_type: str = Field(..., description="Type of MCP server")
    connection_params: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Server-specific connection parameters"
    )
    auto_discover_tools: bool = Field(
        default=False, 
        description="Enable automatic tool discovery"
    )
    timeout: Optional[int] = Field(
        default=None, 
        description="Override default connection timeout"
    )
    context_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional context for the server"
    )
    retry_config: Optional[Dict[str, Union[int, float]]] = Field(
        default=None, 
        description="Override global retry configuration"
    )


# Global configuration instance
_global_config: Optional[DelegantConfig] = None


def get_config() -> DelegantConfig:
    """Get the global Delegant configuration instance."""
    global _global_config
    
    if _global_config is None:
        # Try to load from environment first, then defaults
        _global_config = DelegantConfig.from_env()
        _global_config.configure_logging()
    
    return _global_config


def set_config(config: DelegantConfig) -> None:
    """Set the global Delegant configuration."""
    global _global_config
    _global_config = config
    config.configure_logging()


def configure(
    auto_retry: Optional[bool] = None,
    max_retries: Optional[int] = None,
    retry_backoff: Optional[float] = None,
    connection_timeout: Optional[int] = None,
    lazy_connect: Optional[bool] = None,
    context_extraction: Optional[bool] = None,
    debug_mode: Optional[bool] = None,
    **kwargs: Any
) -> None:
    """Configure global Delegant settings.
    
    This is the main public API for configuration, as mentioned in the spec.
    """
    current_config = get_config()
    
    # Update only provided values
    config_updates = {}
    for key, value in locals().items():
        if key != 'kwargs' and key != 'current_config' and value is not None:
            config_updates[key] = value
    
    # Add any additional kwargs
    config_updates.update(kwargs)
    
    # Create new config with updates
    updated_data = current_config.model_dump()
    updated_data.update(config_updates)
    
    try:
        new_config = DelegantConfig(**updated_data)
        set_config(new_config)
    except Exception as e:
        raise ConfigurationError(
            config_source="runtime_configuration",
            suggested_fix="Check parameter values match expected types",
            original_error=e
        )


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _global_config
    _global_config = None


# Configuration context manager for temporary settings
class ConfigContext:
    """Context manager for temporary configuration changes."""
    
    def __init__(self, **config_overrides: Any) -> None:
        self.overrides = config_overrides
        self.original_config: Optional[DelegantConfig] = None
    
    def __enter__(self) -> DelegantConfig:
        self.original_config = get_config()
        
        # Create temporary config with overrides
        temp_data = self.original_config.model_dump()
        temp_data.update(self.overrides)
        temp_config = DelegantConfig(**temp_data)
        
        set_config(temp_config)
        return temp_config
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.original_config is not None:
            set_config(self.original_config)


# Example usage:
if __name__ == "__main__":
    # Basic configuration
    configure(
        max_retries=5,
        connection_timeout=60,
        debug_mode=True
    )
    
    # Temporary configuration changes
    with ConfigContext(debug_mode=False, max_retries=1):
        config = get_config()
        print(f"Temporary config: retries={config.max_retries}, debug={config.debug_mode}")
    
    # Config is restored after context
    final_config = get_config()
    print(f"Final config: retries={final_config.max_retries}, debug={final_config.debug_mode}")
