"""
Delegant Validation Utilities
=============================

Custom validators and validation functions for server configurations,
agent setups, and runtime parameter validation with comprehensive
error reporting and helpful suggestions.
"""

import inspect
import re
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
import logging

from ..exceptions import ValidationError, ConfigurationError
from ..server import MCPServer

logger = logging.getLogger(__name__)


class ValidationRule:
    """Base class for validation rules."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        """Validate a value. Must be implemented by subclasses.
        
        Args:
            value: Value to validate
            context: Additional context for validation
            
        Returns:
            True if valid, False otherwise
        """
        raise NotImplementedError
    
    def get_error_message(self, value: Any, context: Dict[str, Any] = None) -> str:
        """Get error message for invalid value.
        
        Args:
            value: Invalid value
            context: Additional context
            
        Returns:
            Error message
        """
        return f"Validation failed for rule '{self.name}': {self.description}"


class PathExistsRule(ValidationRule):
    """Validate that a path exists."""
    
    def __init__(self, must_be_dir: bool = False, must_be_file: bool = False):
        super().__init__(
            "path_exists",
            f"Path must exist and be a {'directory' if must_be_dir else 'file' if must_be_file else 'valid path'}"
        )
        self.must_be_dir = must_be_dir
        self.must_be_file = must_be_file
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if not isinstance(value, (str, Path)):
            return False
        
        path = Path(value)
        if not path.exists():
            return False
        
        if self.must_be_dir and not path.is_dir():
            return False
        
        if self.must_be_file and not path.is_file():
            return False
        
        return True
    
    def get_error_message(self, value: Any, context: Dict[str, Any] = None) -> str:
        path = Path(value) if isinstance(value, (str, Path)) else "invalid"
        
        if not path.exists():
            return f"Path does not exist: {path}"
        elif self.must_be_dir and not path.is_dir():
            return f"Path exists but is not a directory: {path}"
        elif self.must_be_file and not path.is_file():
            return f"Path exists but is not a file: {path}"
        else:
            return f"Invalid path: {path}"


class URLValidRule(ValidationRule):
    """Validate URL format."""
    
    def __init__(self, schemes: List[str] = None):
        self.schemes = schemes or ["http", "https"]
        super().__init__(
            "url_valid",
            f"Must be a valid URL with scheme: {', '.join(self.schemes)}"
        )
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if not isinstance(value, str):
            return False
        
        try:
            parsed = urllib.parse.urlparse(value)
            return (
                parsed.scheme in self.schemes and
                parsed.netloc and
                len(parsed.netloc) > 0
            )
        except Exception:
            return False
    
    def get_error_message(self, value: Any, context: Dict[str, Any] = None) -> str:
        return f"Invalid URL: {value}. Must be a valid URL with scheme: {', '.join(self.schemes)}"


class RangeRule(ValidationRule):
    """Validate numeric range."""
    
    def __init__(self, min_value: Optional[float] = None, max_value: Optional[float] = None):
        self.min_value = min_value
        self.max_value = max_value
        
        desc_parts = []
        if min_value is not None:
            desc_parts.append(f"≥ {min_value}")
        if max_value is not None:
            desc_parts.append(f"≤ {max_value}")
        
        super().__init__(
            "range",
            f"Must be numeric value {' and '.join(desc_parts)}" if desc_parts else "Must be numeric"
        )
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return False
        
        if self.min_value is not None and num_value < self.min_value:
            return False
        
        if self.max_value is not None and num_value > self.max_value:
            return False
        
        return True
    
    def get_error_message(self, value: Any, context: Dict[str, Any] = None) -> str:
        if self.min_value is not None and self.max_value is not None:
            return f"Value {value} must be between {self.min_value} and {self.max_value}"
        elif self.min_value is not None:
            return f"Value {value} must be at least {self.min_value}"
        elif self.max_value is not None:
            return f"Value {value} must be at most {self.max_value}"
        else:
            return f"Value {value} must be numeric"


class RegexRule(ValidationRule):
    """Validate against regular expression pattern."""
    
    def __init__(self, pattern: str, description: str = None):
        self.pattern = re.compile(pattern)
        super().__init__(
            "regex",
            description or f"Must match pattern: {pattern}"
        )
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if not isinstance(value, str):
            return False
        
        return bool(self.pattern.match(value))
    
    def get_error_message(self, value: Any, context: Dict[str, Any] = None) -> str:
        return f"Value '{value}' does not match required pattern: {self.pattern.pattern}"


class ChoiceRule(ValidationRule):
    """Validate value is in allowed choices."""
    
    def __init__(self, choices: List[Any], case_sensitive: bool = True):
        self.choices = choices
        self.case_sensitive = case_sensitive
        
        if not case_sensitive:
            self.choices_lower = [str(c).lower() for c in choices]
        
        super().__init__(
            "choice",
            f"Must be one of: {', '.join(str(c) for c in choices)}"
        )
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if self.case_sensitive:
            return value in self.choices
        else:
            return str(value).lower() in self.choices_lower
    
    def get_error_message(self, value: Any, context: Dict[str, Any] = None) -> str:
        return f"Value '{value}' is not valid. Must be one of: {', '.join(str(c) for c in self.choices)}"


class DependencyRule(ValidationRule):
    """Validate that dependencies are satisfied."""
    
    def __init__(self, required_fields: List[str], description: str = None):
        self.required_fields = required_fields
        super().__init__(
            "dependency",
            description or f"Requires fields: {', '.join(required_fields)}"
        )
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if not context:
            return False
        
        return all(field in context and context[field] is not None for field in self.required_fields)
    
    def get_error_message(self, value: Any, context: Dict[str, Any] = None) -> str:
        missing = []
        if context:
            missing = [field for field in self.required_fields if field not in context or context[field] is None]
        else:
            missing = self.required_fields
        
        return f"Missing required dependencies: {', '.join(missing)}"


class ServerConfigValidator:
    """Validator for server configurations."""
    
    def __init__(self):
        # Define validation rules for different server types
        self.server_rules = {
            "FileSystemServer": {
                "root_dir": [PathExistsRule(must_be_dir=True)],
                "max_file_size": [RangeRule(min_value=1024, max_value=1073741824)],  # 1KB to 1GB
                "allowed_extensions": [self._validate_extensions]
            },
            "WebSearchServer": {
                "provider": [ChoiceRule(["duckduckgo", "google", "bing"], case_sensitive=False)],
                "base_url": [URLValidRule()],
                "max_results": [RangeRule(min_value=1, max_value=100)],
                "timeout": [RangeRule(min_value=5, max_value=300)],
                "api_key": [DependencyRule(["provider"], "API key required for Google and Bing providers")]
            },
            "TerminalServer": {
                "shell": [PathExistsRule(must_be_file=True)],
                "timeout_seconds": [RangeRule(min_value=1, max_value=3600)],
                "max_output_size": [RangeRule(min_value=1024, max_value=10485760)],  # 1KB to 10MB
                "allowed_commands": [self._validate_command_list]
            },
            "AtuinServer": {
                "atuin_db_path": [PathExistsRule(must_be_file=True)],
                "max_results": [RangeRule(min_value=1, max_value=10000)]
            }
        }
    
    def _validate_extensions(self, value: Any, context: Dict[str, Any] = None) -> bool:
        """Custom validator for file extensions."""
        if value is None:
            return True  # Optional field
        
        if not isinstance(value, list):
            return False
        
        for ext in value:
            if not isinstance(ext, str) or not ext.startswith('.'):
                return False
        
        return True
    
    def _validate_command_list(self, value: Any, context: Dict[str, Any] = None) -> bool:
        """Custom validator for command lists."""
        if value is None:
            return True  # Optional field
        
        if not isinstance(value, list):
            return False
        
        for cmd in value:
            if not isinstance(cmd, str) or len(cmd.strip()) == 0:
                return False
        
        return True
    
    def validate_server_config(
        self, 
        server_type: Type[MCPServer], 
        config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate server configuration.
        
        Args:
            server_type: Type of server to validate
            config: Configuration dictionary
            
        Returns:
            ValidationResult with validation details
        """
        server_name = server_type.__name__
        errors = {}
        warnings = []
        
        # Get rules for this server type
        type_rules = self.server_rules.get(server_name, {})
        
        # Validate each configured field
        for field_name, field_value in config.items():
            field_rules = type_rules.get(field_name, [])
            
            for rule in field_rules:
                if callable(rule):
                    # Custom validation function
                    if not rule(field_value, config):
                        errors[field_name] = f"Custom validation failed for {field_name}"
                elif isinstance(rule, ValidationRule):
                    # Standard validation rule
                    if not rule.validate(field_value, config):
                        errors[field_name] = rule.get_error_message(field_value, config)
        
        # Check for provider-specific requirements
        if server_name == "WebSearchServer":
            provider = config.get("provider", "").lower()
            if provider in ["google", "bing"] and not config.get("api_key"):
                errors["api_key"] = f"API key required for {provider} provider"
            
            if provider == "google" and not config.get("custom_search_engine_id"):
                errors["custom_search_engine_id"] = "Custom Search Engine ID required for Google provider"
        
        # Generate suggestions for common issues
        suggestions = self._generate_suggestions(server_name, config, errors)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            validated_config=config
        )
    
    def _generate_suggestions(
        self, 
        server_name: str, 
        config: Dict[str, Any], 
        errors: Dict[str, str]
    ) -> List[str]:
        """Generate helpful suggestions for configuration issues."""
        suggestions = []
        
        if server_name == "FileSystemServer":
            if "root_dir" in errors:
                suggestions.append("Create the directory or use an existing path")
            
            if config.get("max_file_size", 0) > 100 * 1024 * 1024:  # 100MB
                suggestions.append("Consider using a smaller max_file_size for better performance")
        
        elif server_name == "WebSearchServer":
            if "api_key" in errors:
                if config.get("provider") == "google":
                    suggestions.append("Get a Google Custom Search API key from Google Cloud Console")
                elif config.get("provider") == "bing":
                    suggestions.append("Get a Bing Search API key from Azure Cognitive Services")
            
            if config.get("max_results", 0) > 50:
                suggestions.append("Large result sets may impact performance and API quotas")
        
        elif server_name == "TerminalServer":
            if "shell" in errors:
                suggestions.append("Use a valid shell path like '/bin/bash' or '/bin/sh'")
            
            if config.get("timeout_seconds", 0) > 300:
                suggestions.append("Long timeouts may cause operations to hang")
        
        return suggestions


class AgentConfigValidator:
    """Validator for agent configurations."""
    
    def validate_agent_class(self, agent_class: Type) -> ValidationResult:
        """Validate agent class definition.
        
        Args:
            agent_class: Agent class to validate
            
        Returns:
            ValidationResult with validation details
        """
        errors = {}
        warnings = []
        suggestions = []
        
        # Check if class has instruction field
        if not hasattr(agent_class, 'instruction'):
            errors["instruction"] = "Agent class must have an 'instruction' field"
        
        # Validate server type annotations
        if hasattr(agent_class, '__annotations__'):
            annotations = agent_class.__annotations__
            
            for field_name, field_type in annotations.items():
                if field_name == 'instruction':
                    continue
                
                # Check if it's a valid server type
                if inspect.isclass(field_type) and issubclass(field_type, MCPServer):
                    # Valid server type
                    continue
                else:
                    warnings.append(f"Field '{field_name}' does not appear to be an MCPServer type")
        
        # Check for required methods
        required_methods = []  # Define any required methods for agents
        for method_name in required_methods:
            if not hasattr(agent_class, method_name):
                errors[method_name] = f"Agent class must implement method '{method_name}'"
        
        # Generate suggestions
        if len(annotations) == 1:  # Only instruction field
            suggestions.append("Consider adding MCP server fields to enable functionality")
        
        if not agent_class.__doc__:
            suggestions.append("Add a docstring to improve context extraction")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            validated_config={"class_name": agent_class.__name__}
        )


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(
        self,
        valid: bool,
        errors: Dict[str, str] = None,
        warnings: List[str] = None,
        suggestions: List[str] = None,
        validated_config: Any = None
    ):
        self.valid = valid
        self.errors = errors or {}
        self.warnings = warnings or []
        self.suggestions = suggestions or []
        self.validated_config = validated_config
    
    def __bool__(self) -> bool:
        """Allow boolean evaluation."""
        return self.valid
    
    def raise_if_invalid(self, validation_target: str = "configuration") -> None:
        """Raise ValidationError if validation failed.
        
        Args:
            validation_target: Name of what was being validated
            
        Raises:
            ValidationError: If validation failed
        """
        if not self.valid:
            raise ValidationError(
                validation_target=validation_target,
                field_errors=self.errors,
                invalid_value=self.validated_config
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "validated_config": self.validated_config
        }


# Convenience functions

def validate_server_config(
    server_type: Type[MCPServer], 
    config: Dict[str, Any],
    raise_on_error: bool = False
) -> ValidationResult:
    """Validate server configuration.
    
    Args:
        server_type: Type of server to validate
        config: Configuration dictionary
        raise_on_error: Whether to raise exception on validation failure
        
    Returns:
        ValidationResult
        
    Raises:
        ValidationError: If validation fails and raise_on_error is True
    """
    validator = ServerConfigValidator()
    result = validator.validate_server_config(server_type, config)
    
    if raise_on_error:
        result.raise_if_invalid(f"{server_type.__name__} configuration")
    
    return result


def validate_agent_class(
    agent_class: Type,
    raise_on_error: bool = False
) -> ValidationResult:
    """Validate agent class definition.
    
    Args:
        agent_class: Agent class to validate
        raise_on_error: Whether to raise exception on validation failure
        
    Returns:
        ValidationResult
        
    Raises:
        ValidationError: If validation fails and raise_on_error is True
    """
    validator = AgentConfigValidator()
    result = validator.validate_agent_class(agent_class)
    
    if raise_on_error:
        result.raise_if_invalid(f"Agent class {agent_class.__name__}")
    
    return result


def validate_file_path(
    path: Union[str, Path],
    must_exist: bool = True,
    must_be_dir: bool = False,
    must_be_file: bool = False
) -> ValidationResult:
    """Validate file path.
    
    Args:
        path: Path to validate
        must_exist: Whether path must exist
        must_be_dir: Whether path must be a directory
        must_be_file: Whether path must be a file
        
    Returns:
        ValidationResult
    """
    errors = {}
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        errors["path"] = f"Path does not exist: {path}"
    elif path_obj.exists():
        if must_be_dir and not path_obj.is_dir():
            errors["path"] = f"Path is not a directory: {path}"
        elif must_be_file and not path_obj.is_file():
            errors["path"] = f"Path is not a file: {path}"
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        validated_config={"path": str(path)}
    )


def validate_url(url: str, schemes: List[str] = None) -> ValidationResult:
    """Validate URL format.
    
    Args:
        url: URL to validate
        schemes: Allowed URL schemes
        
    Returns:
        ValidationResult
    """
    rule = URLValidRule(schemes)
    valid = rule.validate(url)
    
    return ValidationResult(
        valid=valid,
        errors={"url": rule.get_error_message(url)} if not valid else {},
        validated_config={"url": url}
    )


# Example usage and testing
if __name__ == "__main__":
    from ..servers.filesystem import FileSystemServer
    from ..servers.websearch import WebSearchServer
    
    def test_validation():
        """Test validation functionality."""
        
        print("Testing validation utilities...")
        
        # Test server config validation
        print("\n1. Testing FileSystemServer validation:")
        
        # Valid config
        valid_config = {
            "root_dir": "/tmp",
            "max_file_size": 1024 * 1024,  # 1MB
            "allowed_extensions": [".txt", ".py"]
        }
        
        result = validate_server_config(FileSystemServer, valid_config)
        print(f"Valid config: {result.valid}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        # Invalid config
        invalid_config = {
            "root_dir": "/nonexistent/path",
            "max_file_size": -100,
            "allowed_extensions": ["txt", "py"]  # Missing dots
        }
        
        result = validate_server_config(FileSystemServer, invalid_config)
        print(f"Invalid config: {result.valid}")
        print(f"Errors: {result.errors}")
        print(f"Suggestions: {result.suggestions}")
        
        # Test WebSearchServer validation
        print("\n2. Testing WebSearchServer validation:")
        
        websearch_config = {
            "provider": "google",
            "max_results": 150,  # Too high
            # Missing api_key
        }
        
        result = validate_server_config(WebSearchServer, websearch_config)
        print(f"WebSearch config valid: {result.valid}")
        print(f"Errors: {result.errors}")
        print(f"Suggestions: {result.suggestions}")
        
        # Test individual validation functions
        print("\n3. Testing individual validators:")
        
        path_result = validate_file_path("/tmp", must_exist=True, must_be_dir=True)
        print(f"Path validation: {path_result.valid}")
        
        url_result = validate_url("https://example.com")
        print(f"URL validation: {url_result.valid}")
        
        bad_url_result = validate_url("not-a-url")
        print(f"Bad URL validation: {bad_url_result.valid}")
        print(f"URL error: {bad_url_result.errors}")
    
    # Run test
    # test_validation()
