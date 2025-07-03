"""
Delegant Context Extraction System
==================================

Automatic extraction and propagation of contextual information from Python code
to enhance MCP server understanding. Parses docstrings, type annotations, 
variable names, and comments to provide rich semantic context.
"""

import ast
import inspect
import re
from typing import Any, Dict, List, Optional, Union, get_type_hints, get_origin, get_args
from dataclasses import dataclass
from pydantic import BaseModel, Field
import logging

from .exceptions import ContextExtractionError
from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class ParameterInfo:
    """Information about a function/method parameter."""
    name: str
    type_annotation: Optional[str] = None
    description: Optional[str] = None
    default_value: Optional[Any] = None
    is_required: bool = True


class ContextMetadata(BaseModel):
    """Schema for extracted context information."""
    
    class_docstring: Optional[str] = Field(None, description="Class-level documentation")
    method_docstrings: Dict[str, str] = Field(
        default_factory=dict, 
        description="Method documentation by method name"
    )
    type_annotations: Dict[str, str] = Field(
        default_factory=dict, 
        description="Type hint information by attribute name"
    )
    variable_names: List[str] = Field(
        default_factory=list, 
        description="Variable identifiers in scope"
    )
    parameters: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Parameter metadata by method name"
    )
    purpose: Optional[str] = Field(None, description="Inferred purpose from context")
    domain: Optional[str] = Field(None, description="Problem domain (e.g., 'file_operations', 'web_search')")
    relationships: Dict[str, List[str]] = Field(
        default_factory=dict, 
        description="Relationships between components"
    )


class GoogleDocstringParser:
    """Parser for Google-style docstrings."""
    
    SECTIONS = ['Args', 'Arguments', 'Parameters', 'Param', 'Returns', 'Return', 
                'Yields', 'Yield', 'Raises', 'Note', 'Notes', 'Example', 'Examples']
    
    @classmethod
    def parse(cls, docstring: str) -> Dict[str, Any]:
        """Parse Google-style docstring into structured data."""
        if not docstring:
            return {}
        
        # Clean and normalize docstring
        lines = [line.strip() for line in docstring.strip().split('\n')]
        
        result = {
            'description': '',
            'parameters': {},
            'returns': None,
            'raises': {},
            'examples': [],
            'notes': []
        }
        
        current_section = 'description'
        current_content = []
        
        for line in lines:
            # Check if line starts a new section
            section_match = re.match(r'^(Args|Arguments|Parameters|Param|Returns?|Yields?|Raises|Notes?|Examples?):\s*$', line)
            
            if section_match:
                # Save previous section content
                cls._save_section_content(result, current_section, current_content)
                
                # Start new section
                current_section = section_match.group(1).lower()
                if current_section in ['args', 'arguments', 'param']:
                    current_section = 'parameters'
                elif current_section == 'returns':
                    current_section = 'return'
                elif current_section in ['notes']:
                    current_section = 'note'
                elif current_section in ['examples']:
                    current_section = 'example'
                    
                current_content = []
            else:
                current_content.append(line)
        
        # Save final section
        cls._save_section_content(result, current_section, current_content)
        
        return result
    
    @classmethod
    def _save_section_content(cls, result: Dict[str, Any], section: str, content: List[str]) -> None:
        """Save section content to result dictionary."""
        if not content:
            return
        
        content_text = '\n'.join(content).strip()
        
        if section == 'description':
            result['description'] = content_text
        elif section == 'parameters':
            result['parameters'].update(cls._parse_parameters(content_text))
        elif section == 'return':
            result['returns'] = content_text
        elif section == 'raises':
            result['raises'].update(cls._parse_raises(content_text))
        elif section == 'note':
            result['notes'].append(content_text)
        elif section == 'example':
            result['examples'].append(content_text)
    
    @classmethod
    def _parse_parameters(cls, content: str) -> Dict[str, str]:
        """Parse parameter descriptions from docstring content."""
        parameters = {}
        
        # Match parameter patterns like "param_name (type): description"
        param_pattern = re.compile(r'^\s*(\w+)(?:\s*\([^)]+\))?\s*:\s*(.+)$', re.MULTILINE)
        
        for match in param_pattern.finditer(content):
            param_name = match.group(1)
            param_desc = match.group(2).strip()
            parameters[param_name] = param_desc
        
        return parameters
    
    @classmethod
    def _parse_raises(cls, content: str) -> Dict[str, str]:
        """Parse exception descriptions from docstring content."""
        raises = {}
        
        # Match exception patterns like "ExceptionType: description"
        exception_pattern = re.compile(r'^\s*(\w+(?:\.\w+)*)\s*:\s*(.+)$', re.MULTILINE)
        
        for match in exception_pattern.finditer(content):
            exception_type = match.group(1)
            exception_desc = match.group(2).strip()
            raises[exception_type] = exception_desc
        
        return raises


class TypeAnnotationExtractor:
    """Extract and format type annotation information."""
    
    @classmethod
    def extract_from_class(cls, target_class: type) -> Dict[str, str]:
        """Extract type annotations from class attributes."""
        annotations = {}
        
        try:
            # Get type hints for the class
            type_hints = get_type_hints(target_class)
            
            for attr_name, type_hint in type_hints.items():
                annotations[attr_name] = cls._format_type_annotation(type_hint)
                
        except Exception as e:
            logger.warning(f"Failed to extract type hints from {target_class.__name__}: {e}")
        
        return annotations
    
    @classmethod
    def extract_from_function(cls, func) -> Dict[str, str]:
        """Extract type annotations from function parameters and return type."""
        annotations = {}
        
        try:
            # Get function signature
            sig = inspect.signature(func)
            
            # Extract parameter types
            for param_name, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    annotations[param_name] = cls._format_type_annotation(param.annotation)
            
            # Extract return type
            if sig.return_annotation != inspect.Signature.empty:
                annotations['return'] = cls._format_type_annotation(sig.return_annotation)
                
        except Exception as e:
            logger.warning(f"Failed to extract type hints from function {func.__name__}: {e}")
        
        return annotations
    
    @classmethod
    def _format_type_annotation(cls, type_hint) -> str:
        """Format type annotation as human-readable string."""
        try:
            # Handle basic types
            if hasattr(type_hint, '__name__'):
                return type_hint.__name__
            
            # Handle generic types (List, Dict, Optional, etc.)
            origin = get_origin(type_hint)
            args = get_args(type_hint)
            
            if origin is not None:
                origin_name = getattr(origin, '__name__', str(origin))
                
                if args:
                    arg_names = [cls._format_type_annotation(arg) for arg in args]
                    return f"{origin_name}[{', '.join(arg_names)}]"
                else:
                    return origin_name
            
            # Fallback to string representation
            return str(type_hint)
            
        except Exception:
            return str(type_hint)


class PurposeInferrer:
    """Infer purpose and domain from context clues."""
    
    DOMAIN_KEYWORDS = {
        'file_operations': ['file', 'directory', 'path', 'read', 'write', 'delete', 'create'],
        'web_search': ['search', 'query', 'web', 'internet', 'url', 'website'],
        'git_operations': ['git', 'repository', 'commit', 'branch', 'merge', 'clone'],
        'data_analysis': ['data', 'analysis', 'statistics', 'chart', 'graph', 'plot'],
        'text_processing': ['text', 'string', 'parse', 'format', 'tokenize', 'nlp'],
        'terminal_operations': ['command', 'terminal', 'shell', 'execute', 'process', 'stdout']
    }
    
    @classmethod
    def infer_domain(cls, context_text: str) -> Optional[str]:
        """Infer problem domain from context text."""
        if not context_text:
            return None
        
        text_lower = context_text.lower()
        domain_scores = {}
        
        for domain, keywords in cls.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    @classmethod
    def infer_purpose(cls, docstring: str, class_name: str, variable_names: List[str]) -> Optional[str]:
        """Infer overall purpose from available context."""
        context_parts = []
        
        # Use docstring description
        if docstring:
            parsed = GoogleDocstringParser.parse(docstring)
            if parsed.get('description'):
                context_parts.append(parsed['description'])
        
        # Use class name as context
        if class_name:
            # Convert CamelCase to readable text
            readable_name = re.sub(r'([A-Z])', r' \1', class_name).strip()
            context_parts.append(f"Class: {readable_name}")
        
        # Use variable names as context clues
        if variable_names:
            context_parts.append(f"Variables: {', '.join(variable_names)}")
        
        if context_parts:
            return '. '.join(context_parts)
        
        return None


class ContextExtractor:
    """Main context extraction system."""
    
    def __init__(self, max_context_size: Optional[int] = None):
        self.max_context_size = max_context_size or get_config().max_context_size
        self.docstring_parser = GoogleDocstringParser()
        self.type_extractor = TypeAnnotationExtractor()
        self.purpose_inferrer = PurposeInferrer()
    
    def extract_from_class(self, target_class: type, server_attribute_name: Optional[str] = None) -> ContextMetadata:
        """Extract context from a class definition."""
        try:
            context = ContextMetadata()
            
            # Extract class docstring
            if target_class.__doc__:
                context.class_docstring = self._truncate_content(target_class.__doc__)
            
            # Extract type annotations for class attributes
            context.type_annotations = self.type_extractor.extract_from_class(target_class)
            
            # Extract method docstrings
            for attr_name in dir(target_class):
                if not attr_name.startswith('_'):
                    attr = getattr(target_class, attr_name)
                    if inspect.ismethod(attr) or inspect.isfunction(attr):
                        if hasattr(attr, '__doc__') and attr.__doc__:
                            context.method_docstrings[attr_name] = self._truncate_content(attr.__doc__)
            
            # Extract variable names (class attributes)
            context.variable_names = [name for name in context.type_annotations.keys() 
                                    if not name.startswith('_')]
            
            # Infer purpose and domain
            full_context_text = ' '.join([
                context.class_docstring or '',
                ' '.join(context.method_docstrings.values()),
                ' '.join(context.variable_names)
            ])
            
            context.purpose = self.purpose_inferrer.infer_purpose(
                context.class_docstring, 
                target_class.__name__, 
                context.variable_names
            )
            context.domain = self.purpose_inferrer.infer_domain(full_context_text)
            
            # If this is for a specific server attribute, add relationship info
            if server_attribute_name:
                context.relationships['server_attribute'] = [server_attribute_name]
            
            return context
            
        except Exception as e:
            raise ContextExtractionError(
                target_name=target_class.__name__,
                extraction_type="class_extraction",
                original_error=e
            )
    
    def extract_from_function(self, func, context_hint: Optional[str] = None) -> ContextMetadata:
        """Extract context from a function definition."""
        try:
            context = ContextMetadata()
            
            # Extract function docstring
            if func.__doc__:
                parsed_doc = self.docstring_parser.parse(func.__doc__)
                context.method_docstrings[func.__name__] = self._truncate_content(func.__doc__)
                
                # Extract parameter information
                if parsed_doc.get('parameters'):
                    context.parameters[func.__name__] = parsed_doc['parameters']
            
            # Extract type annotations
            context.type_annotations = self.type_extractor.extract_from_function(func)
            
            # Extract parameter names
            try:
                sig = inspect.signature(func)
                context.variable_names = list(sig.parameters.keys())
            except Exception:
                pass
            
            # Add context hint if provided
            if context_hint:
                if context.class_docstring:
                    context.class_docstring += f"\n\nContext: {context_hint}"
                else:
                    context.class_docstring = f"Context: {context_hint}"
            
            # Infer domain
            full_context_text = ' '.join([
                func.__doc__ or '',
                context_hint or '',
                ' '.join(context.variable_names)
            ])
            context.domain = self.purpose_inferrer.infer_domain(full_context_text)
            
            return context
            
        except Exception as e:
            raise ContextExtractionError(
                target_name=func.__name__,
                extraction_type="function_extraction",
                original_error=e
            )
    
    def merge_contexts(self, *contexts: ContextMetadata) -> ContextMetadata:
        """Merge multiple context metadata objects."""
        merged = ContextMetadata()
        
        for context in contexts:
            # Merge docstrings
            if context.class_docstring:
                if merged.class_docstring:
                    merged.class_docstring += f"\n\n{context.class_docstring}"
                else:
                    merged.class_docstring = context.class_docstring
            
            # Merge dictionaries
            merged.method_docstrings.update(context.method_docstrings)
            merged.type_annotations.update(context.type_annotations)
            merged.parameters.update(context.parameters)
            
            # Merge lists
            merged.variable_names.extend(context.variable_names)
            
            # Merge relationships
            for key, values in context.relationships.items():
                if key in merged.relationships:
                    merged.relationships[key].extend(values)
                else:
                    merged.relationships[key] = values.copy()
            
            # Use first non-None purpose and domain
            if not merged.purpose and context.purpose:
                merged.purpose = context.purpose
            if not merged.domain and context.domain:
                merged.domain = context.domain
        
        # Remove duplicates from lists
        merged.variable_names = list(set(merged.variable_names))
        
        return merged
    
    def _truncate_content(self, content: str) -> str:
        """Truncate content to max_context_size if needed."""
        if len(content.encode('utf-8')) <= self.max_context_size:
            return content
        
        # Truncate and add indicator
        truncated = content.encode('utf-8')[:self.max_context_size - 20].decode('utf-8', errors='ignore')
        return f"{truncated}... [truncated]"


# Global context extractor instance
_global_extractor: Optional[ContextExtractor] = None


def get_context_extractor() -> ContextExtractor:
    """Get the global context extractor instance."""
    global _global_extractor
    
    if _global_extractor is None:
        _global_extractor = ContextExtractor()
    
    return _global_extractor


# Example usage and testing
if __name__ == "__main__":
    class ExampleAgent:
        """Agent specialized in document analysis and research.
        
        This agent combines file operations with web search to provide
        comprehensive document analysis capabilities.
        
        Args:
            instruction: The agent's primary instruction
            
        Returns:
            Configured agent instance
            
        Examples:
            >>> agent = ExampleAgent()
            >>> result = agent.analyze_document("report.pdf")
        """
        
        def analyze_document(self, file_path: str, search_terms: Optional[List[str]] = None) -> Dict[str, Any]:
            """Analyze a document and optionally search for related information.
            
            Args:
                file_path: Path to the document to analyze
                search_terms: Optional terms to search for additional context
                
            Returns:
                Analysis results with insights and related information
                
            Raises:
                FileNotFoundError: If the document doesn't exist
            """
            pass
    
    # Test context extraction
    extractor = get_context_extractor()
    context = extractor.extract_from_class(ExampleAgent)
    
    print("Extracted Context:")
    print(f"Class Docstring: {context.class_docstring}")
    print(f"Type Annotations: {context.type_annotations}")
    print(f"Method Docstrings: {context.method_docstrings}")
    print(f"Variable Names: {context.variable_names}")
    print(f"Purpose: {context.purpose}")
    print(f"Domain: {context.domain}")
