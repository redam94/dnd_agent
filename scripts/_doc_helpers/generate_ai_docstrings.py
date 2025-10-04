"""
Multi-AI Docstring Enhancer with OpenAI and Claude Support
Automatically generates detailed, well-formatted docstrings using either OpenAI or Claude
"""

import ast
import json
import re
import os
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
import aiohttp
from abc import ABC, abstractmethod
import logging
from dotenv import load_dotenv
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocstringStyle(Enum):
    """Supported docstring styles."""
    NUMPY = "numpy"
    GOOGLE = "google"
    SPHINX = "sphinx"


class AIProvider(Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    AUTO = "auto"  # Automatically detect based on available API keys


@dataclass
class DocstringComponents:
    """Components of a docstring."""
    summary: str = ""
    description: str = ""
    params: Dict[str, Dict[str, str]] = field(default_factory=dict)
    returns: Dict[str, str] = field(default_factory=dict)
    raises: Dict[str, str] = field(default_factory=dict)
    examples: str = ""
    notes: str = ""
    attributes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    see_also: List[str] = field(default_factory=list)


class BaseAIClient(ABC):
    """Base class for AI API clients."""
    
    @abstractmethod
    async def generate_docstring(self, prompt: str) -> str:
        """Generate a docstring using the AI provider."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (has API key)."""
        pass
    
    def _clean_response(self, response: str) -> str:
        """Clean the AI response to ensure it's just docstring content."""
        # Remove any markdown code blocks
        response = re.sub(r'```python\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Remove triple quotes if present
        response = re.sub(r'^"""\s*', '', response)
        response = re.sub(r'\s*"""$', '', response)
        response = re.sub(r"^'''\s*", '', response)
        response = re.sub(r"\s*'''$", '', response)
        
        return response.strip()


class OpenAIClient(BaseAIClient):
    """OpenAI API client for docstring generation."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize OpenAI client.
        
        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If not provided, looks for OPENAI_API_KEY env var.
        model : str, optional
            Model to use. Defaults to gpt-4 or gpt-3.5-turbo.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        # Default model selection
        if model:
            self.model = model
        else:
            # Use GPT-4 if specified in env, otherwise GPT-3.5
            self.model = os.getenv('OPENAI_MODEL', 'gpt-4o')
            
        # Available models
        self.available_models = [
            'gpt-4-turbo-preview',
            'gpt-4',
            'gpt-4o',
            'gpt-4-32k',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-16k'
        ]
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return bool(self.api_key)
    
    async def generate_docstring(self, prompt: str) -> str:
        """Generate docstring using OpenAI."""
        if not self.is_available():
            raise ValueError("OpenAI API key not configured")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Python developer who writes comprehensive, detailed docstrings. Always follow the specified style guide exactly."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,  # Lower for consistent documentation
            "max_tokens": 2000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    return self._clean_response(content)
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API call failed: {response.status} - {error_text}")


class ClaudeClient(BaseAIClient):
    """Claude/Anthropic API client for docstring generation."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Claude client.
        
        Parameters
        ----------
        api_key : str, optional
            Anthropic API key. If not provided, looks for ANTHROPIC_API_KEY env var.
        model : str, optional
            Model to use. Defaults to claude-3-sonnet.
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.api_url = "https://api.anthropic.com/v1/messages"
        
        # Default model selection
        if model:
            self.model = model
        else:
            self.model = os.getenv('CLAUDE_MODEL', 'claude-3-sonnet-20240229')
        
        # Available models
        self.available_models = [
            'claude-3-opus-20240229',
            'claude-3-sonnet-20240229',
            'claude-3-haiku-20240307'
        ]
    
    def is_available(self) -> bool:
        """Check if Claude is available."""
        return bool(self.api_key)
    
    async def generate_docstring(self, prompt: str) -> str:
        """Generate docstring using Claude."""
        if not self.is_available():
            raise ValueError("Claude API key not configured")
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model,
            "max_tokens": 2000,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result['content'][0]['text']
                    return self._clean_response(content)
                else:
                    error_text = await response.text()
                    raise Exception(f"Claude API call failed: {response.status} - {error_text}")


class MultiAIDocstringEnhancer:
    """
    Enhance docstrings using multiple AI providers.
    
    Supports both OpenAI and Claude/Anthropic APIs with automatic
    fallback and provider selection based on availability.
    """
    
    def __init__(self, 
                 provider: AIProvider = AIProvider.AUTO,
                 openai_api_key: str = None,
                 claude_api_key: str = None,
                 openai_model: str = None,
                 claude_model: str = None):
        """
        Initialize the multi-AI docstring enhancer.
        
        Parameters
        ----------
        provider : AIProvider
            Which AI provider to use. AUTO will detect based on available keys.
        openai_api_key : str, optional
            OpenAI API key. Falls back to OPENAI_API_KEY env var.
        claude_api_key : str, optional
            Claude API key. Falls back to ANTHROPIC_API_KEY env var.
        openai_model : str, optional
            OpenAI model to use (e.g., 'gpt-4', 'gpt-3.5-turbo').
        claude_model : str, optional
            Claude model to use (e.g., 'claude-3-opus-20240229').
        """
        # Initialize clients
        self.openai_client = OpenAIClient(openai_api_key, openai_model)
        self.claude_client = ClaudeClient(claude_api_key, claude_model)
        
        # Determine which provider to use
        self.provider = self._determine_provider(provider)
        
        if self.provider is None:
            raise ValueError(
                "No AI provider available. Please set either OPENAI_API_KEY or "
                "ANTHROPIC_API_KEY environment variable, or pass API keys directly."
            )
        
        logger.info(f"Using AI provider: {self.provider.value}")
    
    def _determine_provider(self, requested: AIProvider) -> Optional[AIProvider]:
        """Determine which provider to use based on availability."""
        if requested == AIProvider.AUTO:
            # Try Claude first (often better for documentation)
            if self.claude_client.is_available():
                return AIProvider.CLAUDE
            elif self.openai_client.is_available():
                return AIProvider.OPENAI
            else:
                return None
        elif requested == AIProvider.OPENAI:
            if self.openai_client.is_available():
                return AIProvider.OPENAI
            else:
                logger.warning("OpenAI requested but not available")
                return None
        elif requested == AIProvider.CLAUDE:
            if self.claude_client.is_available():
                return AIProvider.CLAUDE
            else:
                logger.warning("Claude requested but not available")
                return None
        
        return None
    
    async def enhance_docstring(self,
                               function_code: str,
                               style: str = "numpy",
                               existing_docstring: str = None,
                               context: Dict[str, Any] = None) -> str:
        """
        Enhance or create a docstring for the given function code.
        
        Parameters
        ----------
        function_code : str
            The Python function or class code to document.
        style : str
            Docstring style: 'numpy', 'google', or 'sphinx'.
        existing_docstring : str, optional
            Existing docstring to enhance (if any).
        context : Dict[str, Any], optional
            Additional context from code analysis.
        
        Returns
        -------
        str
            Enhanced docstring content (without triple quotes).
        """
        prompt = self._create_prompt(function_code, style, existing_docstring, context)
        
        # Use the selected provider
        if self.provider == AIProvider.OPENAI:
            response = await self.openai_client.generate_docstring(prompt)
        elif self.provider == AIProvider.CLAUDE:
            response = await self.claude_client.generate_docstring(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        return response
    
    def _create_prompt(self, code: str, style: str, 
                      existing: str = None, context: Dict = None) -> str:
        """Create a detailed prompt for the AI."""
        style_guide = self._get_style_guide(style)
        
        # Add context information if available
        context_info = ""
        if context:
            if context.get('complexity', 0) > 10:
                context_info += "\nNote: This is a complex function with cyclomatic complexity > 10."
            if context.get('is_async'):
                context_info += "\nNote: This is an async function."
            if context.get('is_generator'):
                context_info += "\nNote: This is a generator function."
            if context.get('raises'):
                context_info += f"\nDetected exceptions: {', '.join(context['raises'])}"
        
        prompt = f"""You are an expert Python developer creating comprehensive docstrings.

Generate a detailed {style}-style docstring for this Python code:

```python
{code}
```

{f'Current docstring to enhance: {existing}' if existing else 'No existing docstring.'}
{context_info}

Requirements:
1. Follow {style} docstring conventions exactly
2. Include:
   - Clear, concise one-line summary
   - Detailed description of functionality
   - All parameters with types and descriptions
   - Return value with type and description
   - Exceptions that may be raised
   - At least one practical example
   - Important notes or warnings
3. Infer types from the code when not explicitly annotated
4. Make descriptions helpful and specific, not generic
5. Include edge cases and important behaviors
6. For async functions, mention async behavior
7. For generators, explain what is yielded

{style_guide}

IMPORTANT: Return ONLY the docstring content, without triple quotes.
Do not include ```python``` markers or any other formatting.
Just the pure docstring text that will go between the triple quotes."""
        
        return prompt
    
    def _get_style_guide(self, style: str) -> str:
        """Get style-specific formatting guide."""
        if style.lower() == "numpy":
            return """
NumPy Style Guide:
- Summary line (one line, imperative mood, no period)
- Blank line
- Extended description (if needed)
- Blank line
- Parameters section with "Parameters" underlined with dashes
- Each parameter: "name : type" then indented description on next lines
- Returns section with "Returns" underlined with dashes
- Raises section with "Raises" underlined with dashes
- Examples section with "Examples" underlined with dashes
- Notes section if needed

Example structure:
Summary line in imperative mood

Extended description explaining the function's purpose
and behavior in detail.

Parameters
----------
param1 : str
    Description of param1.
param2 : int, optional
    Description of param2. Default is 10.

Returns
-------
bool
    Description of return value.

Raises
------
ValueError
    When input is invalid.

Examples
--------
>>> result = function_name("test", 5)
>>> print(result)
True

Notes
-----
Additional information about the function.
"""
        elif style.lower() == "google":
            return """
Google Style Guide:
- Summary line (one line)
- Blank line
- Extended description
- Blank line
- Args: section with indented parameter descriptions
- Returns: section with type and description
- Raises: section with exceptions
- Example: section with usage examples

Example structure:
Summary line.

Extended description of the function.

Args:
    param1 (str): Description of param1.
    param2 (int, optional): Description of param2. Defaults to 10.

Returns:
    bool: Description of return value.

Raises:
    ValueError: Description of when this is raised.

Example:
    >>> result = function_name("test", 5)
    >>> print(result)
    True
"""
        else:  # sphinx
            return """
Sphinx Style Guide:
- Summary line
- Blank line
- Extended description
- :param tags for each parameter
- :type tags for parameter types
- :returns: tag for return description
- :rtype: tag for return type
- :raises: tags for exceptions

Example structure:
Summary line.

Extended description of the function.

:param param1: Description of param1
:type param1: str
:param param2: Description of param2
:type param2: int
:returns: Description of return value
:rtype: bool
:raises ValueError: Description of when raised
"""


class CodeAnalyzer:
    """Analyze Python code to provide context for better documentation."""
    
    def analyze_node(self, node: ast.AST, source_lines: List[str] = None) -> Dict[str, Any]:
        """
        Analyze an AST node to extract useful context.
        
        Parameters
        ----------
        node : ast.AST
            The AST node to analyze.
        source_lines : List[str], optional
            Source code lines for additional context.
        
        Returns
        -------
        Dict[str, Any]
            Analysis results including parameters, return types, complexity, etc.
        """
        if isinstance(node, ast.ClassDef):
            return self._analyze_class(node, source_lines)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._analyze_function(node, source_lines)
        else:
            return {}
    
    def _analyze_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                         source_lines: List[str] = None) -> Dict[str, Any]:
        """Analyze a function node."""
        analysis = {
            'name': node.name,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'is_generator': self._is_generator(node),
            'is_property': self._has_decorator(node, 'property'),
            'is_staticmethod': self._has_decorator(node, 'staticmethod'),
            'is_classmethod': self._has_decorator(node, 'classmethod'),
            'parameters': self._extract_parameters(node),
            'return_type': self._extract_return_type(node),
            'raises': self._extract_exceptions(node),
            'complexity': self._calculate_complexity(node),
            'line_count': node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
        }
        
        return analysis
    
    def _analyze_class(self, node: ast.ClassDef, source_lines: List[str] = None) -> Dict[str, Any]:
        """Analyze a class node."""
        analysis = {
            'name': node.name,
            'is_dataclass': self._has_decorator(node, 'dataclass'),
            'bases': [self._get_name(base) for base in node.bases],
            'methods': [],
            'properties': [],
            'class_attributes': self._extract_class_attributes(node),
            'instance_attributes': self._extract_instance_attributes(node)
        }
        
        # Analyze methods
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_analysis = self._analyze_function(item, source_lines)
                if method_analysis.get('is_property'):
                    analysis['properties'].append(method_analysis['name'])
                else:
                    analysis['methods'].append(method_analysis['name'])
        
        return analysis
    
    def _is_generator(self, node: ast.FunctionDef) -> bool:
        """Check if function is a generator."""
        for item in ast.walk(node):
            if isinstance(item, (ast.Yield, ast.YieldFrom)):
                return True
        return False
    
    def _has_decorator(self, node: Union[ast.FunctionDef, ast.ClassDef], name: str) -> bool:
        """Check if node has a specific decorator."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == name:
                return True
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == name:
                    return True
        return False
    
    def _extract_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract parameter information from function."""
        params = []
        args = node.args
        
        # Regular arguments
        for i, arg in enumerate(args.args):
            param = {
                'name': arg.arg,
                'type': self._get_annotation(arg.annotation),
                'has_default': False
            }
            
            # Check for defaults
            default_offset = len(args.args) - len(args.defaults)
            if i >= default_offset:
                param['has_default'] = True
                default_idx = i - default_offset
                if default_idx < len(args.defaults):
                    param['default'] = self._get_default_repr(args.defaults[default_idx])
            
            params.append(param)
        
        # *args
        if args.vararg:
            params.append({
                'name': f"*{args.vararg.arg}",
                'type': self._get_annotation(args.vararg.annotation),
                'is_vararg': True
            })
        
        # Keyword-only arguments
        for i, arg in enumerate(args.kwonlyargs):
            param = {
                'name': arg.arg,
                'type': self._get_annotation(arg.annotation),
                'has_default': i < len(args.kw_defaults) and args.kw_defaults[i] is not None,
                'keyword_only': True
            }
            if param['has_default']:
                param['default'] = self._get_default_repr(args.kw_defaults[i])
            params.append(param)
        
        # **kwargs
        if args.kwarg:
            params.append({
                'name': f"**{args.kwarg.arg}",
                'type': self._get_annotation(args.kwarg.annotation),
                'is_kwarg': True
            })
        
        return params
    
    def _extract_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation."""
        if node.returns:
            return self._get_annotation(node.returns)
        return None
    
    def _extract_exceptions(self, node: ast.FunctionDef) -> List[str]:
        """Extract exceptions that might be raised."""
        exceptions = set()
        for item in ast.walk(node):
            if isinstance(item, ast.Raise):
                if isinstance(item.exc, ast.Call):
                    if isinstance(item.exc.func, ast.Name):
                        exceptions.add(item.exc.func.id)
                elif isinstance(item.exc, ast.Name):
                    exceptions.add(item.exc.id)
        return list(exceptions)
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for item in ast.walk(node):
            if isinstance(item, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(item, ast.BoolOp):
                complexity += len(item.values) - 1
        return complexity
    
    def _extract_class_attributes(self, node: ast.ClassDef) -> Dict[str, str]:
        """Extract class-level attributes."""
        attributes = {}
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                attributes[item.target.id] = self._get_annotation(item.annotation)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes[target.id] = self._infer_type(item.value)
        return attributes
    
    def _extract_instance_attributes(self, node: ast.ClassDef) -> Dict[str, str]:
        """Extract instance attributes from __init__."""
        attributes = {}
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                for stmt in ast.walk(item):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute):
                                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                    attributes[target.attr] = self._infer_type(stmt.value)
        return attributes
    
    def _get_annotation(self, annotation: Optional[ast.AST]) -> Optional[str]:
        """Get type annotation as string."""
        if not annotation:
            return None
        
        try:
            # Try to unparse the annotation (Python 3.9+)
            if hasattr(ast, 'unparse'):
                return ast.unparse(annotation)
            else:
                # Fallback for older Python versions
                return self._annotation_to_string(annotation)
        except:
            return None
    
    def _annotation_to_string(self, annotation: ast.AST) -> str:
        """Convert annotation AST to string (for older Python)."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Subscript):
            base = self._annotation_to_string(annotation.value)
            index = self._annotation_to_string(annotation.slice)
            return f"{base}[{index}]"
        elif isinstance(annotation, ast.Attribute):
            value = self._annotation_to_string(annotation.value)
            return f"{value}.{annotation.attr}"
        elif isinstance(annotation, ast.Tuple):
            elements = [self._annotation_to_string(e) for e in annotation.elts]
            return f"({', '.join(elements)})"
        else:
            return "Any"
    
    def _get_default_repr(self, default: ast.AST) -> str:
        """Get string representation of default value."""
        if isinstance(default, ast.Constant):
            if isinstance(default.value, str):
                return f'"{default.value}"'
            else:
                return str(default.value)
        elif isinstance(default, ast.Name):
            return default.id
        elif isinstance(default, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            return type(default).__name__.lower() + "()"
        else:
            return "..."
    
    def _infer_type(self, value: ast.AST) -> str:
        """Infer type from value."""
        if isinstance(value, ast.Constant):
            return type(value.value).__name__
        elif isinstance(value, ast.List):
            return "List"
        elif isinstance(value, ast.Dict):
            return "Dict"
        elif isinstance(value, ast.Set):
            return "Set"
        elif isinstance(value, ast.Tuple):
            return "Tuple"
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            return value.func.id
        else:
            return "Any"
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from various AST nodes."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return "Unknown"


class DocstringProcessor:
    """
    Main processor for enhancing Python file docstrings.
    
    Coordinates between code analysis, AI generation, and formatting.
    """
    
    def __init__(self,
                 provider: AIProvider = AIProvider.AUTO,
                 style: DocstringStyle = DocstringStyle.NUMPY,
                 openai_api_key: str = None,
                 claude_api_key: str = None,
                 openai_model: str = None,
                 claude_model: str = None):
        """
        Initialize the docstring processor.
        
        Parameters
        ----------
        provider : AIProvider
            AI provider to use (OPENAI, CLAUDE, or AUTO).
        style : DocstringStyle
            Docstring style to generate.
        openai_api_key : str, optional
            OpenAI API key.
        claude_api_key : str, optional
            Claude/Anthropic API key.
        openai_model : str, optional
            Specific OpenAI model to use.
        claude_model : str, optional
            Specific Claude model to use.
        """
        self.enhancer = MultiAIDocstringEnhancer(
            provider=provider,
            openai_api_key=openai_api_key,
            claude_api_key=claude_api_key,
            openai_model=openai_model,
            claude_model=claude_model
        )
        self.analyzer = CodeAnalyzer()
        self.style = style
    
    async def process_file(self, file_path: str, output_path: str = None,
                          skip_existing: bool = False):
        """
        Process a Python file to enhance all docstrings.
        
        Parameters
        ----------
        file_path : str
            Path to the input Python file.
        output_path : str, optional
            Path to save the enhanced file. If None, overwrites input.
        skip_existing : bool
            Skip functions/classes that already have docstrings.
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        enhanced_content = await self.enhance_content(content, skip_existing)
        
        output = output_path or file_path
        with open(output, 'w') as f:
            f.write(enhanced_content)
        
        logger.info(f"✅ Enhanced docstrings in {output}")
    
    async def enhance_content(self, content: str, skip_existing: bool = False) -> str:
        """
        Enhance docstrings in Python content.
        
        Parameters
        ----------
        content : str
            Python source code content.
        skip_existing : bool
            Skip items that already have docstrings.
        
        Returns
        -------
        str
            Enhanced content with improved docstrings.
        """
        tree = ast.parse(content)
        lines = content.split('\n')
        
        # Collect all items to process
        items_to_process = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                existing_docstring = ast.get_docstring(node)
                
                # Skip if requested and has docstring
                if skip_existing and existing_docstring:
                    continue
                
                items_to_process.append({
                    'node': node,
                    'type': 'class' if isinstance(node, ast.ClassDef) else 'function',
                    'name': node.name,
                    'line_start': node.lineno - 1,
                    'line_end': getattr(node, 'end_lineno', node.lineno + 20) - 1,
                    'existing_docstring': existing_docstring
                })
        
        # Sort by line number (reverse) to process from bottom to top
        items_to_process.sort(key=lambda x: x['line_start'], reverse=True)
        
        # Process each item
        for item in items_to_process:
            logger.info(f"🔄 Enhancing {item['type']} '{item['name']}'...")
            
            # Extract code
            code_lines = lines[item['line_start']:item['line_end']+1]
            code = '\n'.join(code_lines)
            
            # Analyze the node for context
            context = self.analyzer.analyze_node(item['node'], lines)
            
            try:
                # Generate enhanced docstring
                enhanced_docstring = await self.enhancer.enhance_docstring(
                    code,
                    self.style.value,
                    item['existing_docstring'],
                    context
                )
                
                # Format with proper indentation
                formatted = self._format_docstring(
                    enhanced_docstring,
                    item['node'],
                    lines
                )
                
                # Apply the docstring
                lines = self._apply_docstring(
                    lines,
                    item['node'],
                    formatted,
                    item['existing_docstring'] is not None
                )
                
                logger.info(f"✅ Enhanced '{item['name']}'")
                
            except Exception as e:
                logger.error(f"❌ Failed to enhance '{item['name']}': {e}")
                continue
        
        return '\n'.join(lines)
    
    def _format_docstring(self, docstring_content: str, 
                         node: ast.AST, lines: List[str]) -> List[str]:
        """Format docstring with proper indentation."""
        # Determine indentation
        if hasattr(node, 'lineno'):
            def_line = lines[node.lineno - 1]
            base_indent = len(def_line) - len(def_line.lstrip())
            body_indent = ' ' * (base_indent + 4)
        else:
            body_indent = '    '
        
        # Split content into lines
        content_lines = docstring_content.split('\n')
        
        # Build formatted docstring
        formatted_lines = [f'{body_indent}"""']
        
        for line in content_lines:
            if line.strip():
                formatted_lines.append(f'{body_indent}{line}')
            else:
                formatted_lines.append('')  # Empty line
        
        formatted_lines.append(f'{body_indent}"""')
        
        return formatted_lines
    
    def _apply_docstring(self, lines: List[str], node: ast.AST,
                        formatted_docstring: List[str], 
                        has_existing: bool) -> List[str]:
        """Apply the formatted docstring to the source lines."""
        if has_existing and node.body and isinstance(node.body[0], ast.Expr):
            # Replace existing docstring
            docstring_node = node.body[0]
            start = docstring_node.lineno - 1
            end = getattr(docstring_node, 'end_lineno', start) - 1
            
            # Replace the lines
            lines[start:end+1] = formatted_docstring
        else:
            # Insert new docstring
            insert_pos = node.lineno
            
            # Find the line with the colon
            for i in range(node.lineno - 1, min(node.lineno + 5, len(lines))):
                if i < len(lines) and lines[i].rstrip().endswith(':'):
                    insert_pos = i + 1
                    break
            
            # Insert the docstring lines
            for i, line in enumerate(formatted_docstring):
                lines.insert(insert_pos + i, line)
        
        return lines

async def process_with_progress(processor: DocstringProcessor, skip_existing: bool = False):
    files = list(Path("src").rglob("*.py"))
    
    for i, file in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing {file.name}...")
        try:
            await processor.process_file(str(file), skip_existing=skip_existing)
            print(f"✅ Success: {file.name}")
        except Exception as e:
            print(f"❌ Failed: {file.name} - {e}")

# Command-line interface
async def main():
    """
    Command-line interface for the docstring enhancer.
    
    Supports both OpenAI and Claude APIs with automatic detection.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance Python docstrings with AI')
    parser.add_argument('file', help='Python file to process or directory')
    parser.add_argument('-p', '--provider', choices=['openai', 'claude', 'auto'],
                       default='auto', help='AI provider to use')
    parser.add_argument('-s', '--style', choices=['numpy', 'google', 'sphinx'],
                       default='numpy', help='Docstring style')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip functions/classes that already have docstrings')
    parser.add_argument('--openai-model', help='OpenAI model (e.g., gpt-4, gpt-3.5-turbo)')
    parser.add_argument('--claude-model', help='Claude model (e.g., claude-3-opus-20240229)')
    
    args = parser.parse_args()
    
    # Convert provider string to enum
    provider_map = {
        'openai': AIProvider.OPENAI,
        'claude': AIProvider.CLAUDE,
        'auto': AIProvider.AUTO
    }
    
    # Convert style string to enum
    style_map = {
        'numpy': DocstringStyle.NUMPY,
        'google': DocstringStyle.GOOGLE,
        'sphinx': DocstringStyle.SPHINX
    }
    
    # Create processor
    processor = DocstringProcessor(
        provider=provider_map[args.provider],
        style=style_map[args.style],
        openai_model=args.openai_model,
        claude_model=args.claude_model
    )
    
    if Path(args.file).is_dir():
        # Process all .py files in directory
        await process_with_progress(processor, skip_existing=args.skip_existing)
    else:
        # Process single file
        await processor.process_file(args.file, skip_existing=args.skip_existing)


if __name__ == "__main__":
    
    asyncio.run(main())
    
