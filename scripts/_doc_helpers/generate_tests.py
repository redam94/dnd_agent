"""
AI-Powered Test Generator for Python Modules
Automatically generates comprehensive test suites with edge cases, mocks, and fixtures.
"""

import os
import ast
import json
import re
import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import hashlib
from enum import Enum
import importlib.util
import inspect
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestFramework(Enum):
    """Supported test frameworks."""
    PYTEST = "pytest"
    UNITTEST = "unittest"

class TestType(Enum):
    """Types of tests to generate."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PARAMETRIZED = "parametrized"
    EDGE_CASE = "edge_case"
    EXCEPTION = "exception"
    MOCK = "mock"

@dataclass
class FunctionSignature:
    """Detailed function signature information."""
    name: str
    params: List[Dict[str, Any]]  # param name, type, default
    return_type: Optional[str]
    decorators: List[str]
    is_async: bool
    is_method: bool
    is_static: bool
    is_class_method: bool
    is_property: bool
    docstring: Optional[str]
    complexity: int
    
    def has_side_effects(self) -> bool:
        """Check if function likely has side effects."""
        keywords = ['save', 'write', 'delete', 'update', 'send', 'post', 'put', 'patch']
        return any(kw in self.name.lower() for kw in keywords)

@dataclass
class ClassInfo:
    """Class information for test generation."""
    name: str
    methods: List[FunctionSignature]
    properties: List[str]
    init_params: List[Dict[str, Any]]
    base_classes: List[str]
    decorators: List[str]
    docstring: Optional[str]
    file_path: str
    dependencies: Set[str] = field(default_factory=set)

@dataclass
class TestCase:
    """Represents a single test case."""
    name: str
    type: TestType
    description: str
    test_code: str
    imports: Set[str] = field(default_factory=set)
    fixtures: Set[str] = field(default_factory=set)
    marks: Set[str] = field(default_factory=set)  # pytest marks

@dataclass
class TestSuite:
    """Collection of test cases for a module."""
    module_path: str
    test_file_path: str
    imports: Set[str]
    fixtures: List[str]
    test_cases: List[TestCase]
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None

class CodeInspector:
    """Advanced code analysis for test generation."""
    
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.classes: Dict[str, ClassInfo] = {}
        self.functions: Dict[str, FunctionSignature] = {}
        self.imports: Dict[str, Set[str]] = {}
        self.dependencies: Dict[str, Set[str]] = {}
    
    def analyze_module(self, module_path: Path) -> Tuple[Dict[str, ClassInfo], Dict[str, FunctionSignature]]:
        """Analyze a Python module for testing."""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(module_path))
            module_classes = {}
            module_functions = {}
            
            # Track imports
            module_imports = self._extract_imports(tree)
            self.imports[str(module_path)] = module_imports
            
            # Analyze top-level elements
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node, source, str(module_path))
                    module_classes[class_info.name] = class_info
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_sig = self._analyze_function(node, source)
                    module_functions[func_sig.name] = func_sig
            
            return module_classes, module_functions
            
        except Exception as e:
            logger.error(f"Error analyzing {module_path}: {e}")
            return {}, {}
    
    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract all imports from a module."""
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    if module:
                        imports.add(f"{module}.{alias.name}")
                    else:
                        imports.add(alias.name)
        return imports
    
    def _analyze_class(self, node: ast.ClassDef, source: str, file_path: str) -> ClassInfo:
        """Analyze a class definition."""
        methods = []
        properties = []
        init_params = []
        
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_sig = self._analyze_function(item, source, is_method=True)
                
                if func_sig.is_property:
                    properties.append(func_sig.name)
                else:
                    methods.append(func_sig)
                
                if func_sig.name == '__init__':
                    init_params = func_sig.params[1:]  # Skip 'self'
        
        # Get base classes
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(ast.unparse(base))
        
        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        return ClassInfo(
            name=node.name,
            methods=methods,
            properties=properties,
            init_params=init_params,
            base_classes=base_classes,
            decorators=decorators,
            docstring=ast.get_docstring(node),
            file_path=file_path
        )
    
    def _analyze_function(self, node, source: str, is_method: bool = False) -> FunctionSignature:
        """Analyze a function/method definition."""
        # Extract parameters with type hints
        params = []
        for i, arg in enumerate(node.args.args):
            param_info = {'name': arg.arg}
            
            # Get type annotation
            if arg.annotation:
                param_info['type'] = ast.unparse(arg.annotation)
            else:
                param_info['type'] = 'Any'
            
            # Get default value
            defaults_start = len(node.args.args) - len(node.args.defaults)
            if i >= defaults_start:
                default_idx = i - defaults_start
                param_info['default'] = ast.unparse(node.args.defaults[default_idx])
            else:
                param_info['default'] = None
            
            params.append(param_info)
        
        # Get return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)
        
        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Determine function characteristics
        is_async = isinstance(node, ast.AsyncFunctionDef)
        is_static = 'staticmethod' in decorators
        is_class_method = 'classmethod' in decorators
        is_property = 'property' in decorators
        
        # Calculate complexity
        complexity = self._calculate_complexity(node)
        
        return FunctionSignature(
            name=node.name,
            params=params,
            return_type=return_type,
            decorators=decorators,
            is_async=is_async,
            is_method=is_method,
            is_static=is_static,
            is_class_method=is_class_method,
            is_property=is_property,
            docstring=ast.get_docstring(node),
            complexity=complexity
        )
    
    def _get_decorator_name(self, decorator) -> str:
        """Extract decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return "unknown"
    
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

class TestStrategy:
    """Determines testing strategy based on code analysis."""
    
    @staticmethod
    def determine_test_types(func: FunctionSignature, class_info: Optional[ClassInfo] = None) -> List[TestType]:
        """Determine which types of tests to generate."""
        test_types = [TestType.UNIT]
        
        # Add parametrized tests for functions with multiple parameters
        if len(func.params) > 2:
            test_types.append(TestType.PARAMETRIZED)
        
        # Add edge case tests for complex functions
        if func.complexity > 3:
            test_types.append(TestType.EDGE_CASE)
        
        # Add exception tests if function might raise exceptions
        if func.docstring and 'raise' in func.docstring.lower():
            test_types.append(TestType.EXCEPTION)
        
        # Add mock tests for functions with side effects
        if func.has_side_effects():
            test_types.append(TestType.MOCK)
        
        # Add integration tests for methods that likely interact with other components
        if class_info and func.name not in ['__init__', '__str__', '__repr__']:
            if any(kw in func.name.lower() for kw in ['process', 'handle', 'execute', 'run']):
                test_types.append(TestType.INTEGRATION)
        
        return test_types
    
    @staticmethod
    def generate_test_data(param: Dict[str, Any]) -> List[Any]:
        """Generate test data based on parameter type."""
        param_type = param.get('type', 'Any')
        param_name = param['name']
        
        # Common test values by type
        test_values = {
            'int': [0, 1, -1, 100, -100, 2**31-1],
            'float': [0.0, 1.0, -1.0, 3.14, float('inf'), float('-inf')],
            'str': ['', 'a', 'test', 'Test String', '123', 'special!@#$', ' spaces ', '\n\t'],
            'bool': [True, False],
            'list': [[], [1], [1, 2, 3], ['a', 'b'], [None]],
            'List': [[], [1], [1, 2, 3], ['a', 'b'], [None]],
            'dict': [{}, {'key': 'value'}, {'a': 1, 'b': 2}],
            'Dict': [{}, {'key': 'value'}, {'a': 1, 'b': 2}],
            'Optional': [None, 'value', 0, ''],
            'Any': [None, 0, '', [], {}, True, 'test'],
            'None': [None],
            'bytes': [b'', b'test', b'\x00\x01\x02'],
            'tuple': [(), (1,), (1, 2), ('a', 'b')],
            'set': [set(), {1}, {1, 2, 3}, {'a', 'b'}],
        }
        
        # Extract base type from complex types like Optional[str], List[int], etc.
        base_type = param_type.split('[')[0].strip()
        
        # Return appropriate test values
        if base_type in test_values:
            return test_values[base_type][:3]  # Return first 3 values
        
        # For custom types, return generic test values
        return [None, 'test_value', 123]

class AITestGenerator:
    """Generates tests using AI."""
    
    def __init__(self, provider: str = "openai", framework: TestFramework = TestFramework.PYTEST):
        self.provider = provider
        self.framework = framework
        self._setup_client()
        self.total_cost = 0.0
    
    def _setup_client(self):
        """Setup AI client based on provider."""
        api_key = os.getenv(f"{self.provider.upper()}_API_KEY")
        if not api_key:
            raise ValueError(f"Please set {self.provider.upper()}_API_KEY environment variable")
        
        if self.provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = "gpt-4o-mini"
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = "claude-3-sonnet-20240229"
        elif self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel('gemini-pro')
            self.model = "gemini-pro"
    
    def generate_test_suite(self, 
                           module_path: str,
                           classes: Dict[str, ClassInfo],
                           functions: Dict[str, FunctionSignature]) -> TestSuite:
        """Generate complete test suite for a module."""
        test_cases = []
        imports = set()
        fixtures = []
        
        # Generate base imports
        if self.framework == TestFramework.PYTEST:
            imports.add("import pytest")
            imports.add("from unittest.mock import Mock, patch, MagicMock")
        else:
            imports.add("import unittest")
            imports.add("from unittest.mock import Mock, patch, MagicMock")
        
        # Add module import
        module_name = Path(module_path).stem
        imports.add(f"from {module_path.replace('/', '.').replace('.py', '')} import *")
        
        # Generate tests for standalone functions
        for func_name, func_sig in functions.items():
            test_types = TestStrategy.determine_test_types(func_sig)
            for test_type in test_types:
                test_case = self._generate_function_test(func_sig, test_type)
                if test_case:
                    test_cases.append(test_case)
                    imports.update(test_case.imports)
        
        # Generate tests for classes
        for class_name, class_info in classes.items():
            class_tests, class_fixtures = self._generate_class_tests(class_info)
            test_cases.extend(class_tests)
            fixtures.extend(class_fixtures)
            
            for test in class_tests:
                imports.update(test.imports)
        
        # Generate test file path
        test_file_path = self._get_test_file_path(module_path)
        
        return TestSuite(
            module_path=module_path,
            test_file_path=test_file_path,
            imports=imports,
            fixtures=fixtures,
            test_cases=test_cases
        )
    
    def _generate_function_test(self, func: FunctionSignature, test_type: TestType) -> Optional[TestCase]:
        """Generate test case for a function."""
        prompt = self._create_function_test_prompt(func, test_type)
        
        try:
            response = self._call_ai(prompt)
            return self._parse_test_response(response, func.name, test_type)
        except Exception as e:
            logger.error(f"Error generating test for {func.name}: {e}")
            return None
    
    def _generate_class_tests(self, class_info: ClassInfo) -> Tuple[List[TestCase], List[str]]:
        """Generate tests for a class."""
        test_cases = []
        fixtures = []
        
        # Generate fixture for class instantiation
        if self.framework == TestFramework.PYTEST:
            fixture_code = self._generate_class_fixture(class_info)
            fixtures.append(fixture_code)
        
        # Generate tests for each method
        for method in class_info.methods:
            if method.name.startswith('_') and method.name != '__init__':
                continue  # Skip private methods except __init__
            
            test_types = TestStrategy.determine_test_types(method, class_info)
            for test_type in test_types:
                test_case = self._generate_method_test(class_info, method, test_type)
                if test_case:
                    test_cases.append(test_case)
        
        return test_cases, fixtures
    
    def _create_function_test_prompt(self, func: FunctionSignature, test_type: TestType) -> str:
        """Create prompt for function test generation."""
        framework_specific = self._get_framework_instructions()
        
        # Generate test data examples
        test_data = []
        for param in func.params:
            test_values = TestStrategy.generate_test_data(param)
            test_data.append(f"{param['name']}: {test_values[:3]}")
        
        return f"""Generate a {test_type.value} test for this Python function using {self.framework.value}.

Function Signature:
```python
def {func.name}({', '.join(f"{p['name']}: {p['type']}" + (f" = {p['default']}" if p['default'] else "") for p in func.params)}) -> {func.return_type or 'None'}:
    '''
    {func.docstring or 'No docstring available'}
    '''
```

Function Characteristics:
- Complexity: {func.complexity}
- Is Async: {func.is_async}
- Decorators: {func.decorators}
- Has Side Effects: {func.has_side_effects()}

Test Type: {test_type.value}
{self._get_test_type_instructions(test_type)}

Suggested Test Data:
{chr(10).join(test_data)}

{framework_specific}

Requirements:
1. Write a complete, runnable test function
2. Include proper assertions
3. Handle edge cases for {test_type.value} test
4. Use descriptive test name following the pattern: test_{func.name}_{test_type.value}
5. Add appropriate mocks if the function has external dependencies
6. Include docstring explaining what the test validates

Return ONLY the test function code without imports or explanations.
"""
    
    def _create_method_test_prompt(self, class_info: ClassInfo, method: FunctionSignature, test_type: TestType) -> str:
        """Create prompt for method test generation."""
        return f"""Generate a {test_type.value} test for this class method using {self.framework.value}.

Class: {class_info.name}
Base Classes: {', '.join(class_info.base_classes) or 'None'}

Method Signature:
```python
def {method.name}({', '.join(f"{p['name']}: {p['type']}" + (f" = {p['default']}" if p['default'] else "") for p in method.params)}) -> {method.return_type or 'None'}:
    '''
    {method.docstring or 'No docstring available'}
    '''
```

Class __init__ parameters:
{', '.join(f"{p['name']}: {p['type']}" for p in class_info.init_params)}

Method Characteristics:
- Is Static: {method.is_static}
- Is Class Method: {method.is_class_method}
- Is Property: {method.is_property}
- Complexity: {method.complexity}

Test Type: {test_type.value}
{self._get_test_type_instructions(test_type)}

{self._get_framework_instructions()}

Requirements:
1. Create instance of {class_info.name} properly (mock dependencies if needed)
2. Test the {method.name} method thoroughly
3. Use descriptive test name: test_{class_info.name}_{method.name}_{test_type.value}
4. Include proper setup and teardown if needed
5. Mock external dependencies appropriately

Return ONLY the test function code.
"""
    
    def _get_framework_instructions(self) -> str:
        """Get framework-specific instructions."""
        if self.framework == TestFramework.PYTEST:
            return """
Framework: pytest
- Use pytest fixtures for setup
- Use pytest.raises for exception testing
- Use @pytest.mark decorators as appropriate
- Use parametrize for multiple test cases
- Assertions: assert statements
"""
        else:
            return """
Framework: unittest
- Inherit from unittest.TestCase
- Use setUp and tearDown methods
- Use self.assertX methods for assertions
- Use self.assertRaises for exception testing
- Use subTest for parametrized tests
"""
    
    def _get_test_type_instructions(self, test_type: TestType) -> str:
        """Get instructions for specific test type."""
        instructions = {
            TestType.UNIT: "Focus on testing the function in isolation with normal inputs.",
            TestType.INTEGRATION: "Test how the function integrates with other components.",
            TestType.PARAMETRIZED: "Create parametrized tests with multiple input combinations.",
            TestType.EDGE_CASE: "Test boundary conditions, empty inputs, None values, extreme values.",
            TestType.EXCEPTION: "Test error handling and exception raising scenarios.",
            TestType.MOCK: "Mock external dependencies and test side effects.",
        }
        return instructions.get(test_type, "")
    
    def _generate_class_fixture(self, class_info: ClassInfo) -> str:
        """Generate pytest fixture for class instantiation."""
        fixture_name = f"{class_info.name.lower()}_instance"
        
        # Generate mock parameters for __init__
        init_mocks = []
        for param in class_info.init_params:
            if 'Optional' in param.get('type', ''):
                init_mocks.append(f"{param['name']}=None")
            else:
                init_mocks.append(f"{param['name']}=Mock()")
        
        fixture_code = f"""
@pytest.fixture
def {fixture_name}():
    \"\"\"Fixture for creating {class_info.name} instance.\"\"\"
    instance = {class_info.name}({', '.join(init_mocks)})
    return instance
"""
        return fixture_code
    
    def _call_ai(self, prompt: str) -> str:
        """Call AI API to generate test code."""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500
                )
                return response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.3
                )
                return response.content[0].text
                
            elif self.provider == "google":
                response = self.client.generate_content(prompt)
                return response.text
                
        except Exception as e:
            logger.error(f"AI API call failed: {e}")
            raise
    
    def _parse_test_response(self, response: str, func_name: str, test_type: TestType) -> TestCase:
        """Parse AI response into TestCase object."""
        # Extract test function name
        test_name_match = re.search(r'def (test_\w+)', response)
        if test_name_match:
            test_name = test_name_match.group(1)
        else:
            test_name = f"test_{func_name}_{test_type.value}"
        
        # Extract imports from response if any
        imports = set()
        import_matches = re.findall(r'^import .*$|^from .* import .*$', response, re.MULTILINE)
        imports.update(import_matches)
        
        # Extract pytest marks
        marks = set()
        mark_matches = re.findall(r'@pytest\.mark\.(\w+)', response)
        marks.update(mark_matches)
        
        # Clean the test code
        test_code = response
        # Remove import statements as they'll be at the top of the file
        test_code = re.sub(r'^import .*$|^from .* import .*$', '', test_code, flags=re.MULTILINE)
        test_code = test_code.strip()
        
        return TestCase(
            name=test_name,
            type=test_type,
            description=f"{test_type.value} test for {func_name}",
            test_code=test_code,
            imports=imports,
            marks=marks
        )
    
    def _get_test_file_path(self, module_path: str) -> str:
        """Generate test file path from module path."""
        module_path = Path(module_path)
        test_dir = module_path.parent.parent / "tests"
        
        # Mirror the source structure in tests
        relative_path = module_path.relative_to(module_path.parent.parent / "src")
        test_file_name = f"test_{relative_path.stem}.py"
        test_file_path = test_dir / relative_path.parent / test_file_name
        
        return str(test_file_path)

class TestFileWriter:
    """Writes test files with proper structure."""
    
    def __init__(self, framework: TestFramework = TestFramework.PYTEST):
        self.framework = framework
    
    def write_test_suite(self, suite: TestSuite, output_dir: Path):
        """Write test suite to file."""
        test_file_path = output_dir / suite.test_file_path
        test_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = self._generate_file_content(suite)
        
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"✅ Written test file: {test_file_path}")
        return test_file_path
    
    def _generate_file_content(self, suite: TestSuite) -> str:
        """Generate complete test file content."""
        sections = []
        
        # File header
        sections.append(self._generate_header(suite))
        
        # Imports
        sections.append(self._generate_imports(suite))
        
        if self.framework == TestFramework.PYTEST:
            # Fixtures
            if suite.fixtures:
                sections.append("# Fixtures")
                sections.extend(suite.fixtures)
            
            # Test functions
            sections.append("\n# Test Cases")
            for test_case in suite.test_cases:
                if test_case.marks:
                    marks = '\n'.join(f"@pytest.mark.{mark}" for mark in test_case.marks)
                    sections.append(marks)
                sections.append(test_case.test_code)
                sections.append("")  # Empty line between tests
        
        else:  # unittest
            # Test class
            sections.append(self._generate_unittest_class(suite))
        
        return '\n'.join(sections)
    
    def _generate_header(self, suite: TestSuite) -> str:
        """Generate file header with docstring."""
        module_name = Path(suite.module_path).stem
        return f'''"""
Test suite for {module_name} module.
Generated automatically by AI Test Generator.
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Module: {suite.module_path}
Test Framework: {self.framework.value}
"""
'''
    
    def _generate_imports(self, suite: TestSuite) -> str:
        """Generate import statements."""
        # Group imports by type
        standard_imports = []
        third_party_imports = []
        local_imports = []
        
        for imp in sorted(suite.imports):
            if imp.startswith('import ') or imp.startswith('from '):
                # Determine import type
                module = imp.split()[1] if imp.startswith('from') else imp.split()[1]
                
                if module in ['os', 'sys', 'json', 'datetime', 'pathlib', 'typing']:
                    standard_imports.append(imp)
                elif module in ['pytest', 'unittest', 'mock', 'numpy', 'pandas']:
                    third_party_imports.append(imp)
                else:
                    local_imports.append(imp)
        
        sections = []
        if standard_imports:
            sections.append('\n'.join(standard_imports))
        if third_party_imports:
            sections.append('\n'.join(third_party_imports))
        if local_imports:
            sections.append('\n'.join(local_imports))
        
        return '\n\n'.join(sections)
    
    def _generate_unittest_class(self, suite: TestSuite) -> str:
        """Generate unittest TestCase class."""
        class_name = f"Test{Path(suite.module_path).stem.title()}"
        
        methods = []
        for test_case in suite.test_cases:
            # Convert pytest-style test to unittest method
            method_code = test_case.test_code.replace('def test_', 'def test_')
            method_code = self._convert_assertions_to_unittest(method_code)
            methods.append(method_code)
        
        return f"""
class {class_name}(unittest.TestCase):
    \"\"\"Test cases for {Path(suite.module_path).stem} module.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test fixtures.\"\"\"
        pass
    
    def tearDown(self):
        \"\"\"Tear down test fixtures.\"\"\"
        pass
    
    {'    '.join(methods)}

if __name__ == '__main__':
    unittest.main()
"""
    
    def _convert_assertions_to_unittest(self, code: str) -> str:
        """Convert pytest assertions to unittest assertions."""
        conversions = {
            r'assert (.+) == (.+)': r'self.assertEqual(\1, \2)',
            r'assert (.+) != (.+)': r'self.assertNotEqual(\1, \2)',
            r'assert (.+) is (.+)': r'self.assertIs(\1, \2)',
            r'assert (.+) is not (.+)': r'self.assertIsNot(\1, \2)',
            r'assert (.+) in (.+)': r'self.assertIn(\1, \2)',
            r'assert (.+) not in (.+)': r'self.assertNotIn(\1, \2)',
            r'assert (.+)': r'self.assertTrue(\1)',
            r'assert not (.+)': r'self.assertFalse(\1)',
        }
        
        for pattern, replacement in conversions.items():
            code = re.sub(pattern, replacement, code)
        
        return code

class TestCoverageAnalyzer:
    """Analyzes test coverage and generates reports."""
    
    def __init__(self, source_dir: Path, test_dir: Path):
        self.source_dir = source_dir
        self.test_dir = test_dir
        self.coverage_data = {}
    
    def analyze_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage for all modules."""
        coverage_report = {
            'timestamp': datetime.now().isoformat(),
            'source_dir': str(self.source_dir),
            'test_dir': str(self.test_dir),
            'modules': {},
            'summary': {}
        }
        
        # Find all source modules
        source_modules = list(self.source_dir.rglob("*.py"))
        
        # Find corresponding test files
        for module_path in source_modules:
            if "__pycache__" in str(module_path):
                continue
            
            module_name = module_path.stem
            test_file_name = f"test_{module_name}.py"
            
            # Look for test file
            test_files = list(self.test_dir.rglob(test_file_name))
            
            module_coverage = {
                'has_tests': len(test_files) > 0,
                'test_files': [str(f) for f in test_files],
                'missing_test_types': []
            }
            
            if test_files:
                # Analyze test file to determine coverage
                test_content = test_files[0].read_text()
                module_coverage['test_count'] = len(re.findall(r'def test_', test_content))
                module_coverage['has_unit_tests'] = 'test_' in test_content
                module_coverage['has_integration_tests'] = 'integration' in test_content.lower()
                module_coverage['has_mocks'] = 'mock' in test_content.lower()
                module_coverage['has_fixtures'] = '@pytest.fixture' in test_content or 'setUp' in test_content
            else:
                module_coverage['test_count'] = 0
                module_coverage['missing_test_types'] = ['unit', 'integration', 'edge_case']
            
            coverage_report['modules'][str(module_path)] = module_coverage
        
        # Generate summary
        total_modules = len(coverage_report['modules'])
        tested_modules = sum(1 for m in coverage_report['modules'].values() if m['has_tests'])
        total_tests = sum(m.get('test_count', 0) for m in coverage_report['modules'].values())
        
        coverage_report['summary'] = {
            'total_modules': total_modules,
            'tested_modules': tested_modules,
            'untested_modules': total_modules - tested_modules,
            'coverage_percentage': (tested_modules / total_modules * 100) if total_modules > 0 else 0,
            'total_tests': total_tests,
            'average_tests_per_module': total_tests / tested_modules if tested_modules > 0 else 0
        }
        
        return coverage_report
    
    def generate_coverage_badge(self, coverage_percentage: float) -> str:
        """Generate coverage badge SVG."""
        color = 'red' if coverage_percentage < 50 else 'yellow' if coverage_percentage < 80 else 'green'
        
        return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="104" height="20">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <mask id="a">
        <rect width="104" height="20" rx="3" fill="#fff"/>
    </mask>
    <g mask="url(#a)">
        <path fill="#555" d="M0 0h63v20H0z"/>
        <path fill="{color}" d="M63 0h41v20H63z"/>
        <path fill="url(#b)" d="M0 0h104v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="31.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
        <text x="31.5" y="14">coverage</text>
        <text x="83.5" y="15" fill="#010101" fill-opacity=".3">{coverage_percentage:.0f}%</text>
        <text x="83.5" y="14">{coverage_percentage:.0f}%</text>
    </g>
</svg>
"""

def main():
    """Main function for AI test generation."""
    parser = argparse.ArgumentParser(description="AI-Powered Test Generator")
    parser.add_argument('--source', default='src', help='Source directory to analyze')
    parser.add_argument('--tests', default='tests', help='Test directory')
    parser.add_argument('--framework', choices=['pytest', 'unittest'], default='pytest',
                       help='Test framework to use')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'google'], 
                       default='openai', help='AI provider')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--dry-run', action='store_true', help='Analyze without generating')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--parallel', action='store_true', help='Generate tests in parallel')
    parser.add_argument('--module', help='Generate tests for specific module only')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    source_dir = Path(args.source)
    test_dir = Path(args.tests)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Analyze source code
    logger.info(f"🔍 Analyzing source code in {source_dir}...")
    inspector = CodeInspector(str(source_dir))
    
    # Find modules to test
    if args.module:
        modules = [source_dir / args.module]
    else:
        modules = list(source_dir.rglob("*.py"))
        modules = [m for m in modules if "__pycache__" not in str(m)]
    
    logger.info(f"📊 Found {len(modules)} Python modules")
    
    if args.dry_run:
        # Just show what would be generated
        for module in modules:
            classes, functions = inspector.analyze_module(module)
            print(f"\nModule: {module}")
            print(f"  Classes: {list(classes.keys())}")
            print(f"  Functions: {list(functions.keys())}")
            
            # Show test strategy
            for func_name, func in functions.items():
                test_types = TestStrategy.determine_test_types(func)
                print(f"    {func_name}: {[t.value for t in test_types]}")
        return
    
    # Step 2: Generate tests
    logger.info(f"🤖 Generating tests using {args.provider}...")
    framework = TestFramework(args.framework)
    test_gen = AITestGenerator(args.provider, framework)
    writer = TestFileWriter(framework)
    
    generated_files = []
    failed_modules = []
    
    for module in modules:
        try:
            logger.info(f"📝 Generating tests for {module.stem}...")
            
            # Analyze module
            classes, functions = inspector.analyze_module(module)
            
            if not classes and not functions:
                logger.warning(f"⚠️  No testable code found in {module}")
                continue
            
            # Generate test suite
            suite = test_gen.generate_test_suite(
                str(module.relative_to(source_dir.parent)),
                classes,
                functions
            )
            
            # Write test file
            test_file = writer.write_test_suite(suite, source_dir.parent)
            generated_files.append(test_file)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate tests for {module}: {e}")
            failed_modules.append(str(module))
    
    # Step 3: Generate coverage report
    if args.coverage:
        logger.info("📊 Generating coverage report...")
        analyzer = TestCoverageAnalyzer(source_dir, test_dir)
        coverage_report = analyzer.analyze_coverage()
        
        # Save report
        report_path = test_dir / "coverage_report.json"
        with open(report_path, 'w') as f:
            json.dump(coverage_report, f, indent=2)
        
        # Generate badge
        badge_svg = analyzer.generate_coverage_badge(coverage_report['summary']['coverage_percentage'])
        badge_path = test_dir / "coverage.svg"
        with open(badge_path, 'w') as f:
            f.write(badge_svg)
        
        logger.info(f"📊 Coverage: {coverage_report['summary']['coverage_percentage']:.1f}%")
        logger.info(f"📊 Report saved to {report_path}")
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("✅ Test Generation Complete!")
    logger.info(f"📁 Generated {len(generated_files)} test files")
    if failed_modules:
        logger.warning(f"⚠️  Failed modules: {failed_modules}")
    logger.info(f"💰 Estimated cost: ${test_gen.total_cost:.2f}")
    logger.info("\n🚀 Run tests with: pytest tests/ -v --cov=src")

if __name__ == "__main__":
    main()
