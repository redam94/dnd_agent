"""
Markdown Documentation Builder
Converts AI-generated documentation into structured markdown files.
Supports GitHub-flavored markdown, MkDocs, and other markdown-based documentation systems.
"""

import ast
import json
import logging
import os
import re
import shutil
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MarkdownStyle(Enum):
    """Markdown documentation styles."""

    GITHUB = "github"
    MKDOCS = "mkdocs"
    SPHINX_MD = "sphinx_md"
    DOCUSAURUS = "docusaurus"
    VUEPRESS = "vuepress"


@dataclass
class MarkdownSection:
    """Represents a section in markdown documentation."""

    title: str
    content: str
    level: int = 2
    anchor: Optional[str] = None
    subsections: List["MarkdownSection"] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Convert section to markdown string."""
        # Generate anchor if not provided
        if not self.anchor:
            self.anchor = self.title.lower().replace(" ", "-").replace("/", "-")

        # Create heading
        heading = f"{'#' * self.level} {self.title}"

        # Build content
        lines = [heading, ""]
        if self.content:
            lines.append(self.content)
            lines.append("")

        # Add subsections
        for subsection in self.subsections:
            lines.append(subsection.to_markdown())
            lines.append("")

        return "\n".join(lines)


@dataclass
class MarkdownDocument:
    """Represents a complete markdown document."""

    title: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sections: List[MarkdownSection] = field(default_factory=list)
    toc: bool = True
    style: MarkdownStyle = MarkdownStyle.GITHUB

    def to_markdown(self) -> str:
        """Convert document to markdown string."""
        lines = []

        # Add frontmatter if needed
        if self.style in [MarkdownStyle.MKDOCS, MarkdownStyle.DOCUSAURUS, MarkdownStyle.VUEPRESS]:
            lines.extend(self._generate_frontmatter())
            lines.append("")

        # Add title
        lines.append(f"# {self.title}")
        lines.append("")

        # Add description
        if self.description:
            lines.append(f"> {self.description}")
            lines.append("")

        # Add table of contents
        if self.toc and len(self.sections) > 2:
            lines.append(self._generate_toc())
            lines.append("")

        # Add sections
        for section in self.sections:
            lines.append(section.to_markdown())

        # Add footer
        lines.extend(self._generate_footer())

        return "\n".join(lines)

    def _generate_frontmatter(self) -> List[str]:
        """Generate frontmatter for static site generators."""
        frontmatter = ["---"]
        frontmatter.append(f"title: {self.title}")

        if self.description:
            frontmatter.append(f"description: {self.description}")

        for key, value in self.metadata.items():
            if isinstance(value, list):
                frontmatter.append(f"{key}:")
                for item in value:
                    frontmatter.append(f"  - {item}")
            else:
                frontmatter.append(f"{key}: {value}")

        frontmatter.append("---")
        return frontmatter

    def _generate_toc(self) -> str:
        """Generate table of contents."""
        toc_lines = ["## Table of Contents", ""]

        for section in self.sections:
            # Main section link
            toc_lines.append(
                f"- [{section.title}](#{section.anchor or section.title.lower().replace(' ', '-')})"
            )

            # Subsection links
            for subsection in section.subsections:
                anchor = subsection.anchor or subsection.title.lower().replace(" ", "-")
                toc_lines.append(f"  - [{subsection.title}](#{anchor})")

        return "\n".join(toc_lines)

    def _generate_footer(self) -> List[str]:
        """Generate document footer."""
        footer = []
        footer.append("")
        footer.append("---")
        footer.append("")
        footer.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        return footer


class CodeDocumenter:
    """Extracts and formats code documentation."""

    def __init__(self, style: MarkdownStyle = MarkdownStyle.GITHUB):
        self.style = style

    def document_function(self, func_info: Dict[str, Any]) -> MarkdownSection:
        """Document a function or method."""
        name = func_info.get("name", "Unknown")
        signature = func_info.get("signature", "")
        docstring = func_info.get("docstring", "")
        source = func_info.get("source", "")
        return_type = func_info.get("return_type", "Any")
        params = func_info.get("params", [])
        decorators = func_info.get("decorators", [])

        # Create section
        section = MarkdownSection(title=f"`{name}()`", content="", level=3)

        # Build content
        content_parts = []

        # Add decorators
        if decorators:
            for decorator in decorators:
                content_parts.append(f"@{decorator}")

        # Add signature with syntax highlighting
        if signature:
            content_parts.append("```python")
            content_parts.append(signature)
            content_parts.append("```")
            content_parts.append("")

        # Add description
        if docstring:
            # Parse docstring for better formatting
            parsed = self._parse_docstring(docstring)

            if parsed.get("summary"):
                content_parts.append(parsed["summary"])
                content_parts.append("")

            if parsed.get("description"):
                content_parts.append(parsed["description"])
                content_parts.append("")

        # Add parameters table
        if params:
            content_parts.append("**Parameters:**")
            content_parts.append("")
            content_parts.append("| Parameter | Type | Default | Description |")
            content_parts.append("|-----------|------|---------|-------------|")

            for param in params:
                param_name = param.get("name", "")
                param_type = param.get("type", "Any")
                param_default = param.get("default", "-")
                param_desc = param.get("description", "")

                # Escape pipe characters
                param_desc = param_desc.replace("|", "\\|")

                content_parts.append(
                    f"| `{param_name}` | `{param_type}` | {param_default} | {param_desc} |"
                )

            content_parts.append("")

        # Add return information
        if return_type and return_type != "None":
            content_parts.append("**Returns:**")
            content_parts.append("")
            content_parts.append(f"- **Type:** `{return_type}`")
            if "returns" in parsed:
                content_parts.append(f"- **Description:** {parsed['returns']}")
            content_parts.append("")

        # Add raises information
        if "raises" in parsed and parsed["raises"]:
            content_parts.append("**Raises:**")
            content_parts.append("")
            for exception, desc in parsed["raises"].items():
                content_parts.append(f"- `{exception}`: {desc}")
            content_parts.append("")

        # Add examples
        if "examples" in parsed and parsed["examples"]:
            content_parts.append("**Examples:**")
            content_parts.append("")
            content_parts.append("```python")
            content_parts.append(parsed["examples"])
            content_parts.append("```")
            content_parts.append("")

        # Add source code (collapsible for GitHub)
        if source and self.style == MarkdownStyle.GITHUB:
            content_parts.append("<details>")
            content_parts.append("<summary>View Source</summary>")
            content_parts.append("")
            content_parts.append("```python")
            content_parts.append(source[:1000])  # Limit source length
            if len(source) > 1000:
                content_parts.append("# ... (truncated)")
            content_parts.append("```")
            content_parts.append("")
            content_parts.append("</details>")

        section.content = "\n".join(content_parts)
        return section

    def document_class(self, class_info: Dict[str, Any]) -> MarkdownSection:
        """Document a class."""
        name = class_info.get("name", "Unknown")
        docstring = class_info.get("docstring", "")
        bases = class_info.get("base_classes", [])
        methods = class_info.get("methods", [])
        properties = class_info.get("properties", [])
        init_params = class_info.get("init_params", [])

        # Create main section
        section = MarkdownSection(title=f"Class: `{name}`", content="", level=2)

        # Build content
        content_parts = []

        # Add inheritance info
        if bases:
            content_parts.append(f"**Inherits from:** {', '.join([f'`{base}`' for base in bases])}")
            content_parts.append("")

        # Add description
        if docstring:
            parsed = self._parse_docstring(docstring)
            if parsed.get("summary"):
                content_parts.append(parsed["summary"])
                content_parts.append("")
            if parsed.get("description"):
                content_parts.append(parsed["description"])
                content_parts.append("")

        # Add constructor parameters
        if init_params:
            content_parts.append("### Constructor")
            content_parts.append("")
            content_parts.append("```python")
            params_str = ", ".join([p.get("name", "") for p in init_params])
            content_parts.append(f"{name}({params_str})")
            content_parts.append("```")
            content_parts.append("")

            content_parts.append("**Parameters:**")
            content_parts.append("")
            content_parts.append("| Parameter | Type | Default | Description |")
            content_parts.append("|-----------|------|---------|-------------|")

            for param in init_params:
                param_name = param.get("name", "")
                param_type = param.get("type", "Any")
                param_default = param.get("default", "-")
                param_desc = param.get("description", "")
                content_parts.append(
                    f"| `{param_name}` | `{param_type}` | {param_default} | {param_desc} |"
                )

            content_parts.append("")

        # Add properties
        if properties:
            content_parts.append("### Properties")
            content_parts.append("")
            for prop in properties:
                content_parts.append(f"- `{prop}`")
            content_parts.append("")

        section.content = "\n".join(content_parts)

        # Add method subsections
        if methods:
            methods_section = MarkdownSection(title="Methods", content="", level=3)

            for method in methods:
                if not method["name"].startswith("_") or method["name"] == "__init__":
                    method_section = self.document_function(method)
                    methods_section.subsections.append(method_section)

            section.subsections.append(methods_section)

        return section

    def document_module(self, module_info: Dict[str, Any]) -> MarkdownDocument:
        """Document a module."""
        name = module_info.get("name", "Unknown Module")
        path = module_info.get("path", "")
        docstring = module_info.get("docstring", "")
        classes = module_info.get("classes", [])
        functions = module_info.get("functions", [])
        imports = module_info.get("imports", [])

        # Create document
        doc = MarkdownDocument(
            title=f"Module: {name}",
            description=docstring or f"Documentation for {name} module",
            metadata={"module": name, "path": path, "generated": datetime.now().isoformat()},
        )

        # Add overview section
        if docstring:
            overview = MarkdownSection(title="Overview", content=docstring, level=2)
            doc.sections.append(overview)

        # Add imports section if relevant
        if imports:
            imports_section = MarkdownSection(title="Dependencies", content="", level=2)
            imports_content = ["This module imports the following dependencies:", ""]
            imports_content.append("```python")
            for imp in imports[:10]:  # Limit to first 10
                imports_content.append(imp)
            if len(imports) > 10:
                imports_content.append(f"# ... and {len(imports) - 10} more")
            imports_content.append("```")
            imports_section.content = "\n".join(imports_content)
            doc.sections.append(imports_section)

        # Add classes section
        if classes:
            for class_info in classes:
                class_section = self.document_class(class_info)
                doc.sections.append(class_section)

        # Add functions section
        if functions:
            functions_section = MarkdownSection(title="Functions", content="", level=2)

            for func_info in functions:
                func_section = self.document_function(func_info)
                functions_section.subsections.append(func_section)

            doc.sections.append(functions_section)

        return doc

    def _parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse docstring into structured format."""
        parsed = {
            "summary": "",
            "description": "",
            "params": {},
            "returns": "",
            "raises": {},
            "examples": "",
        }

        if not docstring:
            return parsed

        lines = docstring.strip().split("\n")

        # Extract summary (first line)
        if lines:
            parsed["summary"] = lines[0].strip()

        # Parse sections
        current_section = "description"
        section_content = []

        for line in lines[1:]:
            line = line.strip()

            # Check for section headers
            if line.lower() in ["parameters:", "params:", "arguments:", "args:"]:
                current_section = "params"
                section_content = []
            elif line.lower() in ["returns:", "return:"]:
                current_section = "returns"
                section_content = []
            elif line.lower() in ["raises:", "raise:", "except:", "exceptions:"]:
                current_section = "raises"
                section_content = []
            elif line.lower() in ["example:", "examples:", "usage:"]:
                current_section = "examples"
                section_content = []
            elif line.lower() in ["note:", "notes:"]:
                current_section = "notes"
                section_content = []
            else:
                section_content.append(line)

        # Store the last section
        if current_section == "description":
            parsed["description"] = "\n".join(section_content).strip()
        elif current_section == "returns":
            parsed["returns"] = "\n".join(section_content).strip()
        elif current_section == "examples":
            parsed["examples"] = "\n".join(section_content).strip()

        return parsed


class MarkdownAPIGenerator:
    """Generates API reference documentation in markdown."""

    def __init__(
        self, source_dir: Path, output_dir: Path, style: MarkdownStyle = MarkdownStyle.GITHUB
    ):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.style = style
        self.documenter = CodeDocumenter(style)

    def generate_api_reference(self, modules_info: Dict[str, Any]) -> Dict[str, Path]:
        """Generate complete API reference in markdown."""
        generated_files = {}

        # Create API directory
        api_dir = self.output_dir / "api"
        api_dir.mkdir(parents=True, exist_ok=True)

        # Generate index
        index_doc = self._generate_api_index(modules_info)
        index_path = api_dir / "README.md"
        self._write_document(index_doc, index_path)
        generated_files["index"] = index_path

        # Generate module documentation
        for module_name, module_info in modules_info.items():
            module_doc = self.documenter.document_module(module_info)

            # Create safe filename
            safe_name = module_name.replace("/", "_").replace(".", "_")
            module_path = api_dir / f"{safe_name}.md"

            self._write_document(module_doc, module_path)
            generated_files[module_name] = module_path

        logger.info(f"Generated {len(generated_files)} API documentation files")
        return generated_files

    def _generate_api_index(self, modules_info: Dict[str, Any]) -> MarkdownDocument:
        """Generate API reference index."""
        doc = MarkdownDocument(
            title="API Reference",
            description="Complete API documentation for all modules",
            style=self.style,
        )

        # Overview section
        overview = MarkdownSection(
            title="Overview",
            content=f"This API reference covers {len(modules_info)} modules.",
            level=2,
        )
        doc.sections.append(overview)

        # Modules list section
        modules_section = MarkdownSection(title="Modules", content="", level=2)

        # Group modules by package
        packages = {}
        for module_name in sorted(modules_info.keys()):
            parts = module_name.split("/")
            if len(parts) > 1:
                package = parts[0]
            else:
                package = "root"

            if package not in packages:
                packages[package] = []
            packages[package].append(module_name)

        # Create content for each package
        content_lines = []
        for package, modules in sorted(packages.items()):
            if package != "root":
                content_lines.append(f"### {package}")
                content_lines.append("")

            for module in modules:
                module_info = modules_info[module]
                safe_name = module.replace("/", "_").replace(".", "_")

                # Create link to module doc
                link = f"./{safe_name}.md"
                description = (
                    module_info.get("docstring", "").split("\n")[0]
                    if module_info.get("docstring")
                    else "No description"
                )

                content_lines.append(f"- [{module}]({link}) - {description}")

            content_lines.append("")

        modules_section.content = "\n".join(content_lines)
        doc.sections.append(modules_section)

        return doc

    def _write_document(self, doc: MarkdownDocument, path: Path):
        """Write markdown document to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc.to_markdown())


class MarkdownGuideGenerator:
    """Generates user guides and tutorials in markdown."""

    def __init__(
        self,
        output_dir: Path,
        project_info: Dict[str, Any],
        style: MarkdownStyle = MarkdownStyle.GITHUB,
    ):
        self.output_dir = output_dir
        self.project_info = project_info
        self.style = style

    def generate_guides(self) -> Dict[str, Path]:
        """Generate all guide documents."""
        guides_dir = self.output_dir / "guides"
        guides_dir.mkdir(parents=True, exist_ok=True)

        generated = {}

        # Generate installation guide
        install_guide = self._generate_installation_guide()
        install_path = guides_dir / "installation.md"
        self._write_document(install_guide, install_path)
        generated["installation"] = install_path

        # Generate quickstart guide
        quickstart = self._generate_quickstart_guide()
        quickstart_path = guides_dir / "quickstart.md"
        self._write_document(quickstart, quickstart_path)
        generated["quickstart"] = quickstart_path

        # Generate configuration guide
        config_guide = self._generate_configuration_guide()
        config_path = guides_dir / "configuration.md"
        self._write_document(config_guide, config_path)
        generated["configuration"] = config_path

        # Generate development guide
        dev_guide = self._generate_development_guide()
        dev_path = guides_dir / "development.md"
        self._write_document(dev_guide, dev_path)
        generated["development"] = dev_path

        # Generate contributing guide
        contrib_guide = self._generate_contributing_guide()
        contrib_path = guides_dir / "contributing.md"
        self._write_document(contrib_guide, contrib_path)
        generated["contributing"] = contrib_path

        return generated

    def _generate_installation_guide(self) -> MarkdownDocument:
        """Generate installation guide."""
        doc = MarkdownDocument(
            title="Installation Guide",
            description="Complete installation instructions",
            style=self.style,
        )

        # Requirements section
        requirements = MarkdownSection(
            title="Requirements",
            content=f"""
Before installing, ensure you have:

- Python {self.project_info.get('python_version', '3.9')} or higher
- pip package manager
- (Optional) virtualenv or conda for environment management
            """.strip(),
            level=2,
        )
        doc.sections.append(requirements)

        # Installation methods
        install_section = MarkdownSection(title="Installation Methods", content="", level=2)

        # PyPI installation
        pypi = MarkdownSection(
            title="Install from PyPI",
            content=f"""
```bash
pip install {self.project_info.get('package_name', 'package-name')}
```

For specific version:
```bash
pip install {self.project_info.get('package_name', 'package-name')}==1.0.0
```
            """.strip(),
            level=3,
        )
        install_section.subsections.append(pypi)

        # Source installation
        source = MarkdownSection(
            title="Install from Source",
            content=f"""
```bash
git clone {self.project_info.get('repository_url', 'https://github.com/username/repo')}
cd {self.project_info.get('package_name', 'package-name')}
pip install -e .
```

For development installation with all dependencies:
```bash
pip install -e ".[dev,docs,test]"
```
            """.strip(),
            level=3,
        )
        install_section.subsections.append(source)

        doc.sections.append(install_section)

        # Verification section
        verify = MarkdownSection(
            title="Verify Installation",
            content=f"""
```python
import {self.project_info.get('module_name', 'module_name')}
print({self.project_info.get('module_name', 'module_name')}.__version__)
```
            """.strip(),
            level=2,
        )
        doc.sections.append(verify)

        return doc

    def _generate_quickstart_guide(self) -> MarkdownDocument:
        """Generate quickstart guide."""
        doc = MarkdownDocument(
            title="Quick Start Guide",
            description="Get up and running in 5 minutes",
            style=self.style,
        )

        # Basic usage
        basic = MarkdownSection(
            title="Basic Usage",
            content=f"""
Here's a simple example to get you started:

```python
from {self.project_info.get('module_name', 'module_name')} import main_function

# Initialize
instance = main_function()

# Use the functionality
result = instance.process(data)
print(result)
```
            """.strip(),
            level=2,
        )
        doc.sections.append(basic)

        # Common use cases
        use_cases = MarkdownSection(title="Common Use Cases", content="", level=2)

        # Add example use cases
        for i, use_case in enumerate(["Data Processing", "API Integration", "File Management"], 1):
            case_section = MarkdownSection(
                title=use_case,
                content=f"""
```python
# Example for {use_case}
# Add your specific code here
```
                """.strip(),
                level=3,
            )
            use_cases.subsections.append(case_section)

        doc.sections.append(use_cases)

        return doc

    def _generate_configuration_guide(self) -> MarkdownDocument:
        """Generate configuration guide."""
        doc = MarkdownDocument(
            title="Configuration Guide",
            description="Configure the application for your needs",
            style=self.style,
        )

        # Configuration file
        config_file = MarkdownSection(
            title="Configuration File",
            content="""
Create a configuration file `config.yaml`:

```yaml
# Application settings
app:
  name: MyApp
  version: 1.0.0
  debug: false

# Database settings
database:
  host: localhost
  port: 5432
  name: mydb

# API settings
api:
  base_url: https://api.example.com
  timeout: 30
  retry_count: 3
```
            """.strip(),
            level=2,
        )
        doc.sections.append(config_file)

        # Environment variables
        env_vars = MarkdownSection(
            title="Environment Variables",
            content="""
You can also configure using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_DEBUG` | Enable debug mode | `false` |
| `DB_HOST` | Database host | `localhost` |
| `DB_PORT` | Database port | `5432` |
| `API_KEY` | API key for external services | - |
            """.strip(),
            level=2,
        )
        doc.sections.append(env_vars)

        return doc

    def _generate_development_guide(self) -> MarkdownDocument:
        """Generate development guide."""
        doc = MarkdownDocument(
            title="Development Guide",
            description="Guide for contributors and developers",
            style=self.style,
        )

        # Setup section
        setup = MarkdownSection(
            title="Development Setup",
            content="""
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-name>
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```
            """.strip(),
            level=2,
        )
        doc.sections.append(setup)

        # Testing section
        testing = MarkdownSection(
            title="Running Tests",
            content="""
Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_specific.py
```

Run with markers:
```bash
pytest -m "not slow"
```
            """.strip(),
            level=2,
        )
        doc.sections.append(testing)

        # Code quality
        quality = MarkdownSection(
            title="Code Quality",
            content="""
Format code:
```bash
black src/ tests/
isort src/ tests/
```

Run linting:
```bash
flake8 src/ tests/
mypy src/
```

Run all quality checks:
```bash
make quality
```
            """.strip(),
            level=2,
        )
        doc.sections.append(quality)

        return doc

    def _generate_contributing_guide(self) -> MarkdownDocument:
        """Generate contributing guide."""
        doc = MarkdownDocument(
            title="Contributing Guide",
            description="How to contribute to this project",
            style=self.style,
        )

        # Welcome section
        welcome = MarkdownSection(
            title="Welcome Contributors!",
            content="We welcome contributions from everyone. This guide will help you get started.",
            level=2,
        )
        doc.sections.append(welcome)

        # Process section
        process = MarkdownSection(
            title="Contribution Process",
            content="""
1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your feature/fix
4. **Make your changes** and commit them
5. **Push to your fork** on GitHub
6. **Open a Pull Request** from your fork to our main branch
            """.strip(),
            level=2,
        )
        doc.sections.append(process)

        # Guidelines
        guidelines = MarkdownSection(
            title="Guidelines",
            content="""
### Code Style
- Follow PEP 8
- Use type hints
- Write docstrings for all public functions
- Keep line length under 100 characters

### Commit Messages
- Use descriptive commit messages
- Start with a verb (Add, Fix, Update, etc.)
- Reference issue numbers when applicable

### Testing
- Write tests for new features
- Ensure all tests pass before submitting
- Maintain or improve code coverage
            """.strip(),
            level=2,
        )
        doc.sections.append(guidelines)

        return doc

    def _write_document(self, doc: MarkdownDocument, path: Path):
        """Write markdown document to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc.to_markdown())


class MarkdownDocumentationBuilder:
    """Main builder for complete markdown documentation."""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        project_info: Dict[str, Any] = None,
        style: MarkdownStyle = MarkdownStyle.GITHUB,
    ):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.project_info = project_info or {}
        self.style = style

        # Initialize generators
        self.api_generator = MarkdownAPIGenerator(source_dir, output_dir, style)
        self.guide_generator = MarkdownGuideGenerator(output_dir, project_info, style)

    def build_documentation(
        self, modules_info: Dict[str, Any], include_api: bool = True, include_guides: bool = True
    ) -> Dict[str, Any]:
        """Build complete markdown documentation."""
        logger.info("🏗️ Building markdown documentation...")

        results = {
            "output_dir": str(self.output_dir),
            "style": self.style.value,
            "generated_files": {},
        }

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate main README
        readme = self._generate_main_readme(modules_info)
        readme_path = self.output_dir / "README.md"
        self._write_document(readme, readme_path)
        results["generated_files"]["readme"] = str(readme_path)

        # Generate API documentation
        if include_api and modules_info:
            api_files = self.api_generator.generate_api_reference(modules_info)
            results["generated_files"]["api"] = {k: str(v) for k, v in api_files.items()}

        # Generate guides
        if include_guides:
            guide_files = self.guide_generator.generate_guides()
            results["generated_files"]["guides"] = {k: str(v) for k, v in guide_files.items()}

        # Generate navigation/sidebar for different platforms
        if self.style == MarkdownStyle.MKDOCS:
            self._generate_mkdocs_config()
        elif self.style == MarkdownStyle.DOCUSAURUS:
            self._generate_docusaurus_sidebar()
        elif self.style == MarkdownStyle.VUEPRESS:
            self._generate_vuepress_config()

        # Generate index file
        index = self._generate_index_file()
        index_path = self.output_dir / "index.md"
        self._write_document(index, index_path)
        results["generated_files"]["index"] = str(index_path)

        logger.info(f"✅ Generated {len(results['generated_files'])} documentation files")
        return results

    def _generate_main_readme(self, modules_info: Dict[str, Any]) -> MarkdownDocument:
        """Generate main README file."""
        doc = MarkdownDocument(
            title=self.project_info.get("project_name", "Project Documentation"),
            description=self.project_info.get("description", ""),
            style=self.style,
        )

        # Badges section
        badges = MarkdownSection(title="", content=self._generate_badges(), level=0)
        doc.sections.append(badges)

        # Overview
        overview = MarkdownSection(
            title="Overview",
            content=self.project_info.get("long_description", "Project overview goes here."),
            level=2,
        )
        doc.sections.append(overview)

        # Features
        features = MarkdownSection(
            title="Features",
            content="""
- ✨ Feature 1
- 🚀 Feature 2
- 🔧 Feature 3
- 📚 Comprehensive documentation
- 🧪 Extensive test coverage
            """.strip(),
            level=2,
        )
        doc.sections.append(features)

        # Installation
        install = MarkdownSection(
            title="Installation",
            content=f"""
```bash
pip install {self.project_info.get('package_name', 'package-name')}
```

For more detailed instructions, see the [Installation Guide](guides/installation.md).
            """.strip(),
            level=2,
        )
        doc.sections.append(install)

        # Quick Start
        quickstart = MarkdownSection(
            title="Quick Start",
            content=f"""
```python
from {self.project_info.get('module_name', 'module')} import main

# Your code here
result = main()
```

For more examples, see the [Quick Start Guide](guides/quickstart.md).
            """.strip(),
            level=2,
        )
        doc.sections.append(quickstart)

        # Documentation
        docs_section = MarkdownSection(
            title="Documentation",
            content="""
- [Installation Guide](guides/installation.md)
- [Quick Start Guide](guides/quickstart.md)
- [Configuration Guide](guides/configuration.md)
- [API Reference](api/README.md)
- [Development Guide](guides/development.md)
- [Contributing Guide](guides/contributing.md)
            """.strip(),
            level=2,
        )
        doc.sections.append(docs_section)

        # Project structure
        if modules_info:
            structure = MarkdownSection(
                title="Project Structure",
                content=self._generate_project_structure(modules_info),
                level=2,
            )
            doc.sections.append(structure)

        return doc

    def _generate_badges(self) -> str:
        """Generate status badges."""
        badges = []

        # Version badge
        if self.project_info.get("version"):
            badges.append(
                f"![Version](https://img.shields.io/badge/version-{self.project_info['version']}-blue)"
            )

        # Python version
        if self.project_info.get("python_version"):
            badges.append(
                f"![Python](https://img.shields.io/badge/python-{self.project_info['python_version']}%2B-blue)"
            )

        # License
        if self.project_info.get("license"):
            badges.append(
                f"![License](https://img.shields.io/badge/license-{self.project_info['license']}-green)"
            )

        # Tests
        badges.append("![Tests](https://img.shields.io/badge/tests-passing-brightgreen)")

        # Coverage
        badges.append("![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)")

        # Documentation
        badges.append("![Docs](https://img.shields.io/badge/docs-latest-brightgreen)")

        return " ".join(badges)

    def _generate_project_structure(self, modules_info: Dict[str, Any]) -> str:
        """Generate project structure tree."""
        lines = ["```"]
        lines.append(f"{self.project_info.get('package_name', 'project')}/")

        # Group modules by directory
        dirs = {}
        for module_path in sorted(modules_info.keys()):
            parts = module_path.split("/")
            if len(parts) > 1:
                dir_name = parts[0]
                if dir_name not in dirs:
                    dirs[dir_name] = []
                dirs[dir_name].append("/".join(parts[1:]))
            else:
                if "root" not in dirs:
                    dirs["root"] = []
                dirs["root"].append(module_path)

        # Generate tree
        for dir_name, files in sorted(dirs.items()):
            if dir_name != "root":
                lines.append(f"├── {dir_name}/")
                for file in files:
                    lines.append(f"│   ├── {file}")
            else:
                for file in files:
                    lines.append(f"├── {file}")

        lines.append("```")
        return "\n".join(lines)

    def _generate_index_file(self) -> MarkdownDocument:
        """Generate index/home file for documentation."""
        doc = MarkdownDocument(
            title="Documentation",
            description="Welcome to the documentation",
            style=self.style,
            toc=False,
        )

        # Welcome section
        welcome = MarkdownSection(
            title="",
            content=f"""
Welcome to the documentation for **{self.project_info.get('project_name', 'Project')}**!

{self.project_info.get('description', '')}
            """.strip(),
            level=0,
        )
        doc.sections.append(welcome)

        # Getting started
        getting_started = MarkdownSection(
            title="Getting Started",
            content="""
New to the project? Start here:

1. 📦 [Installation Guide](guides/installation.md) - Get the project installed
2. 🚀 [Quick Start Guide](guides/quickstart.md) - Learn the basics
3. ⚙️ [Configuration Guide](guides/configuration.md) - Configure for your needs
            """.strip(),
            level=2,
        )
        doc.sections.append(getting_started)

        # Reference
        reference = MarkdownSection(
            title="Reference",
            content="""
Detailed technical documentation:

- 📚 [API Reference](api/README.md) - Complete API documentation
- 🔧 [Configuration Reference](guides/configuration.md) - All configuration options
- 📖 [Examples](examples/README.md) - Code examples and tutorials
            """.strip(),
            level=2,
        )
        doc.sections.append(reference)

        # Contributing
        contributing = MarkdownSection(
            title="Contributing",
            content="""
Want to contribute?

- 💻 [Development Guide](guides/development.md) - Set up development environment
- 🤝 [Contributing Guide](guides/contributing.md) - Contribution guidelines
- 🐛 [Issue Tracker](<github-issues-url>) - Report bugs or request features
            """.strip(),
            level=2,
        )
        doc.sections.append(contributing)

        return doc

    def _generate_mkdocs_config(self):
        """Generate MkDocs configuration."""
        config = {
            "site_name": self.project_info.get("project_name", "Documentation"),
            "site_description": self.project_info.get("description", ""),
            "repo_url": self.project_info.get("repository_url", ""),
            "theme": {
                "name": "material",
                "palette": {"primary": "indigo", "accent": "indigo"},
                "features": [
                    "navigation.tabs",
                    "navigation.sections",
                    "toc.integrate",
                    "search.suggest",
                    "search.highlight",
                ],
            },
            "nav": [
                {"Home": "index.md"},
                {
                    "Guides": [
                        {"Installation": "guides/installation.md"},
                        {"Quick Start": "guides/quickstart.md"},
                        {"Configuration": "guides/configuration.md"},
                        {"Development": "guides/development.md"},
                        {"Contributing": "guides/contributing.md"},
                    ]
                },
                {"API Reference": "api/README.md"},
            ],
            "plugins": ["search", "mkdocstrings"],
        }

        config_path = self.output_dir / "mkdocs.yml"
        with open(config_path, "w") as f:
            import yaml

            yaml.dump(config, f, default_flow_style=False)

    def _generate_docusaurus_sidebar(self):
        """Generate Docusaurus sidebar configuration."""
        sidebar = {
            "docs": [
                {"type": "doc", "id": "index"},
                {
                    "type": "category",
                    "label": "Guides",
                    "items": [
                        "guides/installation",
                        "guides/quickstart",
                        "guides/configuration",
                        "guides/development",
                        "guides/contributing",
                    ],
                },
                {"type": "category", "label": "API Reference", "items": ["api/index"]},
            ]
        }

        sidebar_path = self.output_dir / "sidebar.js"
        with open(sidebar_path, "w") as f:
            f.write(f"module.exports = {json.dumps(sidebar, indent=2)};")

    def _generate_vuepress_config(self):
        """Generate VuePress configuration."""
        config = {
            "title": self.project_info.get("project_name", "Documentation"),
            "description": self.project_info.get("description", ""),
            "themeConfig": {
                "nav": [
                    {"text": "Home", "link": "/"},
                    {"text": "Guide", "link": "/guides/"},
                    {"text": "API", "link": "/api/"},
                ],
                "sidebar": {
                    "/guides/": [
                        "installation",
                        "quickstart",
                        "configuration",
                        "development",
                        "contributing",
                    ],
                    "/api/": "auto",
                },
            },
        }

        config_path = self.output_dir / ".vuepress" / "config.js"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            f.write(f"module.exports = {json.dumps(config, indent=2)}")

    def _write_document(self, doc: MarkdownDocument, path: Path):
        """Write markdown document to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc.to_markdown())


def main():
    """Main function for markdown documentation generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate markdown documentation")
    parser.add_argument("--source", default="src", help="Source directory")
    parser.add_argument("--output", default="docs_markdown", help="Output directory")
    parser.add_argument(
        "--style",
        choices=["github", "mkdocs", "sphinx_md", "docusaurus", "vuepress"],
        default="github",
        help="Markdown style",
    )
    parser.add_argument("--modules-file", help="JSON file with module information")
    parser.add_argument("--project-name", default="My Project", help="Project name")
    parser.add_argument("--project-version", default="1.0.0", help="Project version")

    args = parser.parse_args()

    # Load modules info if provided
    modules_info = {}
    if args.modules_file and Path(args.modules_file).exists():
        with open(args.modules_file, "r") as f:
            modules_info = json.load(f)

    # Project info
    project_info = {
        "project_name": args.project_name,
        "version": args.project_version,
        "package_name": args.project_name.lower().replace(" ", "-"),
        "module_name": args.project_name.lower().replace(" ", "_"),
        "python_version": "3.9",
        "license": "MIT",
        "repository_url": "https://github.com/username/repo",
        "description": "A comprehensive Python project with AI-powered documentation",
    }

    # Build documentation
    builder = MarkdownDocumentationBuilder(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        project_info=project_info,
        style=MarkdownStyle(args.style),
    )

    results = builder.build_documentation(
        modules_info=modules_info, include_api=True, include_guides=True
    )

    # Print results
    print("\n✅ Markdown documentation generated successfully!")
    print(f"📁 Output directory: {results['output_dir']}")
    print(f"📝 Style: {results['style']}")
    print(f"📄 Files generated: {len(results['generated_files'])}")

    # Instructions for viewing
    print("\n📖 View your documentation:")
    if args.style == "github":
        print(f"  Open {Path(args.output) / 'README.md'} in any markdown viewer")
    elif args.style == "mkdocs":
        print("  Run: mkdocs serve")
    elif args.style == "docusaurus":
        print("  Run: npm run start")
    elif args.style == "vuepress":
        print("  Run: vuepress dev")


if __name__ == "__main__":
    main()
