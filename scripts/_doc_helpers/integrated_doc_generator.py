"""
Integrated AI Documentation Generator with Markdown Output
Combines AI-powered documentation generation with markdown formatting.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import the documentation generators
from generate_docs import BatchAIDocGenerator, DocStyle, DocumentationCache, EnhancedCodeAnalyzer
from markdown_builder import (
    CodeDocumenter,
    MarkdownDocument,
    MarkdownDocumentationBuilder,
    MarkdownSection,
    MarkdownStyle,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegratedDocumentationGenerator:
    """
    Integrates AI documentation generation with markdown output.
    Provides a complete pipeline from code analysis to markdown documentation.
    """

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        ai_provider: str = "openai",
        markdown_style: MarkdownStyle = MarkdownStyle.GITHUB,
        doc_style: DocStyle = DocStyle.NUMPY,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the integrated documentation generator.

        Args:
            source_dir: Directory containing source code
            output_dir: Directory for output documentation
            ai_provider: AI provider for documentation generation
            markdown_style: Style of markdown to generate
            doc_style: Docstring style (numpy, google, sphinx)
            cache_dir: Directory for caching AI responses
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.ai_provider = ai_provider
        self.markdown_style = markdown_style
        self.doc_style = doc_style
        self.cache_dir = cache_dir or Path(".doc_cache")

        # Results tracking
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "analyzed_modules": {},
            "generated_docs": {},
            "markdown_files": {},
            "statistics": {},
        }

    def generate_complete_documentation(
        self, enhance_with_ai: bool = True, include_source: bool = False, batch_size: int = 5
    ) -> Dict[str, Any]:
        """
        Generate complete documentation with AI enhancement and markdown output.

        Args:
            enhance_with_ai: Whether to use AI to enhance documentation
            include_source: Whether to include source code in documentation
            batch_size: Batch size for AI requests

        Returns:
            Dictionary containing generation results
        """
        logger.info("🚀 Starting integrated documentation generation...")

        # Phase 1: Analyze code structure
        logger.info("📊 Phase 1: Analyzing code structure...")
        analyzer = EnhancedCodeAnalyzer(str(self.source_dir))
        elements = analyzer.analyze()

        self.results["analyzed_modules"] = {
            "total_elements": len(elements),
            "by_type": self._count_by_type(elements),
        }
        logger.info(f"✅ Analyzed {len(elements)} code elements")

        # Phase 2: Enhance with AI if requested
        enhanced_docs = {}
        if enhance_with_ai:
            logger.info("🤖 Phase 2: Enhancing documentation with AI...")
            doc_gen = BatchAIDocGenerator(
                provider=self.ai_provider, cache_dir=self.cache_dir, batch_size=batch_size
            )

            # Convert elements to list for batch processing
            elements_list = list(elements.values())
            enhanced_docs = doc_gen.generate_batch_documentation(
                elements_list, style=self.doc_style
            )

            self.results["generated_docs"] = {
                "enhanced_count": len(enhanced_docs),
                "ai_provider": self.ai_provider,
                "total_cost": doc_gen.total_cost,
            }
            logger.info(f"✅ Enhanced {len(enhanced_docs)} documentation entries")

        # Phase 3: Convert to structured module information
        logger.info("📝 Phase 3: Structuring module information...")
        modules_info = self._structure_modules_info(elements, enhanced_docs, include_source)

        # Phase 4: Generate markdown documentation
        logger.info("📄 Phase 4: Generating markdown documentation...")
        markdown_results = self._generate_markdown_docs(modules_info)

        self.results["markdown_files"] = markdown_results

        # Phase 5: Generate additional formats
        logger.info("📚 Phase 5: Generating additional documentation formats...")
        self._generate_additional_formats(modules_info)

        # Calculate statistics
        self._calculate_statistics(elements, enhanced_docs)

        # Save results
        self._save_results()

        logger.info("✅ Documentation generation complete!")
        self._print_summary()

        return self.results

    def _count_by_type(self, elements: Dict[str, Any]) -> Dict[str, int]:
        """Count elements by type."""
        counts = {}
        for element in elements.values():
            elem_type = element.type
            counts[elem_type] = counts.get(elem_type, 0) + 1
        return counts

    def _structure_modules_info(
        self, elements: Dict[str, Any], enhanced_docs: Dict[str, str], include_source: bool
    ) -> Dict[str, Any]:
        """
        Structure code elements into module information for markdown generation.
        """
        modules_info = {}

        for key, element in elements.items():
            # Determine module path
            if "::" in key:
                module_path = key.split("::")[0]
            else:
                module_path = key

            # Initialize module info if not exists
            if module_path not in modules_info:
                modules_info[module_path] = {
                    "name": Path(module_path).stem,
                    "path": module_path,
                    "docstring": "",
                    "classes": [],
                    "functions": [],
                    "imports": [],
                    "statistics": {},
                }

            # Add enhanced documentation if available
            doc_content = enhanced_docs.get(element.hash, element.docstring)

            # Structure element information
            element_info = {
                "name": element.name,
                "signature": element.signature,
                "docstring": doc_content,
                "type": element.type,
                "decorators": element.decorators,
                "complexity": element.complexity,
                "line_number": element.line_number,
            }

            # Add source if requested
            if include_source:
                element_info["source"] = element.source

            # Parse parameters for functions/methods
            if element.type in ["function", "method"]:
                element_info["params"] = self._parse_parameters(element)
                element_info["return_type"] = self._extract_return_type(element)

            # Categorize element
            if element.type == "module":
                modules_info[module_path]["docstring"] = doc_content
            elif element.type == "class":
                # For classes, we need to gather methods
                class_info = element_info.copy()
                class_info["methods"] = []
                class_info["properties"] = []
                class_info["base_classes"] = []

                # Find methods belonging to this class
                class_prefix = f"{key}::"
                for method_key, method_element in elements.items():
                    if method_key.startswith(class_prefix) and method_element.type in [
                        "method",
                        "function",
                    ]:
                        method_info = {
                            "name": method_element.name,
                            "signature": method_element.signature,
                            "docstring": enhanced_docs.get(
                                method_element.hash, method_element.docstring
                            ),
                            "params": self._parse_parameters(method_element),
                            "return_type": self._extract_return_type(method_element),
                            "decorators": method_element.decorators,
                        }

                        if "property" in method_element.decorators:
                            class_info["properties"].append(method_element.name)
                        else:
                            class_info["methods"].append(method_info)

                modules_info[module_path]["classes"].append(class_info)

            elif element.type == "function" and "::" not in key:
                # Module-level function
                modules_info[module_path]["functions"].append(element_info)

        return modules_info

    def _parse_parameters(self, element: Any) -> List[Dict[str, Any]]:
        """Parse function/method parameters."""
        params = []
        # This would need actual AST parsing in production
        # For now, return a simplified version
        if hasattr(element, "params"):
            return element.params
        return params

    def _extract_return_type(self, element: Any) -> Optional[str]:
        """Extract return type from element."""
        if hasattr(element, "return_type"):
            return element.return_type
        return None

    def _generate_markdown_docs(self, modules_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate markdown documentation files."""
        # Get project information
        project_info = self._get_project_info()

        # Create markdown builder
        builder = MarkdownDocumentationBuilder(
            source_dir=self.source_dir,
            output_dir=self.output_dir,
            project_info=project_info,
            style=self.markdown_style,
        )

        # Build documentation
        results = builder.build_documentation(
            modules_info=modules_info, include_api=True, include_guides=True
        )

        return results

    def _get_project_info(self) -> Dict[str, Any]:
        """Get project information for documentation."""
        # Try to read from setup.py, pyproject.toml, or package.json
        project_info = {
            "project_name": self.source_dir.name,
            "package_name": self.source_dir.name.lower().replace("_", "-"),
            "module_name": self.source_dir.name.lower().replace("-", "_"),
            "version": "1.0.0",
            "python_version": "3.9",
            "description": "AI-enhanced documentation",
            "license": "MIT",
        }

        # Try to read pyproject.toml
        pyproject_path = self.source_dir.parent / "pyproject.toml"
        if pyproject_path.exists():
            try:
                import toml

                data = toml.load(pyproject_path)
                if "project" in data:
                    project = data["project"]
                    project_info["project_name"] = project.get("name", project_info["project_name"])
                    project_info["version"] = project.get("version", project_info["version"])
                    project_info["description"] = project.get(
                        "description", project_info["description"]
                    )
            except ImportError:
                pass  # toml not installed

        return project_info

    def _generate_additional_formats(self, modules_info: Dict[str, Any]):
        """Generate additional documentation formats."""
        # Generate JSON API specification
        api_json_path = self.output_dir / "api.json"
        with open(api_json_path, "w") as f:
            json.dump(modules_info, f, indent=2, default=str)

        # Generate search index for documentation
        search_index = self._generate_search_index(modules_info)
        search_path = self.output_dir / "search_index.json"
        with open(search_path, "w") as f:
            json.dump(search_index, f, indent=2)

        # Generate sitemap for documentation
        sitemap = self._generate_sitemap()
        sitemap_path = self.output_dir / "sitemap.txt"
        with open(sitemap_path, "w") as f:
            f.write("\n".join(sitemap))

    def _generate_search_index(self, modules_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate search index for documentation."""
        index = []

        for module_path, module_info in modules_info.items():
            # Index module
            _name = module_info["name"]
            _content = module_info.get("docstring", "")
            if _content is None:
                _content = ""
            index.append(
                {
                    "title": f"Module: {_name}",
                    "url": f"api/{_name}.md",
                    "content": _content[:200],
                    "type": "module",
                }
            )

            # Index classes
            for class_info in module_info.get("classes", []):
                index.append(
                    {
                        "title": f"Class: {class_info['name']}",
                        "url": f"api/{module_info['name']}.md#{class_info['name'].lower()}",
                        "content": class_info.get("docstring", "")[:200],
                        "type": "class",
                    }
                )

                # Index methods
                for method in class_info.get("methods", []):
                    index.append(
                        {
                            "title": f"{class_info['name']}.{method['name']}",
                            "url": f"api/{module_info['name']}.md#{class_info['name'].lower()}-{method['name'].lower()}",
                            "content": method.get("docstring", "")[:200],
                            "type": "method",
                        }
                    )

            # Index functions
            for func in module_info.get("functions", []):
                index.append(
                    {
                        "title": f"Function: {func['name']}",
                        "url": f"api/{module_info['name']}.md#{func['name'].lower()}",
                        "content": func.get("docstring", "")[:200],
                        "type": "function",
                    }
                )

        return index

    def _generate_sitemap(self) -> List[str]:
        """Generate sitemap for documentation."""
        sitemap = []

        # Add main pages
        sitemap.extend(["index.md", "README.md", "api/README.md"])

        # Add guide pages
        guides = [
            "guides/installation.md",
            "guides/quickstart.md",
            "guides/configuration.md",
            "guides/development.md",
            "guides/contributing.md",
        ]
        sitemap.extend(guides)

        # Add API pages
        for file in (self.output_dir / "api").glob("*.md"):
            sitemap.append(f"api/{file.name}")

        return sitemap

    def _calculate_statistics(self, elements: Dict[str, Any], enhanced_docs: Dict[str, str]):
        """Calculate documentation statistics."""
        total_elements = len(elements)
        documented_before = sum(1 for e in elements.values() if e.docstring)
        documented_after = len(enhanced_docs)

        self.results["statistics"] = {
            "total_elements": total_elements,
            "documented_before": documented_before,
            "documented_after": documented_after,
            "coverage_before": (
                (documented_before / total_elements * 100) if total_elements > 0 else 0
            ),
            "coverage_after": (
                ((documented_before + documented_after) / total_elements * 100)
                if total_elements > 0
                else 0
            ),
            "improvement": documented_after,
            "complexity_average": (
                sum(e.complexity for e in elements.values()) / total_elements
                if total_elements > 0
                else 0
            ),
        }

    def _save_results(self):
        """Save generation results to file."""
        results_path = self.output_dir / "generation_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def _print_summary(self):
        """Print generation summary."""
        print("\n" + "=" * 60)
        print("📚 Documentation Generation Summary")
        print("=" * 60)

        stats = self.results.get("statistics", {})
        print(f"\n📊 Code Analysis:")
        print(f"  Elements analyzed: {stats.get('total_elements', 0)}")
        print(f"  Average complexity: {stats.get('complexity_average', 0):.1f}")

        if self.results.get("generated_docs"):
            print(f"\n🤖 AI Enhancement:")
            print(
                f"  Documents enhanced: {self.results['generated_docs'].get('enhanced_count', 0)}"
            )
            print(f"  Total cost: ${self.results['generated_docs'].get('total_cost', 0):.2f}")

        print(f"\n📝 Documentation Coverage:")
        print(f"  Before: {stats.get('coverage_before', 0):.1f}%")
        print(f"  After: {stats.get('coverage_after', 0):.1f}%")
        print(f"  Improvement: +{stats.get('improvement', 0)} items")

        print(f"\n📁 Output:")
        print(f"  Location: {self.output_dir}")
        print(f"  Markdown style: {self.markdown_style.value}")

        print("\n📖 View your documentation:")
        print(f"  Open: {self.output_dir / 'README.md'}")

        if self.markdown_style == MarkdownStyle.MKDOCS:
            print("  Or run: mkdocs serve")
        elif self.markdown_style == MarkdownStyle.GITHUB:
            print("  Or push to GitHub for automatic rendering")

        print("\n" + "=" * 60)


def main():
    """Main entry point for integrated documentation generation."""
    parser = argparse.ArgumentParser(description="Generate AI-enhanced markdown documentation")

    # Input/Output options
    parser.add_argument("--source", default="src", help="Source directory")
    parser.add_argument("--output", default="docs_markdown", help="Output directory")

    # AI options
    parser.add_argument(
        "--ai-provider",
        choices=["openai", "anthropic", "google", "none"],
        default="openai",
        help="AI provider for enhancement",
    )
    parser.add_argument("--no-ai", action="store_true", help="Skip AI enhancement")
    parser.add_argument("--batch-size", type=int, default=5, help="AI batch size")
    parser.add_argument("--cache-dir", default=".doc_cache", help="Cache directory")

    # Documentation options
    parser.add_argument(
        "--markdown-style",
        choices=["github", "mkdocs", "sphinx_md", "docusaurus", "vuepress"],
        default="github",
        help="Markdown documentation style",
    )
    parser.add_argument(
        "--doc-style",
        choices=["numpy", "google", "sphinx", "markdown"],
        default="numpy",
        help="Docstring style",
    )
    parser.add_argument(
        "--include-source", action="store_true", help="Include source code in documentation"
    )

    # Additional options
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without generating")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create generator
    generator = IntegratedDocumentationGenerator(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        ai_provider="none" if args.no_ai else args.ai_provider,
        markdown_style=MarkdownStyle(args.markdown_style),
        doc_style=DocStyle(args.doc_style),
        cache_dir=Path(args.cache_dir),
    )

    if args.dry_run:
        # Just analyze
        analyzer = EnhancedCodeAnalyzer(str(Path(args.source)))
        elements = analyzer.analyze()
        print(f"Would process {len(elements)} code elements")
        for key, element in list(elements.items())[:5]:
            print(f"  - {element.type}: {element.name}")
        return

    # Generate documentation
    results = generator.generate_complete_documentation(
        enhance_with_ai=not args.no_ai,
        include_source=args.include_source,
        batch_size=args.batch_size,
    )

    # Save final report
    report_path = Path(args.output) / "documentation_report.md"
    with open(report_path, "w") as f:
        f.write("# Documentation Generation Report\n\n")
        f.write(f"Generated: {results['timestamp']}\n\n")
        f.write("## Statistics\n\n")
        for key, value in results.get("statistics", {}).items():
            f.write(f"- {key}: {value}\n")


if __name__ == "__main__":
    main()
