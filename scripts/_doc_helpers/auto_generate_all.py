"""
Combined automation script for generating tests and documentation.
Orchestrates both AI generators for complete automation.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_docs import main as generate_docs_main
from generate_tests import main as generate_tests_main

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AutomationOrchestrator:
    """Orchestrates AI generation for tests and documentation."""

    def __init__(self, config_path: Path = None):
        self.config = self._load_config(config_path)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "documentation": {},
            "tests": {},
            "quality": {},
        }

    def _load_config(self, config_path: Path = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path("automation/config.yaml")

        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "ai_generation": {
                "documentation": {"enabled": True, "provider": "openai", "style": "numpy"},
                "tests": {"enabled": True, "provider": "openai", "framework": "pytest"},
            },
            "paths": {"source_dir": "src", "test_dir": "tests", "docs_dir": "docs"},
        }

    async def run_generation(
        self, docs: bool = True, tests: bool = True, analyze: bool = True
    ) -> Dict[str, Any]:
        """Run the complete generation pipeline."""
        logger.info("🚀 Starting AI generation pipeline...")

        # Check for API keys
        if not self._check_api_keys():
            return self.results

        # Phase 1: Analyze code structure
        if analyze:
            logger.info("📊 Phase 1: Analyzing code structure...")
            self.results["analysis"] = await self._analyze_code()

        # Phase 2: Generate documentation
        if docs and self.config["ai_generation"]["documentation"]["enabled"]:
            logger.info("📚 Phase 2: Generating documentation...")
            self.results["documentation"] = await self._generate_documentation()

        # Phase 3: Generate tests
        if tests and self.config["ai_generation"]["tests"]["enabled"]:
            logger.info("🧪 Phase 3: Generating tests...")
            self.results["tests"] = await self._generate_tests()

        # Phase 4: Quality check
        if analyze:
            logger.info("✅ Phase 4: Running quality checks...")
            self.results["quality"] = await self._run_quality_checks()

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

        return self.results

    def _check_api_keys(self) -> bool:
        """Check if required API keys are set."""
        provider = self.config["ai_generation"]["documentation"].get("provider", "openai")

        if provider == "none":
            return True

        key_name = f"{provider.upper()}_API_KEY"
        if not os.getenv(key_name):
            logger.error(f"❌ {key_name} not found in environment variables!")
            logger.info(f"Please set {key_name} in your .env file or environment")
            return False
        return True

    async def _analyze_code(self) -> Dict[str, Any]:
        """Analyze code structure."""
        source_dir = Path(self.config["paths"]["source_dir"])

        # Count files and lines
        py_files = list(source_dir.rglob("*.py"))
        total_lines = 0

        for file in py_files:
            with open(file, "r") as f:
                total_lines += len(f.readlines())

        return {
            "files": len(py_files),
            "lines_of_code": total_lines,
            "modules": [str(f.relative_to(source_dir)) for f in py_files],
        }

    async def _generate_documentation(self) -> Dict[str, Any]:
        """Generate documentation using AI."""
        try:
            # Prepare arguments
            args = [
                "--source",
                self.config["paths"]["source_dir"],
                "--docs",
                self.config["paths"]["docs_dir"],
                "--style",
                self.config["ai_generation"]["documentation"]["style"],
                "--provider",
                self.config["ai_generation"]["documentation"]["provider"],
                "--enhance",
            ]

            # Run documentation generator
            sys.argv = ["generate_docs.py"] + args
            generate_docs_main()

            return {"status": "success", "message": "Documentation generated successfully"}

        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def _generate_tests(self) -> Dict[str, Any]:
        """Generate tests using AI."""
        try:
            # Prepare arguments
            args = [
                "--source",
                self.config["paths"]["source_dir"],
                "--tests",
                self.config["paths"]["test_dir"],
                "--framework",
                self.config["ai_generation"]["tests"]["framework"],
                "--provider",
                self.config["ai_generation"]["tests"]["provider"],
            ]

            # Run test generator
            sys.argv = ["generate_tests.py"] + args
            generate_tests_main()

            return {"status": "success", "message": "Tests generated successfully"}

        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def _run_quality_checks(self) -> Dict[str, Any]:
        """Run code quality checks."""
        import subprocess

        results = {}

        # Run pytest
        try:
            result = subprocess.run(
                ["pytest", self.config["paths"]["test_dir"], "--cov", "--quiet"],
                capture_output=True,
                text=True,
            )
            results["tests"] = {
                "passed": "FAILED" not in result.stdout,
                "output": result.stdout[-500:],  # Last 500 chars
            }
        except Exception as e:
            results["tests"] = {"error": str(e)}

        # Run linting
        try:
            result = subprocess.run(
                ["flake8", self.config["paths"]["source_dir"], "--count"],
                capture_output=True,
                text=True,
            )
            results["linting"] = {
                "issues": int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
            }
        except Exception as e:
            results["linting"] = {"error": str(e)}

        return results

    def _save_results(self):
        """Save generation results to file."""
        results_file = Path("automation_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"📁 Results saved to {results_file}")

    def _print_summary(self):
        """Print generation summary."""
        print("\n" + "=" * 60)
        print("🎉 AI Generation Pipeline Complete!")
        print("=" * 60)

        if "analysis" in self.results:
            print(f"\n📊 Code Analysis:")
            print(f"  - Files analyzed: {self.results['analysis']['files']}")
            print(f"  - Lines of code: {self.results['analysis']['lines_of_code']}")

        if "documentation" in self.results:
            print(f"\n📚 Documentation:")
            print(f"  - Status: {self.results['documentation'].get('status', 'N/A')}")

        if "tests" in self.results:
            print(f"\n🧪 Tests:")
            print(f"  - Status: {self.results['tests'].get('status', 'N/A')}")

        if "quality" in self.results:
            print(f"\n✅ Quality Checks:")
            if "tests" in self.results["quality"]:
                print(f"  - Tests passed: {self.results['quality']['tests'].get('passed', 'N/A')}")
            if "linting" in self.results["quality"]:
                print(
                    f"  - Linting issues: {self.results['quality']['linting'].get('issues', 'N/A')}"
                )

        print("\n" + "=" * 60)
        print("📖 Next steps:")
        print("  1. Review generated tests: make test")
        print("  2. View documentation: make serve-docs")
        print("  3. Check coverage: make coverage")
        print("=" * 60 + "\n")


def main():
    """Main entry point for automation script."""
    parser = argparse.ArgumentParser(
        description="Orchestrate AI generation for tests and documentation"
    )
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--docs", action="store_true", default=True, help="Generate documentation")
    parser.add_argument("--tests", action="store_true", default=True, help="Generate tests")
    parser.add_argument(
        "--no-docs", dest="docs", action="store_false", help="Skip documentation generation"
    )
    parser.add_argument(
        "--no-tests", dest="tests", action="store_false", help="Skip test generation"
    )
    parser.add_argument("--analyze", action="store_true", default=True, help="Run code analysis")

    args = parser.parse_args()

    # Run orchestrator
    orchestrator = AutomationOrchestrator(config_path=Path(args.config) if args.config else None)

    # Run async pipeline
    asyncio.run(orchestrator.run_generation(docs=args.docs, tests=args.tests, analyze=args.analyze))


if __name__ == "__main__":
    main()
