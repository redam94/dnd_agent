"""
Wrapper script for markdown documentation generation.
Integrates with the project's documentation pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the integrated generator
from integrated_doc_generator import main

if __name__ == "__main__":
    main()
