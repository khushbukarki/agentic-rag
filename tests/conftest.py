import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Fast, offline settings for every test run: no model download, no API calls.
os.environ.update(
    EMBEDDER="hash",
    VECTOR_STORE="memory",
    OPENAI_API_KEY="",
    SAMPLE_DOCS_DIR=str(ROOT / "data/sample_docs"),
    LOAD_SAMPLE_DOCS="true",
)
