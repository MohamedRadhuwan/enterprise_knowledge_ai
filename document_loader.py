from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_sample_document(file_name: str = "leave_policy.txt") -> None:
    doc_path = Path("data/documents") / file_name
    if not doc_path.exists():
        logger.warning(f"Sample document '{doc_path}' not found.")
        return

    text = doc_path.read_text(encoding="utf-8")
    print(f"=== Content of {file_name} ===")
    print(text[:500] + ("..." if len(text) > 500 else ""))


if __name__ == "__main__":
    inspect_sample_document()