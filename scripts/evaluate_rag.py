import json
import os
from pathlib import Path

from app.rag.evaluation_harness import run_rag_evaluation


def main() -> None:
    output_path = Path(os.getenv("RAG_EVALUATION_OUTPUT_PATH", "scripts/rag_evaluation_results.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = run_rag_evaluation(output_path=str(output_path))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
