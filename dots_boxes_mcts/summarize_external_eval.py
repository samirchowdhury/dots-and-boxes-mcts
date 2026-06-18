from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_boxes_mcts.external_bot_common import summarize_external_records


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf8") as input_file:
        for line in input_file:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize external-bot eval JSONL output.")
    parser.add_argument("path", type=Path, help="Path to a dotsandboxes.org eval JSONL file.")
    args = parser.parse_args()

    print(json.dumps(summarize_external_records(read_jsonl(args.path)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
