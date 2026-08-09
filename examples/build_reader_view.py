"""Print one deterministic reader-SFT record from the bundled demo input."""

from __future__ import annotations

import json
from pathlib import Path

from bactrainus.data import ReaderViewBuilder, load_hotpot_examples


def main() -> None:
    source = Path(__file__).with_name("sample_hotpot.json")
    example = load_hotpot_examples(source, split="train")[0]
    record = ReaderViewBuilder().build(example)
    print(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
