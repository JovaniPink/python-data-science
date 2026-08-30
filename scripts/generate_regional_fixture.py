"""Generate the complete ignored cross-language synthetic source bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.test_regional import synthetic_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.output_dir / "source-bytes"
    source_dir.mkdir(parents=True, exist_ok=True)
    bundle = synthetic_bundle(source_dir)
    output = args.output_dir / "regional-source-bundle.v1.json"
    output.write_text(
        json.dumps(bundle, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
