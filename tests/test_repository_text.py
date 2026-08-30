"""Repository-wide printable ASCII and US English checks."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

NON_US_SPELLINGS = (
    "analy" + "se",
    "analy" + "sed",
    "analy" + "sing",
    "behavio" + "ur",
    "behavio" + "urs",
    "behavio" + "ural",
    "catalog" + "ue",
    "cent" + "re",
    "cent" + "res",
    "colo" + "ur",
    "colo" + "urs",
    "defen" + "ce",
    "favo" + "ur",
    "favo" + "urite",
    "gre" + "y",
    "fulfil" + "ment",
    "initial" + "ise",
    "initial" + "ised",
    "initial" + "ising",
    "label" + "led",
    "label" + "ling",
    "licen" + "ce",
    "model" + "led",
    "model" + "ling",
    "offen" + "ce",
    "optim" + "ise",
    "optim" + "ised",
    "optim" + "ising",
    "organi" + "sation",
    "organi" + "sations",
    "organi" + "se",
    "organi" + "sed",
    "organi" + "sing",
    "priorit" + "ise",
    "priorit" + "ised",
    "priorit" + "ising",
    "progra" + "mme",
    "recogni" + "se",
    "recogni" + "sed",
    "recogni" + "sing",
    "standard" + "ise",
    "standard" + "ised",
    "standard" + "ising",
    "summar" + "ise",
    "summar" + "ised",
    "summar" + "ising",
    "synchron" + "ise",
    "synchron" + "ised",
    "synchron" + "ising",
    "travel" + "led",
    "travel" + "ler",
    "travel" + "ling",
    "visual" + "ise",
    "visual" + "ised",
    "visual" + "ising",
)


def test_repository_text_uses_printable_ascii_and_us_english() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    violations: list[str] = []
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, NON_US_SPELLINGS)) + r")\b", re.I)

    for raw_path in sorted(completed.stdout.split(b"\0")):
        if not raw_path:
            continue
        if any(byte < 32 or byte > 126 for byte in raw_path):
            violations.append(f"{raw_path!r}: non-ASCII path")
            continue
        path = root / raw_path.decode("ascii")
        content = path.read_bytes()
        if any(byte not in (9, 10, 13) and not 32 <= byte <= 126 for byte in content):
            violations.append(f"{path.relative_to(root)}: non-ASCII content")
            continue
        text = content.decode("ascii")
        for line_number, line in enumerate(text.splitlines(), 1):
            for match in pattern.finditer(line):
                violations.append(
                    f"{path.relative_to(root)}:{line_number}: non-US spelling {match.group(0)}"
                )

    assert violations == []
