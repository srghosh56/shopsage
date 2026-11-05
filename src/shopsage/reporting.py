"""Write generated sections into docs/RESULTS.md between named markers."""

from __future__ import annotations

import re
from pathlib import Path

from shopsage.config import DOCS_DIR

RESULTS = DOCS_DIR / "RESULTS.md"


def write_section(name: str, markdown: str, path: Path = RESULTS) -> None:
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    block = f"{start}\n{markdown.strip()}\n{end}"
    text = path.read_text() if path.exists() else "# Results\n"
    if start in text:
        text = re.sub(re.escape(start) + ".*?" + re.escape(end), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)


def md_table(rows: list[dict], columns: list[str], fmt: str = "{:.4f}") -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" if i == 0 else "---:" for i in range(len(columns))) + "|"
    body = []
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            cells.append(fmt.format(v) if isinstance(v, float) else str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])
