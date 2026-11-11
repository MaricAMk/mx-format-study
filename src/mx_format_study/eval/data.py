from __future__ import annotations

import json
import urllib.request
from pathlib import Path

_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=Salesforce/wikitext&config=wikitext-2-raw-v1&split=test"
    "&offset={offset}&length={length}"
)
_PAGE = 100
_EXPECTED_ROWS = 4358

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def wikitext2_test_text(cache_dir: Path = DATA_DIR) -> str:
    cache = cache_dir / "wikitext-2-raw-v1-test.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    rows: list[str] = []
    while True:
        url = _ROWS_URL.format(offset=len(rows), length=_PAGE)
        with urllib.request.urlopen(url, timeout=60) as resp:
            page = json.load(resp)
        rows.extend(r["row"]["text"] for r in page["rows"])
        if len(rows) >= page["num_rows_total"]:
            break

    if len(rows) != _EXPECTED_ROWS:
        raise RuntimeError(f"expected {_EXPECTED_ROWS} WikiText-2 test rows, got {len(rows)}")

    text = "\n\n".join(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text
