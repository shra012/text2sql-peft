"""Pull the published series back from the portfolio site into blog/series and blog/svg.

The reverse of port_series.py, for when the posts were edited directly in the site. Inline charts are
matched, in order, to the <!--SVG:name--> placeholders already in blog/series/partN.html, saved back to
blog/svg/<name>.svg and replaced by their placeholder. The result is re-expanded and must reproduce the
published post exactly; the hero images are copied back too.

  .venv/bin/python blog/tools/pull_series.py
"""

from __future__ import annotations

import json
import re
import shutil

from port_series import PARTS, ROOT, SITE, WRAP, expand, post_module

IDS = {1: "finetuning_series_1_methods", 2: "finetuning_series_2_how", 3: "finetuning_series_3_results"}
SVG_RE = re.compile(re.escape(WRAP) + r"(<svg\b.*?</svg>)</div>", re.S)
PLACEHOLDER_RE = re.compile(r"<!--\s*SVG:([A-Za-z0-9_]+)\s*-->")


def unjs(s: str) -> str:
    """Undo port_series.js() escaping of a template literal."""
    return s.replace("\\${", "${").replace("\\`", "`").replace("\\\\", "\\")


def field(src: str, name: str):
    m = re.search(rf"^  {name}: (.*),$", src, re.M)
    if not m:
        raise SystemExit(f"missing {name}")
    return json.loads(m.group(1))


def pull_part(n: int) -> None:
    html_path = ROOT / f"blog/series/part{n}.html"
    names = iter(PLACEHOLDER_RE.findall(html_path.read_text()))
    src = (SITE / f"src/constants/posts/{IDS[n]}.js").read_text()
    content = unjs(re.search(r"  content: `\n(.*)\n`,\n};", src, re.S).group(1))

    def back(m: re.Match) -> str:
        name = next(names, None)
        if not name:
            raise SystemExit(f"part{n}: the post has more charts than {html_path.name}")
        path = ROOT / f"blog/svg/{name}.svg"
        ending = "\n" if path.read_text().endswith("\n") else ""  # keep the file's trailing newline as it was
        path.write_text(m.group(1) + ending)
        return f"<!--SVG:{name}-->"

    html = SVG_RE.sub(back, content)
    if next(names, None):
        raise SystemExit(f"part{n}: the post has fewer charts than {html_path.name}")
    meta = {
        "id": IDS[n],
        "title": field(src, "title"),
        "aiSummary": field(src, "aiSummary"),
        "tags": field(src, "tags"),
    }
    html_path.write_text(html)
    (ROOT / f"blog/series/part{n}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    hero = f"hero_part{n}.png"
    if post_module(meta, n, f"blog/text2sql-peft/{hero}", expand(html, n)) != src:
        raise SystemExit(f"part{n}: round trip does not reproduce the published post")
    shutil.copy(SITE / f"public/blog/text2sql-peft/{hero}", ROOT / f"blog/series/{hero}")
    print(f"part{n}: {meta['title']!r}  {content.count('<svg')} charts  round trip ok")


def main() -> None:
    for n in PARTS:
        pull_part(n)


if __name__ == "__main__":
    main()
