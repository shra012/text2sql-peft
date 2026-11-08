"""Port the 3-part series (blog/series/part{1,2,3}.{html,json}) into the portfolio site, one module per post.

- expands <!--SVG:name--> to blog/svg/name.svg inside the house scroll wrapper
- checks the house format (no h1/h2/figure/img, balanced tags, every SVG placeholder resolved)
- writes frontend/src/constants/posts/<id>.js (with part: N for ordering) and removes the old single post
- copies hero_partN.png into frontend/public/blog/text2sql-peft/

  .venv/bin/python blog/tools/port_series.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = Path("/home/shravan/Workspace/Portfolio/frontend")
WRAP = '<div style="overflow-x:auto;margin:1.9rem 0">'
OLD_IDS = ["text2sql_finetuning_5_techniques"]
DATE = "November 07, 2026"
PARTS = (1, 2, 3)
HTML_VOID = {"br", "hr", "img", "input", "meta", "link"}
SVG_LEAVES = {"path", "rect", "line", "circle", "polyline", "polygon", "stop", "use", "ellipse"}
VOID = HTML_VOID | SVG_LEAVES  # tags that never get a separate closing tag in our markup


class Balance(HTMLParser):
    """Collects unmatched open/close tags."""

    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass  # self-closing tags (<path/>, <br/>) never need a matching close tag

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")


def expand(html: str, part: int) -> str:
    def svg(m: re.Match) -> str:
        f = ROOT / f"blog/svg/{m.group(1)}.svg"
        if not f.exists():
            raise SystemExit(f"part{part}: missing SVG {f}")
        return WRAP + f.read_text().strip() + "</div>"

    return re.sub(r"<!--\s*SVG:([A-Za-z0-9_]+)\s*-->", svg, html)


def check(html: str, part: int) -> None:
    bad = [t for t in ("<h1", "<h2", "<figure", "<img", "{{", "<!--SVG") if t in html]
    if bad:
        raise SystemExit(f"part{part}: forbidden/unresolved markup {bad}")
    b = Balance()
    b.feed(html)
    if b.errors or b.stack:
        raise SystemExit(f"part{part}: unbalanced HTML {b.errors[:5]} open={b.stack[:5]}")


def js(s: str) -> str:
    """Escape for a JS template literal."""
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def post_module(meta: dict, part: int, img: str, html: str) -> str:
    return (
        "const post = {\n"
        f"  id: {json.dumps(meta['id'])},\n"
        f"  title: {json.dumps(meta['title'], ensure_ascii=False)},\n"
        f"  date: {json.dumps(DATE)},\n"
        f"  part: {part},\n"
        f"  aiSummary: {json.dumps(meta['aiSummary'], ensure_ascii=False)},\n"
        f"  image: {json.dumps(img)},\n"
        f"  tags: {json.dumps(meta['tags'], ensure_ascii=False)},\n"
        "  type: 'Article',\n"
        f"  content: `\n{js(html)}\n`,\n"
        "};\n\nexport default post;\n"
    )


def port_part(n: int, dry_run: bool) -> str:
    """Build (and unless dry_run, write) one post; returns its id."""
    meta = json.loads((ROOT / f"blog/series/part{n}.json").read_text())
    html = expand((ROOT / f"blog/series/part{n}.html").read_text(), n)
    check(html, n)
    hero = ROOT / f"blog/series/hero_part{n}.png"
    if not hero.exists():
        raise SystemExit(f"missing hero image {hero}")
    img = f"blog/text2sql-peft/hero_part{n}.png"
    if not dry_run:
        if hero.exists():
            shutil.copy(hero, SITE / "public" / img)
        (SITE / f"src/constants/posts/{meta['id']}.js").write_text(post_module(meta, n, img, html))
    print(f"part{n}: {meta['id']}  {len(html):,} chars  {html.count('<svg')} svgs  image={img}")
    return meta["id"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Port the blog series into the portfolio site.")
    ap.add_argument("--dry-run", action="store_true", help="expand and check only; write nothing")
    args = ap.parse_args()
    ids = [port_part(n, args.dry_run) for n in PARTS]
    if not args.dry_run:
        for old in OLD_IDS:
            (SITE / f"src/constants/posts/{old}.js").unlink(missing_ok=True)
    for n in PARTS:  # links between parts must point at real ids
        html = (ROOT / f"blog/series/part{n}.html").read_text()
        for target in re.findall(r"href=['\"]#/blog/([^'\"]+)['\"]", html):
            if target not in ids:
                print(f"WARNING part{n} links to unknown post {target}")


if __name__ == "__main__":
    main()
