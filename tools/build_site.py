#!/usr/bin/env python3
"""Regenerate docs/index.html from the module headers.

The site is a view of the modules, the same way the notebooks are. Every node,
title, description and dependency edge is read out of the `.py` headers, so the
page cannot drift from the code. Run this after adding or renaming a module:

    python3 tools/build_site.py

It needs nothing but the standard library, so it runs on the host as well as in
the container.
"""
import glob
import html
import json
import os
import re

TRACKS = ["foundations", "core", "bioinformatics", "exercises"]
def _valid_keys():
    """Module ids that a dependency line may legitimately name.

    Derived from the tree rather than hardcoded: a hardcoded range silently
    drops edges for any module added later, which is how 41-46 went unlinked.
    """
    out = set()
    for track in TRACKS:
        for p in glob.glob(f"{ROOT}/statsPy/{track}/*.py"):
            out.add(os.path.basename(p)[:-3].split("_")[0])
    return out
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean(text):
    """Strip the light markup and LaTeX that headers use, for plain-text display."""
    text = re.sub(r"\$[^$]*\$", "", text)          # inline maths
    text = re.sub(r"\*\*|\*|`", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def describe(header):
    """One or two sentences describing the module, taken from its header.

    Headers vary: some open with a prose paragraph, some with a numbered
    'What you will learn' list, and the applied ones open with a markdown
    'Dataset card' table. Two passes: prefer real prose, and fall back to the
    bullets when a header has nothing else, so a table pipe or a list marker
    never becomes the summary.
    """
    header = re.sub(r"```.*?```", " ", header, flags=re.S)   # drop docker snippets
    sections = re.split(r"^##\s+", header, flags=re.M)

    for strict in (True, False):
        for sec in sections:
            lines = sec.split("\n")[1:] if sec else []
            keep = []
            for l in lines:
                t = l.strip()
                if not t or t in ("-", "*") or t.startswith(("|", ">", "#")):
                    continue
                if re.match(r"\*?\*?(Curriculum link|Assumes|Core modules used)", t):
                    continue
                if t.startswith(("- ", "* ")):
                    if strict:
                        continue
                    t = t[2:]
                keep.append(t)
            blob = clean(" ".join(keep))
            if strict:
                # "Part A: trajectories: ..." is scaffolding, not a description.
                blob = re.sub(r"^Part [A-Z]:\s*[^:]{0,30}:\s*", "", blob)
            numbered = re.match(r"^\d+\.\s*(.+)", blob)
            if numbered:
                blob = re.split(r"\s\d+\.\s", numbered.group(1))[0]
            if len(blob) < 40:
                continue
            if strict and not blob[:1].isupper():
                continue            # a fragment continuing a list, not prose
            parts = re.split(r"(?<=[.?])\s+", blob)
            out = parts[0]
            if len(out) < 90 and len(parts) > 1:
                out += " " + parts[1]
            return shorten(out, 230)
    return ""


def shorten(text, limit):
    """Trim to `limit` on a word boundary, without orphaning an open bracket."""
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0]
    while text.count("(") > text.count(")"):
        text = text[:text.rfind("(")].rstrip()
    return text.rstrip(" ,;:").rstrip()


def collect():
    nodes, by_key = [], {}
    for track in TRACKS:
        for order, path in enumerate(sorted(glob.glob(f"{ROOT}/statsPy/{track}/*.py"))):
            base = os.path.basename(path)[:-3]
            txt = open(path, encoding="utf-8").read()
            key = base.split("_")[0]
            m = re.search(r"^#\s+#\s+(.+)$", txt, re.M)
            full = clean(m.group(1)) if m else base
            title = full.split(":", 1)[1].strip() if ":" in full else full
            mt = re.search(r"\*\*Curriculum link:\*\*\s*(.+)$", txt, re.M)
            topic = clean(mt.group(1)) if mt else ""
            topic = re.sub(r",?\s*equations.*$", "", topic).replace("stats.md -> ", "").strip()
            cell = txt.split("# %%")[1] if "# %%" in txt else txt
            cell = "\n".join(re.sub(r"^#\s?", "", l) for l in cell.split("\n")
                              if l.startswith("#"))
            desc = describe(cell)
            deps = []
            md = re.search(r"\*\*(?:Assumes|Core modules used):\*\*\s*(.+?)(?:\n#\s*\n)", txt, re.S)
            if md:
                blob = re.sub(r"\n#\s?", " ", md.group(1)).replace("`", "")
                blob = re.sub(r"\([^)]*\)", " ", blob)          # drop glosses
                # 28a / 28b are module ids too, so the suffix must be allowed.
                deps = re.findall(r"\b([EF]\d|\d{2}[ab]?)\b", blob)
            nodes.append(dict(key=key, id=base, track=track, order=order, title=title,
                              topic=topic, desc=desc, _deps=deps))
            by_key[key] = base

    valid = _valid_keys()
    edges = []
    for n in nodes:
        for d in n.pop("_deps"):
            if (d in valid or re.match(r"^[EF]\d$", d)) and by_key.get(d, n["id"]) != n["id"]:
                edges.append({"from": by_key[d], "to": n["id"], "kind": "dep"})
    for track in TRACKS:                                       # reading order
        seq = [n for n in nodes if n["track"] == track]
        edges += [{"from": a["id"], "to": b["id"], "kind": "order"}
                  for a, b in zip(seq, seq[1:])]
    # Cross-track reading order: finish the beginner track before core, and
    # the whole core track before the applied one. Within a track, the file
    # order IS the reading order (see README section 6), because the numbers
    # were chosen so that sorting them gives the teaching sequence.
    for a, b in [("F6_connecting_variables", "01_study_design_and_estimands"),
                 ("40_simulation_and_benchmarking",
                  "20_bulk_rnaseq_differential_expression")]:
        edges.append({"from": a, "to": b, "kind": "order"})

    for n in nodes:                                            # escape once, here
        for f in ("title", "topic", "desc"):
            n[f] = html.escape(n[f], quote=False)
    return {"nodes": nodes, "edges": edges}


def main():
    data = collect()
    tpl = open(f"{ROOT}/docs/_template.html", encoding="utf-8").read()
    out = tpl.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    open(f"{ROOT}/docs/index.html", "w", encoding="utf-8").write(out)
    deps = sum(e["kind"] == "dep" for e in data["edges"])
    print(f"docs/index.html: {len(data['nodes'])} modules, {deps} dependency edges")


if __name__ == "__main__":
    main()
