#!/usr/bin/env python3
"""Generate the stats.md <-> module concordance, in both directions.

Since the renumbering, a module's number IS its `stats.md` topic number, so
the concordance is mostly mechanical. What is NOT mechanical, and what this
tool exists to keep honest, is:

  * the handful of topics served by more than one module (28a/28b), and the
    one module serving more than one topic (34);
  * the READING ORDER, which is no longer the same as the numeric order -
    the applied modules are numbered 20-30 but are read after the core ones;
  * the foundations and exercise tracks, which are not numbered by topic.

Every table it writes is delimited by markers, so the surrounding prose is
edited by hand and the tables are never edited by hand:

    <!-- BEGIN crossref:NAME -->   ... generated ...   <!-- END crossref:NAME -->

Run after adding or renaming a module.  `tools/check_docs.py` verifies the
committed files match what this would generate.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reading order is a teaching decision, not a numeric fact.  Stated once, here.
READING_ORDER = [
    ("Absolute basics", ["F1", "F2", "F3", "F4", "F5", "F6"], "-"),
    ("Foundations", ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"],
     "`E1` design audit"),
    ("Models", ["10", "11", "12", "13", "14", "15"], "-"),
    ("High-dimensional", ["16", "17", "18", "19"], "-"),
    ("Inference and validation", ["28a", "31", "32", "33", "34"],
     "`E3` prediction audit, `E4` causal question"),
    ("Modern core methods", ["36", "37", "38", "39", "40"], "-"),
    ("Applied assays", ["20", "21", "22", "23", "24", "25", "26", "27", "28b",
                        "29", "30"], "`E2` RNA-seq end to end"),
    ("Advanced applied", ["41", "42", "43", "44", "45", "46"], "-"),
]


def topics_from_stats_md():
    """topic number -> (title, part)."""
    out, part = {}, ""
    for line in open(os.path.join(ROOT, "stats.md")):
        m = re.match(r"^# (Part [^:]+):", line)
        if m:
            part = m.group(1)
        m = re.match(r"^## Topic (\d+): (.+)", line)
        if m:
            out[int(m.group(1))] = (m.group(2).strip(), part)
    return out


def modules():
    """module id -> dict(slug, dir, topics, title)."""
    out = {}
    for sub in ("foundations", "core", "bioinformatics", "exercises"):
        d = os.path.join(ROOT, "statsPy", sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py"):
                continue
            stem = fn[:-3]
            txt = open(os.path.join(d, fn)).read()
            mid = stem.split("_")[0]
            m = re.search(r"Curriculum link:\*{0,2}\s*`?stats\.md`?\s*->\s*(.+)", txt)
            spec = m.group(1).strip() if m else ""
            head = re.sub(r"\(\d+\.\d+\)", "", spec.split("equation")[0])
            topics = [int(x) for x in re.findall(r"\b(\d+)\b", head)]
            m = re.search(r"^#+ #+ (?:Module|Applied|Exercise|Foundations?)[^:]*:\s*(.+)",
                          txt, re.M)
            title = m.group(1).strip() if m else stem
            sections = re.findall(r"§[\d.]+(?:\s*-\s*§[\d.]+)?", spec)
            out[mid] = dict(slug=stem, dir=sub, topics=topics, title=title,
                            sections=", ".join(sections))
    return out


def table_concordance(tops, mods):
    rows = ["| `stats.md` topic | Title | Module (add `statsPy/`+`.py` or `statsR/`+`.R`) |",
            "|---|---|---|"]
    by_topic = {}
    for mid, m in mods.items():
        if m["dir"] in ("foundations", "exercises"):
            continue
        for t in m["topics"]:
            by_topic.setdefault(t, []).append(mid)
    for t in sorted(tops):
        title, _ = tops[t]
        ids = sorted(by_topic.get(t, []))
        if not ids:
            cell = "-"
        else:
            cell = "<br>".join(f"`{mods[i]['dir']}/{mods[i]['slug']}`" for i in ids)
        rows.append(f"| {t} | {title} | {cell} |")
    return "\n".join(rows)


def table_reading_order(mods):
    rows = ["| Stage | Modules, in order | Exercise |", "|---|---|---|"]
    for stage, ids, ex in READING_ORDER:
        cells = []
        for i in ids:
            m = mods.get(i)
            cells.append(f"`{i}`" if m else f"`{i}`")
        rows.append(f"| {stage} | {' '.join(cells)} | {ex} |")
    return "\n".join(rows)


def table_track(mods, sub, header):
    rows = [header, "|---|---|---|"]
    for mid in sorted(mods, key=lambda k: (mods[k]["dir"] != sub, k)):
        m = mods[mid]
        if m["dir"] != sub:
            continue
        if sub == "foundations":
            tcell = m["sections"] or "Part 0"
        elif sub == "exercises":
            tcell = ", ".join(str(t) for t in m["topics"])
        else:
            tcell = ", ".join(str(t) for t in m["topics"]) or "-"
        rows.append(f"| `{mid}` | `{m['slug']}` | {tcell} |")
    return "\n".join(rows)


def splice(path, name, body):
    p = os.path.join(ROOT, path)
    src = open(p).read()
    b, e = f"<!-- BEGIN crossref:{name} -->", f"<!-- END crossref:{name} -->"
    if b not in src:
        return False
    pat = re.compile(re.escape(b) + r".*?" + re.escape(e), re.S)
    new = pat.sub(f"{b}\n{body}\n{e}", src)
    if new != src:
        open(p, "w").write(new)
        return True
    return False


def rewrite_topic_module_lines(tops, mods):
    """Make every topic's **Module:** line name the real file(s), with the
    directory, so a reader can open it without consulting a table."""
    by_topic = {}
    for mid, m in mods.items():
        if m["dir"] in ("foundations", "exercises"):
            continue
        for t in m["topics"]:
            by_topic.setdefault(t, []).append(mid)
    p = os.path.join(ROOT, "stats.md")
    src = open(p).read()
    cur = {"t": None}

    def line_sub(m):
        t = cur["t"]
        ids = sorted(by_topic.get(t, []))
        if not ids:
            return m.group(0)
        return "**Module:** " + ", ".join(
            f"`{mods[i]['dir']}/{mods[i]['slug']}`" for i in ids)

    out = []
    for line in src.split("\n"):
        mt = re.match(r"^## Topic (\d+):", line)
        if mt:
            cur["t"] = int(mt.group(1))
        if line.startswith("**Module:**") and cur["t"] is not None:
            line = line_sub(re.match(r"(.*)", line))
        out.append(line)
    new = "\n".join(out)
    if new != src:
        open(p, "w").write(new)
        return True
    return False


def main():
    tops, mods = topics_from_stats_md(), modules()
    blocks = {
        "concordance": table_concordance(tops, mods),
        "reading-order": table_reading_order(mods),
        "foundations": table_track(mods, "foundations",
                                  "| # | Module | Covers |"),
        "exercises": table_track(mods, "exercises",
                                 "| # | Exercise | Topics combined |"),
    }
    touched = []
    if rewrite_topic_module_lines(tops, mods):
        touched.append("stats.md:**Module:** lines")
    for path in ("stats.md", "README.md"):
        for name, body in blocks.items():
            if splice(path, name, body):
                touched.append(f"{path}:{name}")
    print(f"crossref: {len(tops)} topics, {len(mods)} modules")
    print("  updated: " + (", ".join(touched) if touched else "nothing (in sync)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
