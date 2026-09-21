#!/usr/bin/env python3
"""Fast consistency checks that need nothing but the standard library.

Run locally or in CI. The slow, authoritative checks are the Docker suites in
tools/run_*_tests.sh and tools/check_solutions.sh; these catch the cheap
mistakes in seconds:

  1. every Python module compiles
  2. every in-document heading link resolves
  3. every footnote reference has a definition
  4. every relative file link points at a file that exists
  5. docs/index.html is in sync with the module headers
"""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = ["README.md", "stats.md", "CURRICULUM.md", "data/README.md"]
fail = []


def slug(heading):
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s", "-", s)


def check_python():
    bad = []
    for f in glob.glob(f"{ROOT}/statsPy/*/*.py") + glob.glob(f"{ROOT}/tools/*.py"):
        if "/notebooks/" in f:
            continue
        try:
            # compile() parses without writing a .pyc anywhere.
            compile(open(f, encoding="utf-8").read(), f, "exec")
        except SyntaxError as e:
            bad.append(f"{os.path.relpath(f, ROOT)}:{e.lineno}: {e.msg}")
    return bad


def check_markdown():
    bad = []
    for rel in DOCS:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            bad.append(f"{rel}: missing")
            continue
        text = open(path, encoding="utf-8").read()
        heads = {slug(m) for m in re.findall(r"^#{1,6}\s+(.*)$", text, re.M)}
        for link in re.findall(r"\]\(#([^)]+)\)", text):
            if link not in heads:
                bad.append(f"{rel}: heading link #{link} does not resolve")
        refs = set(re.findall(r"\[\^([A-Za-z0-9_-]+)\](?!:)", text))
        defs = set(re.findall(r"^\[\^([A-Za-z0-9_-]+)\]:", text, re.M))
        for r in sorted(refs - defs):
            bad.append(f"{rel}: footnote [^{r}] has no definition")
        base = os.path.dirname(path)
        for link in re.findall(r"\]\(([^)#][^)]*)\)", text):
            if link.startswith(("http", "mailto")):
                continue
            if not os.path.exists(os.path.join(base, link.split("#")[0])):
                bad.append(f"{rel}: file link {link} does not exist")
    return bad


def check_site():
    page = os.path.join(ROOT, "docs/index.html")
    if not os.path.exists(page):
        return ["docs/index.html: missing"]
    before = open(page, encoding="utf-8").read()
    subprocess.run([sys.executable, os.path.join(ROOT, "tools/build_site.py")],
                   check=True, capture_output=True)
    after = open(page, encoding="utf-8").read()
    if before != after:
        return ["docs/index.html is stale: run python3 tools/build_site.py"]
    return []


def check_crossref():
    """stats.md and README tables must match tools/build_crossref.py output."""
    paths = [os.path.join(ROOT, p) for p in ("stats.md", "README.md")]
    before = {p: open(p, encoding="utf-8").read() for p in paths}
    subprocess.run([sys.executable, os.path.join(ROOT, "tools/build_crossref.py")],
                   check=True, capture_output=True)
    stale = [os.path.basename(p) for p in paths
             if open(p, encoding="utf-8").read() != before[p]]
    if stale:
        return [f"{n}: cross-reference tables are stale, run "
                f"python3 tools/build_crossref.py" for n in stale]
    return []


def check_module_numbering():
    """A module's number must equal the stats.md topic it declares."""
    problems = []
    for sub in ("core", "bioinformatics"):
        d = os.path.join(ROOT, "statsPy", sub)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py"):
                continue
            stem = fn[:-3]
            mid = stem.split("_")[0]
            if mid in ("00",):                      # setup has no topic
                continue
            txt = open(os.path.join(d, fn), encoding="utf-8").read()
            m = re.search(r"Curriculum link:\*{0,2}\s*`?stats\.md`?\s*->\s*(.+)",
                          txt)
            if not m:
                problems.append(f"{sub}/{fn}: no Curriculum link")
                continue
            head = re.sub(r"\(\d+\.\d+\)", "", m.group(1).split("equation")[0])
            topics = re.findall(r"\b(\d+)\b", head)
            if str(int(mid.rstrip("ab"))) not in topics:
                problems.append(
                    f"{sub}/{fn}: numbered {mid} but declares Topic(s) "
                    f"{', '.join(topics) or '?'}")
            r = os.path.join(ROOT, "statsR", sub, stem + ".R")
            if not os.path.exists(r):
                problems.append(f"statsR/{sub}/{stem}.R: missing R counterpart")
    return problems


for name, fn in [("python compiles", check_python),
                 ("markdown links", check_markdown),
                 ("module numbering", check_module_numbering),
                 ("cross-reference tables", check_crossref),
                 ("site in sync", check_site)]:
    problems = fn()
    print(f"{'ok  ' if not problems else 'FAIL'}  {name}"
          f"{'' if not problems else f' ({len(problems)})'}")
    fail += problems

if fail:
    print()
    for f in fail[:25]:
        print("  " + f)
    sys.exit(1)
print("\nall checks passed")
