# [Statistics for Bioinformatics](https://bhagesh-h.github.io/Bioinformatics/): a runnable curriculum

A self-contained course in statistics for biological data, in **both R and
Python**, where every claim is demonstrated by code you can run.

The material is built around one idea: *most statistical errors in
bioinformatics are not errors of calculation, they are errors of judgement* -
about what the independent unit is, what the estimand is, and what was decided
after looking at the data. Every module therefore does the wrong thing first,
measures how wrong it is, then does it properly and measures that too.

Nothing is installed on your machine. Everything runs in Docker.

`docs/index.html` is a single page showing every module and how they depend on
each other. Open it locally, or publish it with GitHub Pages
([section 14](#14-the-interactive-diagram)).


## Table of contents

1. [What is in here](#1-what-is-in-here)
2. [Requirements](#2-requirements)
3. [Quick start](#3-quick-start)
4. [Directory structure](#4-directory-structure)
5. [The four kinds of file](#5-the-four-kinds-of-file)
6. [How to learn from this](#6-how-to-learn-from-this)
7. [How to solve the problems](#7-how-to-solve-the-problems)
8. [Module map](#8-module-map)
9. [Running the tests](#9-running-the-tests)
10. [Building the notebooks](#10-building-the-notebooks)
11. [Reproducing the environment](#11-reproducing-the-environment)
12. [Conventions used throughout](#12-conventions-used-throughout)
13. [Using real public data](#13-using-real-public-data)
14. [The interactive diagram](#14-the-interactive-diagram)
15. [Troubleshooting](#15-troubleshooting)


## 1. What is in here

| Component | Count | What it is |
|---|---|---|
| `stats.md` | Part 0 + 35 topics, 305 numbered equations | The mathematical companion. **Part 0 starts from zero and assumes no maths at all**; Topics 1-35 are the full treatment |
| `statsPy/foundations/`, `statsR/foundations/` | 6 modules each | Statistics from scratch for absolute beginners: variables, probability, distributions, standard errors, p-values, causal roles |
| `statsPy/core/`, `statsR/core/` | 25 modules each | Statistical foundations, from study design to reproducibility |
| `statsPy/bioinformatics/`, `statsR/bioinformatics/` | 11 modules each | Applied analyses: RNA-seq, single-cell, methylation, GWAS, proteomics, microbiome, spatial, survival, enrichment, networks |
| `statsPy/exercises/`, `statsR/exercises/` | 4 exercises each | Integrative problem sets combining several topics, with full worked solutions |
| `statsPy/notebooks/`, `statsR/notebooks/` | 46 each | Generated `.ipynb` / `.Rmd` views of every script |
| `tools/` | 7 scripts | Test runners, solution checker, notebook builder, site builder, fast doc checks |
| `docs/` | 1 page | A single-page map of all 46 modules and their dependencies, for GitHub Pages |
| `env/`, `docker/` | - | Pinned environment manifests and Dockerfiles |

The R and Python tracks are **parallel but not identical**. They cover the same
statistics, and where the two languages differ in an instructive way the
modules say so - for example `nlme::lme` reports denominator degrees of
freedom that `statsmodels.mixedlm` does not, which changes the answer when
there are few clusters (`E1`).


## 2. Requirements

* **Docker.** That is the entire list.
* About 3 GB of disk for the two images (Python 1.2 GB, R 1.8 GB).
* A terminal. Everything below is run from the repository root.

You do **not** need R, Python, conda, or any package installed locally. If you
already have them, this material does not touch them.


## 3. Quick start

```bash
# 1. Build the two images (once, ~10 minutes)
docker build -t learn-stats-py:1.0 -f docker/Dockerfile.python .
docker build -t learn-stats-r:1.0  -f docker/Dockerfile.r      .

# 2. Run your first module
#    New to statistics? Start with the foundations track:
docker run --rm -v "$PWD":/work -w /work learn-stats-py:1.0 \
    python statsPy/foundations/F1_variables_populations_samples.py

#    Already comfortable with standard errors and p-values? Start at Topic 1:
docker run --rm -v "$PWD":/work -w /work learn-stats-py:1.0 \
    python statsPy/core/01_study_design_and_estimands.py

# 3. Or the R version of either
docker run --rm -v "$PWD":/work -w /work learn-stats-r:1.0 \
    Rscript statsR/foundations/F1_variables_populations_samples.R
```

Each module prints a narrated analysis to the terminal and writes any figures
under `results/<module_name>/`. To put figures somewhere else, set `STATS_OUT`:

```bash
docker run --rm -e STATS_OUT=/work/myresults -v "$PWD":/work -w /work \
    learn-stats-py:1.0 python statsPy/core/17_pca.py
```

A convenience wrapper, so you do not have to type the `docker run` line each
time:

```bash
# put this in your shell profile if you like
runpy() { docker run --rm -v "$PWD":/work -w /work learn-stats-py:1.0 python "$@"; }
runr()  { docker run --rm -v "$PWD":/work -w /work learn-stats-r:1.0 Rscript "$@"; }

runpy statsPy/core/08_multiple_testing.py
runr  statsR/core/08_multiple_testing.R
```


## 4. Directory structure

```
.
+-- README.md                      <- you are here
+-- stats.md                       <- the mathematical companion (read alongside)
+-- CURRICULUM.md                  <- the decision-centred syllabus this grew from
|
+-- statsPy/                       -- PYTHON TRACK --------------------------
|   +-- foundations/               6 beginner modules       (F1 ... F6)
|   +-- core/                      25 statistics modules    (00 ... 24)
|   +-- bioinformatics/            11 applied modules       (30 ... 40)
|   +-- exercises/                 4 integrative problem sets (E1 ... E4)
|   \-- notebooks/                 generated .ipynb, mirroring the above
|       +-- foundations/  core/  bioinformatics/  exercises/
|
+-- statsR/                        -- R TRACK -------------------------------
|   +-- foundations/               6 beginner modules       (F1 ... F6)
|   +-- core/                      25 statistics modules    (00 ... 24)
|   +-- bioinformatics/            11 applied modules       (30 ... 40)
|   +-- exercises/                 4 integrative problem sets (E1 ... E4)
|   \-- notebooks/                 generated .Rmd, mirroring the above
|       +-- foundations/  core/  bioinformatics/  exercises/
|
+-- env/
|   +-- requirements.txt           Python dependencies, pinned by lower bound
|   \-- r-packages.R               R dependencies - documentation AND installer
|
+-- docker/
|   +-- Dockerfile.python          python:3.12-slim + env/requirements.txt
|   \-- Dockerfile.r               rocker/r-ver:4.5.2 (pinned CRAN snapshot)
|
+-- tools/
|   +-- run_python_tests.sh        run every Python module, report PASS/FAIL
|   +-- run_r_tests.sh             run every R module, report PASS/FAIL
|   +-- check_solutions.sh         run every module with its SOLUTIONS enabled
|   +-- uncomment_solutions.py     helper used by check_solutions.sh
|   +-- build_notebooks.sh         regenerate all notebooks from the scripts
|   +-- build_site.py              regenerate docs/index.html from the headers
|   \-- check_docs.py              fast consistency checks, no Docker needed
|
+-- docs/
|   +-- index.html                 the module map (GitHub Pages entry point)
|   \-- _template.html             its source, with the data injected
|
+-- data/
|   +-- README.md                  where to get the REAL public dataset for
|   |                              each applied module (see §13)
|   +-- raw/                       (empty - all data is simulated in-module)
|   \-- derived/
\-- results/                       figures, created on first run
```

### Naming

| Pattern | Meaning |
|---|---|
| `F1`-`F6` | foundations - statistics from zero, no prerequisites |
| `00`-`24` | core statistics, in teaching order - later modules assume earlier ones |
| `30`-`40` | applied bioinformatics, each mapping to one assay or question |
| `E1`-`E4` | integrative exercises that combine several modules |

The gap between 24 and 30 is deliberate: it leaves room for core modules
without renumbering the applied ones.


## 5. The four kinds of file

**1. `stats.md` - the mathematics.** Part 0 (statistics from zero, no maths
assumed) plus 35 topics, 305 numbered equations in total, a
notation table, three appendices (a distribution reference sheet, the
identities worth memorising, and a symbol-to-code dictionary showing the same
quantity in R and Python). Every script cross-references it by equation
number, e.g. `eq. (1.8)`. Read a topic, then run the matching module.

**2. The scripts - `.py` and `.R`.** These are the source of truth. They are
plain scripts: no notebook JSON, no hidden state, diffable and greppable. They
are what the tests execute. Every module has the same shape:

```
header comment      title, curriculum link, the question being answered
setup               imports, output directory, a `header()` helper
numbered sections   the statistical content, each printing its result
PROBLEMS            2-3 exercises with commented-out worked solutions
                    (foundations, applied modules and exercises - the
                     core modules 00-24 are worked demonstrations
                     throughout)
figure              written to results/<module>/
what to take away   the module in five or six lines
```

Across both languages that is **142 problems**, each with a full worked
solution and an explanation of why the obvious alternative is wrong.

**3. The notebooks - `.ipynb` and `.Rmd`.** Generated from the scripts by
`tools/build_notebooks.sh`; see [§10](#10-building-the-notebooks). They are a
*view*, never the source. **Do not edit a notebook**: your changes will be
overwritten the next time they are built. Edit the script and rebuild.

**4. The tools.** Shell scripts that run everything in Docker. See
[§9](#9-running-the-tests).


## 6. How to learn from this

### Start here if you have never studied statistics

You are the intended reader of the **foundations track**, and nothing in it
assumes any mathematics beyond arithmetic.

1. Read **Part 0 of `stats.md`**: it defines every term from scratch and ends
   with a plain-English glossary.
2. Run **`F1` -> `F6`** in your language of choice. Each one *simulates* an
   entire population so you can see the true answer, then shows what happens
   when you only get a sample of it.
3. Then join the main path below at Topic 1.

| Module | What it gives you |
|---|---|
| `F1` | units, populations, samples; why bias and noise are different problems |
| `F2` | probability, independence, and why $P(A\mid B) \neq P(B\mid A)$ |
| `F3` | the distributions, and the mechanism behind each one |
| `F4` | estimates, standard errors, SD vs SE, the central limit theorem |
| `F5` | confidence intervals and every kind of p-value |
| `F6` | the linear model, and confounders vs mediators vs colliders |

The foundations are not a simplified version of the main course. They are the
same ideas, verified by simulation, before the biology is layered on.

### The intended path

1. **Read the topic in `stats.md`.** Get the definitions and the equations.
2. **Run the module** and read its output alongside the script. The narration
   refers to numbers that were just printed; the point is to connect the
   equation to a number you watched appear.
3. **Break it.** Change a parameter - the sample size, the effect, the
   correlation - and re-run. Most modules are built so that one parameter
   controls whether the method works.
4. **Do the PROBLEMS** at the end of the module ([§7](#7-how-to-solve-the-problems)).
5. **After a block of modules, do the matching exercise** (`E1`-`E4`).

### Suggested order

| Stage | Modules | Then do |
|---|---|---|
| **Absolute basics** | `F1`-`F6` | - |
| Foundations | `00`-`09` | `E1` (design audit) |
| Models | `10`-`15` | - |
| High-dimensional | `16`-`19` | - |
| Inference and validation | `20`-`24` | `E3` (prediction audit), `E4` (causal) |
| Applied | `30`-`40` | `E2` (RNA-seq end to end) |

If you are here for one assay, read `01`, `08`, `13` and `19` first - those
four cover the mistakes that actually invalidate published analyses - then go
straight to your module in `bioinformatics/`.

### If you are short of time

Run these six, in this order. They contain most of what goes wrong in practice:

`01_study_design_and_estimands`, `08_multiple_testing`,
`19_batch_effects`, `21_prediction_and_validation`,
`22_causal_inference`, `30_bulk_rnaseq_differential_expression`


## 7. How to solve the problems

Every foundations module (`F1`-`F6`), applied module (`30`-`40`) and
exercise (`E1`-`E4`) ends with a
`PROBLEMS` section - 142 problems in total across the two languages. Each
problem has two blocks:

```python
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# result = ...
# print(...)
#
# # Why this is the right answer, and what it would have looked like if you
# # had done it the other way.
```

The R version is identical apart from the comment marker (`##`).

**The workflow:**

1. Write your attempt in the `YOUR CODE HERE` block and run the module.
2. When you are done - or genuinely stuck - uncomment the solution block and
   run it again. In most editors: select the block and press
   `Ctrl+/` (VS Code), or `Ctrl+Shift+C` (RStudio).
3. Read the prose at the end of the solution. **That is the actual answer.**
   The code shows *what* to compute; the comment explains *why*, and usually
   what the common wrong approach would have given you.

To check every shipped solution still runs:

```bash
bash tools/check_solutions.sh              # both languages
bash tools/check_solutions.sh python       # one language
bash tools/check_solutions.sh r exercises  # one language, one directory
```

This rewrites each module with its solutions enabled (into a temporary
directory - your files are not modified) and executes it. It is how the
solutions are kept honest: code that is never run rots.


## 8. Module map

### Foundations: `statsPy/foundations/`, `statsR/foundations/`

No prerequisites. Read alongside **Part 0** of `stats.md`.

| # | Module | Part 0 section | The idea |
|---|---|---|---|
| F1 | `variables_populations_samples` | §0.1-§0.3 | your data is evidence about the answer, not the answer |
| F2 | `probability_basics` | §0.4 | independence is the assumption biology breaks |
| F3 | `distributions` | §0.5 | each shape comes from a mechanism |
| F4 | `estimates_and_standard_errors` | §0.6 | SD describes the data; SE describes your knowledge |
| F5 | `confidence_intervals_and_pvalues` | §0.7-§0.8 | both are guarantees about a *procedure* |
| F6 | `connecting_variables` | §0.9 | adjusting for more variables is not safer |

### Core: `statsPy/core/`, `statsR/core/`

| # | Module | Topic in `stats.md` |
|---|---|---|
| 00 | `setup_and_environment` | - |
| 01 | `study_design_and_estimands` | 1 |
| 02 | `data_structures_and_scales` | 2 |
| 03 | `exploratory_data_analysis` | 3 |
| 04 | `probability_and_sampling` | 4 |
| 05 | `estimation_and_intervals` | 5 |
| 06 | `hypothesis_tests_and_pvalues` | 6 |
| 07 | `ttests_ranks_permutation` | 7 |
| 08 | `multiple_testing` | 8 |
| 09 | `power_and_sample_size` | 9 |
| 10 | `correlation_and_dependence` | 10 |
| 11 | `linear_models` | 11 |
| 12 | `anova_and_contrasts` | 12 |
| 13 | `generalized_linear_models` | 13 |
| 14 | `mixed_models` | 14 |
| 15 | `missing_data_and_censoring` | 15 |
| 16 | `matrix_algebra` | 16 |
| 17 | `pca` | 17 |
| 18 | `distances_and_clustering` | 18 |
| 19 | `batch_effects` | 19 |
| 20 | `survival_and_longitudinal` | 28 |
| 21 | `prediction_and_validation` | 31 |
| 22 | `causal_inference` | 32 |
| 23 | `bayesian_and_shrinkage` | 33 |
| 24 | `diagnostics_and_reproducibility` | 34, 35 |

### Applied: `statsPy/bioinformatics/`, `statsR/bioinformatics/`

| # | Module | Topic | The central trap |
|---|---|---|---|
| 30 | `bulk_rnaseq_differential_expression` | 20 | dispersion cannot be estimated per gene at n = 3 |
| 31 | `single_cell_pseudobulk` | 21 | cells are not replicates |
| 32 | `cytometry_differential_abundance` | 22 | cell proportions are compositional |
| 33 | `methylation_epigenomics` | 23 | beta values are heteroscedastic; use M-values |
| 34 | `gwas_association` | 24 | population structure is confounding |
| 35 | `proteomics_missing_values` | 25 | missingness depends on abundance (MNAR) |
| 36 | `microbiome_compositional` | 26 | you only ever observe proportions |
| 37 | `spatial_omics` | 27 | spots are not patients; space induces correlation |
| 38 | `survival_biomarkers` | 28 | events, not patients; never dichotomise on a data-chosen cut-point |
| 39 | `gene_set_enrichment` | 29 | genes in a pathway are correlated |
| 40 | `networks_and_multiomics` | 30 | correlation networks find modules in pure noise |

### Exercises: `statsPy/exercises/`, `statsR/exercises/`

| # | Exercise | Combines | The brief |
|---|---|---|---|
| E1 | `design_audit` | 1, 7, 9, 14, 19 | A collaborator's *p* = 2e-12 from 120 cells. Find both errors. |
| E2 | `rnaseq_end_to_end` | 8, 13, 20, 29 | Build a DE pipeline and check every step, including the negative control. |
| E3 | `prediction_audit` | 18, 21, 31 | A signature with a large AUC and four leaks. Quantify each. |
| E4 | `causal_question` | 11, 15, 32, 34 | Does the drug work? One of the covariates will make your answer worse. |


## 9. Running the tests

```bash
bash tools/run_python_tests.sh                 # all 46 Python modules
bash tools/run_python_tests.sh foundations
bash tools/run_python_tests.sh core            # only statsPy/core
bash tools/run_python_tests.sh bioinformatics
bash tools/run_python_tests.sh exercises

bash tools/run_r_tests.sh                      # all 46 R modules
bash tools/run_r_tests.sh core
```

Output is a table of `MODULE / STATUS / SECONDS` and a non-zero exit code if
anything failed; per-module logs are left in `/tmp/<module>.log`.

These runners execute each module **as a learner first sees it**, with the
solutions still commented out. To execute the solutions as well, use
`tools/check_solutions.sh` ([§7](#7-how-to-solve-the-problems)).

A full run of everything takes roughly 40-70 minutes, most of it Docker
start-up rather than computation.

For a fast check that needs no Docker and no dependencies, run:

```bash
python3 tools/check_docs.py
```

It verifies that every Python module compiles, that every heading link,
footnote and relative file link in the documentation resolves, and that
`docs/index.html` is still in sync with the module headers. This is what runs
in CI on every push; the Docker suites are too slow for that.


## 10. Building the notebooks

```bash
bash tools/build_notebooks.sh            # both languages, 92 notebooks
bash tools/build_notebooks.sh python     # .py -> .ipynb
bash tools/build_notebooks.sh r          # .R  -> .Rmd
```

Both tracks use a **single-source literate format**, so the script and the
notebook can never disagree:

* **Python** uses [jupytext](https://jupytext.readthedocs.io/) *percent*
  format. `# %%` starts a code cell, `# %% [markdown]` a markdown cell. The
  `.py` file is valid Python and a valid notebook at the same time.
* **R** uses `knitr::spin`. Lines beginning `#'` become markdown, lines
  beginning `#+` set chunk options, everything else is code.

To work in a notebook interactively:

```bash
# JupyterLab at http://localhost:8888
docker run --rm -p 8888:8888 -v "$PWD":/work -w /work learn-stats-py:1.0 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# Render an R notebook to HTML
docker run --rm -v "$PWD":/work -w /work learn-stats-r:1.0 Rscript -e \
    "rmarkdown::render('statsR/notebooks/core/17_pca.Rmd')"
```

> **Edit the script, not the notebook.** Notebooks are regenerated and any
> edits made in them will be lost.


## 11. Reproducing the environment

Three files fully determine the environment; there is nothing else to install.

| File | Purpose |
|---|---|
| `env/requirements.txt` | Python packages, each with a lower bound and a comment saying why it is there |
| `env/r-packages.R` | R packages - this file is both the documentation and the installer, and **fails the build loudly** if anything is missing |
| `docker/Dockerfile.python`, `docker/Dockerfile.r` | the two images |

```bash
docker build -t learn-stats-py:1.0 -f docker/Dockerfile.python .
docker build -t learn-stats-r:1.0  -f docker/Dockerfile.r      .
```

Determinism:

* The R image is `rocker/r-ver:4.5.2`, which pins a **dated CRAN snapshot** -
  the same build produces the same package versions.
* The Python image sets `PYTHONHASHSEED=0` and `MPLBACKEND=Agg`, so figures
  render headlessly and iteration order is stable.
* Every module seeds its own generator explicitly. Reported numbers are
  reproducible run to run.

### If you would rather use conda

The Docker images are the supported path, but the Python side maps directly:

```bash
conda create -n statsbio python=3.12
conda activate statsbio
pip install -r env/requirements.txt
```

You are then responsible for your own environment, and exact numbers may
differ slightly with different library versions.


## 12. Conventions used throughout

**Every module is self-contained.** Data is simulated inside the module from a
stated generative model, so you always know the truth and can measure how far
an estimate is from it. `data/raw/` is deliberately empty: nothing here
depends on a download that might disappear. For the real public dataset behind
each applied module, see [`data/README.md`](data/README.md).

**Equations are numbered `(topic.index)`** and refer to `stats.md`. `eq. (1.8)`
is the design effect, in Topic 1.

**Wrong first, right second.** Modules show the naive analysis, measure the
damage, and then fix it. The numbers in the narration are the numbers the
script prints - if you change a seed, the text may no longer match, and that
is a feature: it means the prose was written against real output.

**The negative control is the point.** Many modules end by running the whole
pipeline on data with no signal. A method that cannot return "nothing" on
noise is not usable, no matter how good it looks on real data.

**One idea recurs more than any other.** The design effect

$$\mathrm{DE} = 1 + (m-1)\rho$$

appears in Topic 1 as pseudoreplication, in Topic 21 as cells within patients,
in Topic 27 as spots within tissue, and in Topic 29 as genes within a pathway
(`camera`'s variance inflation factor). It is the same equation every time.


## 13. Using real public data

Every module simulates its own data, so that the truth is known and the size
of each error can be *measured* rather than asserted. Once a module's lesson
has landed, the natural next step is to re-run it on the real dataset its
design was modelled on.

**[`data/README.md`](data/README.md)** maps each applied module to its
canonical public dataset - `airway`/GSE52778 for bulk RNA-seq, Kang 2018
(GSE96583) for single-cell, Bodenmiller BCR-XL for cytometry, the Visium DLPFC
samples for spatial, TCGA for survival and multi-omics - with accessions,
loading code, and what to look for in each.

Keep both. Real data gives you the answer; a simulation with parameters
matched to it tells you what that answer can and cannot support.

## 14. The interactive diagram

`docs/index.html` is a single page showing all 46 modules as a dependency
diagram: four columns for the four tracks, dashed edges for reading order,
solid edges for declared dependencies. Point at a module to trace everything it
needs and everything that needs it.

Open it directly:

```bash
xdg-open docs/index.html      # or: open docs/index.html
```

It is a single static file with no build step and no network calls apart from
the webfont, so it works from `file://`.

To publish it, go to **Settings, Pages** in the repository and set the source
to the `main` branch, `/docs` folder. GitHub then serves it at
`https://<user>.github.io/<repo>/`.

The page is generated from the module headers, exactly as the notebooks are, so
it cannot drift from the code. After adding or renaming a module:

```bash
python3 tools/build_site.py
```

That reads the title, curriculum link, description and `Core modules used`
line out of each `statsPy/**/*.py` header and rewrites `docs/index.html` from
`docs/_template.html`. It uses only the standard library.


## 15. Troubleshooting

**`docker: permission denied`**: your user is not in the `docker` group. Use
`sudo`, or add yourself: `sudo usermod -aG docker $USER` and log out and back
in.

**`Unable to find image 'learn-stats-py:1.0'`**: build the images first
([§3](#3-quick-start)). The tag must match exactly.

**Figures do not appear**: they are written to `results/<module>/`, not
displayed; the containers are headless. Set `STATS_OUT` to change the
location.

**A module writes nothing to `results/`**: check you mounted the repository
(`-v "$PWD":/work -w /work`) and that you are in the repository root.

**A test fails after I edited something**: read `/tmp/<module>.log`. The
runners keep the full output of every module.

**R cannot find a package**: rebuild `learn-stats-r:1.0`. `env/r-packages.R`
fails the build if a package is missing, so a successful build guarantees the
package set.

**A number in the prose does not match my output**: you changed a seed or a
parameter. The narration was written against the committed seeds.
