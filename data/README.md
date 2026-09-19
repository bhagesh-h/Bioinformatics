# Using real public data

Every module in this course simulates its own data. That is a deliberate
choice, not a shortcut, and it is worth understanding before you swap in a
real dataset.

**Why the modules simulate.** The whole method of this course is to do the
wrong thing, *measure how wrong it is*, then do it properly and measure that
too. Measuring error requires knowing the truth - the real effect size, which
genes are truly differential, which patients truly benefit. Real data never
tells you that. A simulated cohort does, so statements like "the naive
estimate had the wrong sign" or "the realised false discovery proportion was
0.14 against a nominal 0.05" are *checked*, not asserted. Simulation also
keeps `data/raw/` empty, so nothing here breaks when a download moves or a
server goes offline.

**Why you should still use real data.** Simulated data is clean in ways real
data is not: real count matrices have stranger outliers, real batches are
messier, and real metadata is incomplete in informative ways. Once a module's
lesson has landed, re-running it on the real analogue is the best next step.

This file lists, for each applied module, the canonical public dataset that
the module's design is modelled on, and how to get it.


## Getting the data

Almost all of these ship as Bioconductor *experiment data* packages, which is
by far the easiest route: no manual download, no file formats to parse, and
the sample metadata arrives already attached to the assay.

They are **not** installed in `learn-stats-r:1.0`, because doing so would make
the image large and require network access at build time. To use them, add the
ones you want to `env/r-packages.R` and rebuild, or install into a running
container:

```r
# inside the R container
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("airway")
```

```bash
docker run --rm -it -v "$PWD":/work -w /work learn-stats-r:1.0 R
```

> Note that installing into a running container is temporary - it is lost when
> the container exits. For anything you will use repeatedly, add the package
> to `env/r-packages.R` and rebuild the image, so the environment stays
> reproducible. That is the whole point of `env/`.


## Module-by-module

### 30: Bulk RNA-seq differential expression

**`airway`**, GEO [GSE52778](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52778),
Himes *et al.* 2014, *PLoS ONE* 9(6):e99625

Four primary human airway smooth muscle cell lines, each with an untreated and
a dexamethasone-treated sample. This is exactly the structure module 30
simulates: **n = 4 per arm, paired by cell line**, which is why the module
insists that dispersion cannot be estimated per gene without shrinkage.

```r
library(airway); data(airway)
se <- airway
se$dex <- relevel(se$dex, "untrt")   # set the reference level EXPLICITLY (Module 30)
counts <- assay(se); design <- colData(se)
```

Because samples are paired by cell line, the correct design includes the cell
line: `~ cell + dex`. Compare the result with and without it - that is the
paired-design lesson from Topic 12 on real data.

### 31: Single-cell pseudobulk

**`muscData::Kang18_8vs8()`**, GEO
[GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583),
Kang *et al.* 2018, *Nature Biotechnology*

10x PBMCs from **8 lupus patients before and after IFN-beta stimulation**. The
8-vs-8 patient structure with thousands of cells per patient is precisely the
setting of module 31: the cells are not replicates, the patients are.

```r
library(muscData); sce <- Kang18_8vs8()
# `ind` = patient, `stim` = condition, `cell` = annotated cell type
```

Aggregate to pseudobulk per (patient x cell type) and compare against a
per-cell test. The false-positive inflation you measured in the simulation is
visible here as implausibly small p-values.

### 32: Cytometry differential abundance

**`HDCytoData::Bodenmiller_BCR_XL_flowSet()`**, Bodenmiller *et al.* 2012,
packaged by Weber & Soneson 2019, *F1000Research* 8:1459

16 paired samples (**8 donors x unstimulated / BCR-XL stimulated**), 172,791
cells, 24 markers. Paired design, compositional cell-type proportions - both
of the traps module 32 is about.

### 33: Methylation

**`minfiData`** for a small worked 450k example; **TCGA 450k/EPIC** via
`curatedTCGAData` for a real cohort.

The module's central point - that beta values are heteroscedastic and M-values
are the right scale for testing - is easiest to see on real data by plotting
the mean-variance relationship of both scales.

### 34: GWAS

**1000 Genomes** ([internationalgenome.org](https://www.internationalgenome.org/))
for genotypes and genuine population structure; `snpStats` ships small example
data for mechanics.

Population structure is the whole lesson of module 34, and 1000 Genomes has
real structure rather than simulated structure - compute principal components
on the genotype matrix and watch them recover continental ancestry.

### 35: Proteomics and missing values

**PRIDE** ([ebi.ac.uk/pride](https://www.ebi.ac.uk/pride/)) for raw
submissions; **CPTAC** for cancer proteogenomics; `msdata` for small examples.

Plot detection rate against mean abundance. The monotone relationship you see
is what makes the missingness MNAR, and it is why module 35 warns against
imputing as though it were MCAR.

### 36: Microbiome

**`curatedMetagenomicData`**, Pasolli *et al.*, 5,716 uniformly processed
samples across 34 diseases and 28 countries, MetaPhlAn/HUMAnN profiles with
curated per-participant metadata. `phyloseq::GlobalPatterns` for a quick
example.

```r
library(curatedMetagenomicData)
se <- curatedMetagenomicData("ZellerG_2014.relative_abundance", dryrun = FALSE)
```

Relative abundances are compositional by construction - the closure problem
module 36 is built around.

### 37: Spatial omics

**`STexampleData::Visium_humanDLPFC()`**, Maynard, Collado-Torres *et al.*
2021, *Nature Neuroscience*; full dataset via `spatialLIBD`

The packaged sample is 151673, one of **12 samples from 3 donors**. That ratio
is module 37's entire lesson: thousands of spots, but three independent
biological units.

### 38: Survival with biomarkers

**TCGA** via `curatedTCGAData` or `TCGAbiolinks`; `survival::lung` (ships with
R, no download) for mechanics.

```r
library(survival)
fit <- survfit(Surv(time, status) ~ sex, data = lung)
```

On TCGA, try the module's optimal-cut-point experiment on genes you have no
prior reason to believe in. The rate at which "significant" cut-points appear
is the point.

### 39: Gene set enrichment

**MSigDB** ([gsea-msigdb.org](https://www.gsea-msigdb.org/gsea/msigdb)) via
`msigdbr`; **GO** via `org.Hs.eg.db` and `GO.db`.

Use the Hallmark collection first: it is curated to reduce the redundancy that
makes GO output so hard to interpret (module 39, section 8).

### 40: Networks and multi-omics

**TCGA multi-assay** via `curatedTCGAData`, which returns a
`MultiAssayExperiment` - the structure module 40 simulates. The **CLL
dataset** from the MOFA paper (Argelaguet *et al.* 2018, *Molecular Systems
Biology*) is the canonical multi-omics factor-analysis example.


## A caution

When you move to real data you lose the ground truth, and with it the ability
to check whether a method worked. Keep both: run the real analysis, then run
the module's simulation with parameters matched to what you measured in the
real data (sample size, dispersion, correlation, missingness rate). The
simulation tells you what your real analysis can and cannot detect.

That pairing - real data for the answer, matched simulation for the error
rate - is the most useful habit in this entire course.
