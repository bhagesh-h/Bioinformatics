#!/usr/bin/env python3
"""Download the real public datasets that data/README.md maps to each module.

Every module in this course simulates its own data, so nothing here is needed
to run the curriculum. This script is for the step after a lesson has landed:
re-running the analysis on the dataset the module was modelled on.

Nothing is installed on the host. Each fetch runs inside the project's R
container, which already has BiocManager available, and the results are written
to data/raw/<key>/ in portable formats:

    counts.csv.gz     the assay matrix, features x samples
    coldata.csv       sample metadata
    rowdata.csv       feature metadata, where the dataset has any
    object.rds        the original Bioconductor object, for R users
    PROVENANCE.txt    what was downloaded, from where, and when

Usage
    python3 tools/fetch_data.py --list
    python3 tools/fetch_data.py --get airway
    python3 tools/fetch_data.py --get airway kang18 bodenmiller
    python3 tools/fetch_data.py --get airway --force
    python3 tools/fetch_data.py --get tcga --dry-run

The first fetch of a given dataset installs its Bioconductor package inside a
throwaway container, so it is slow. Pass --keep-image to build a cached image
instead if you plan to fetch several.
"""
import argparse
import datetime
import os
import shlex
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
R_IMAGE = os.environ.get("R_IMAGE", "learn-stats-r:1.0")

# Each entry carries the R that loads the dataset and leaves three objects
# behind: `mat` (assay), `cd` (colData), and optionally `rd` (rowData).
DATASETS = {
    "airway": dict(
        module="30 bulk RNA-seq",
        pkg="airway",
        size="~5 MB",
        cite="Himes et al. 2014, PLoS ONE 9(6):e99625. GEO GSE52778",
        note="4 cell lines, untreated vs dexamethasone. Paired by cell line, "
             "so the correct design is ~ cell + dex.",
        r="""
            library(airway); data(airway)
            se <- airway
            mat <- assay(se); cd <- as.data.frame(colData(se))
            rd <- as.data.frame(rowData(se)); obj <- se
        """),
    "kang18": dict(
        module="31 single-cell pseudobulk",
        pkg="muscData",
        size="~150 MB",
        cite="Kang et al. 2018, Nature Biotechnology. GEO GSE96583",
        note="8 lupus patients before and after IFN-beta. Aggregate to "
             "pseudobulk per (patient x cell type) before testing.",
        r="""
            library(muscData); library(SingleCellExperiment)
            sce <- Kang18_8vs8()
            cd <- as.data.frame(colData(sce))
            # Pseudobulk immediately: the per-cell matrix is large and the
            # module's whole point is that the sample is the unit.
            grp <- paste(cd$ind, cd$cell, sep = "|")
            keep <- !is.na(cd$cell) & !is.na(cd$ind)
            m <- counts(sce)[, keep]; g <- grp[keep]
            mat <- vapply(split(seq_len(ncol(m)), g),
                          function(i) Matrix::rowSums(m[, i, drop = FALSE]),
                          numeric(nrow(m)))
            cd <- unique(data.frame(sample = g[keep[keep]],
                                    ind = cd$ind[keep], stim = cd$stim[keep],
                                    cell = cd$cell[keep]))
            cd <- cd[match(colnames(mat), paste(cd$ind, cd$cell, sep = "|")), ]
            rd <- NULL; obj <- NULL
        """),
    "bodenmiller": dict(
        module="32 cytometry",
        pkg="HDCytoData",
        size="~60 MB",
        cite="Bodenmiller et al. 2012; packaged by Weber & Soneson 2019, "
             "F1000Research 8:1459",
        note="16 paired samples, 8 donors x (unstimulated, BCR-XL). "
             "172,791 cells, 24 markers. The unit is the donor.",
        r="""
            library(HDCytoData)
            se <- Bodenmiller_BCR_XL_SE()
            mat <- t(assay(se)); cd <- as.data.frame(rowData(se))
            rd <- as.data.frame(colData(se)); obj <- se
        """),
    "minfi": dict(
        module="33 methylation",
        pkg="minfiData",
        size="~70 MB",
        cite="Bioconductor minfiData, 450k example",
        note="Small 450k example. Plot the mean-variance relationship of beta "
             "against M values to see why M is the testing scale.",
        r="""
            library(minfi); library(minfiData)
            gr <- preprocessRaw(RGsetEx)
            mat <- getBeta(gr); cd <- as.data.frame(pData(RGsetEx))
            rd <- NULL; obj <- NULL
        """),
    "snpstats": dict(
        module="34 GWAS",
        pkg="snpStats",
        size="~3 MB",
        cite="Bioconductor snpStats example data",
        note="Small genotype example for mechanics. For genuine population "
             "structure use 1000 Genomes: https://www.internationalgenome.org/",
        r="""
            library(snpStats); data(for.exercise)
            mat <- t(as(snps.10, "numeric")); cd <- as.data.frame(subject.support)
            rd <- as.data.frame(snp.support); obj <- NULL
        """),
    "msdata": dict(
        module="35 proteomics",
        pkg="msdata",
        size="~20 MB",
        cite="Bioconductor msdata",
        note="Raw MS example files. For a quantitative matrix with real "
             "missingness use PRIDE: https://www.ebi.ac.uk/pride/",
        r="""
            library(msdata)
            f <- list.files(system.file("microtofq", package = "msdata"),
                            full.names = TRUE)
            mat <- matrix(nrow = 0, ncol = 0)
            cd <- data.frame(file = basename(f), path = f)
            rd <- NULL; obj <- NULL
        """),
    "metagenomic": dict(
        module="36 microbiome",
        pkg="curatedMetagenomicData",
        size="~300 MB",
        cite="Pasolli et al., curatedMetagenomicData. MetaPhlAn profiles",
        note="Relative abundances are compositional by construction, which is "
             "the whole subject of module 36.",
        r="""
            library(curatedMetagenomicData)
            se <- curatedMetagenomicData("ZellerG_2014.relative_abundance",
                                         dryrun = FALSE, counts = FALSE)[[1]]
            mat <- assay(se); cd <- as.data.frame(colData(se))
            rd <- NULL; obj <- se
        """),
    "visium": dict(
        module="37 spatial",
        pkg="STexampleData",
        size="~120 MB",
        cite="Maynard, Collado-Torres et al. 2021, Nature Neuroscience",
        note="Sample 151673 of 12, from 3 donors. Thousands of spots, three "
             "independent biological units.",
        r="""
            library(STexampleData); library(SpatialExperiment)
            spe <- Visium_humanDLPFC()
            mat <- counts(spe)
            cd <- cbind(as.data.frame(colData(spe)), as.data.frame(spatialCoords(spe)))
            rd <- as.data.frame(rowData(spe)); obj <- spe
        """),
    "lung": dict(
        module="38 survival",
        pkg=None,                       # ships with R
        size="< 1 MB",
        cite="Loprinzi et al. 1994; survival::lung",
        note="No download required. For a molecular cohort use "
             "curatedTCGAData (key: tcga).",
        r="""
            library(survival); data(lung)
            mat <- matrix(nrow = 0, ncol = 0)
            cd <- lung; rd <- NULL; obj <- NULL
        """),
    "msigdb": dict(
        module="39 gene set enrichment",
        pkg="msigdbr",
        size="~30 MB",
        cite="Liberzon et al., MSigDB; msigdbr R package",
        note="Start with the Hallmark collection, which is curated to reduce "
             "the redundancy that makes GO output hard to read.",
        r="""
            library(msigdbr)
            h <- msigdbr(species = "Homo sapiens", collection = "H")
            mat <- matrix(nrow = 0, ncol = 0)
            cd <- as.data.frame(h[, c("gs_name", "gene_symbol")])
            rd <- NULL; obj <- NULL
        """),
    "tcga": dict(
        module="40 multi-omics, 38 survival",
        pkg="curatedTCGAData",
        size="~1 GB, varies by assay",
        cite="curatedTCGAData, Ramos et al.",
        note="Returns a MultiAssayExperiment, the structure module 40 "
             "simulates. Edit the disease code in the script for other cohorts.",
        r="""
            library(curatedTCGAData); library(MultiAssayExperiment)
            mae <- curatedTCGAData("BRCA", c("RNASeq2GeneNorm"),
                                   version = "2.0.1", dry.run = FALSE)
            se <- mae[[1]]
            mat <- assay(se); cd <- as.data.frame(colData(mae))
            rd <- NULL; obj <- mae
        """),
}

R_TEMPLATE = r"""
pkg <- {pkg}
suppressWarnings(suppressMessages({{
  if (!is.null(pkg) && !requireNamespace(pkg, quietly = TRUE)) {{
    if (!requireNamespace("BiocManager", quietly = TRUE))
      install.packages("BiocManager", repos = "https://cloud.r-project.org")
    message("installing ", pkg, " ...")
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  }}
}}))
outdir <- "{outdir}"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
mat <- NULL; cd <- NULL; rd <- NULL; obj <- NULL
suppressWarnings(suppressMessages({{
{body}
}}))
if (!is.null(mat) && length(mat) > 0) {{
  gz <- gzfile(file.path(outdir, "counts.csv.gz"), "w")
  write.csv(as.matrix(mat), gz); close(gz)
  cat(sprintf("  counts.csv.gz   %d features x %d samples\n", nrow(mat), ncol(mat)))
}}
if (!is.null(cd)) {{
  write.csv(cd, file.path(outdir, "coldata.csv"), row.names = TRUE)
  cat(sprintf("  coldata.csv     %d rows x %d columns\n", nrow(cd), ncol(cd)))
}}
if (!is.null(rd)) {{
  write.csv(rd, file.path(outdir, "rowdata.csv"), row.names = TRUE)
  cat(sprintf("  rowdata.csv     %d rows x %d columns\n", nrow(rd), ncol(rd)))
}}
if (!is.null(obj)) {{
  saveRDS(obj, file.path(outdir, "object.rds"))
  cat("  object.rds      original Bioconductor object\n")
}}
cat("done\n")
"""


def human(path):
    total = sum(os.path.getsize(os.path.join(d, f))
                for d, _, fs in os.walk(path) for f in fs)
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024 or unit == "GB":
            return f"{total:.0f} {unit}"
        total /= 1024


def do_list():
    print(f"{'key':<14}{'module':<28}{'size':<26}dataset")
    print("-" * 78)
    for k, d in DATASETS.items():
        here = os.path.join(RAW, k)
        mark = f"  [have: {human(here)}]" if os.path.isdir(here) else ""
        pkg = d["pkg"] or "base R"
        print(f"{k:<14}{d['module']:<28}{d['size']:<26}{pkg}{mark}")
    print("\nDetails are in data/README.md. Fetch with --get <key> [<key> ...].")


def fetch(key, force=False, dry=False):
    d = DATASETS[key]
    outdir = os.path.join(RAW, key)
    if os.path.isdir(outdir) and os.listdir(outdir) and not force:
        print(f"[{key}] already present at data/raw/{key} ({human(outdir)}). "
              f"Use --force to refetch.")
        return True
    body = textwrap.dedent(d["r"]).strip()
    pkg = f'"{d["pkg"]}"' if d["pkg"] else "NULL"
    script = R_TEMPLATE.format(pkg=pkg, outdir=f"/work/data/raw/{key}", body=body)
    cmd = ["docker", "run", "--rm",
           "--user", f"{os.getuid()}:{os.getgid()}",
           "-v", f"{ROOT}:/work", "-w", "/work",
           R_IMAGE, "Rscript", "-e", script]
    print(f"\n[{key}] {d['module']}  ({d['size']})")
    print(f"  source: {d['cite']}")
    if dry:
        print("  dry run, command would be:")
        print("   ", " ".join(shlex.quote(c) for c in cmd[:-1]), "<R script>")
        return True
    os.makedirs(outdir, exist_ok=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  FAILED. The package may need a Bioconductor release matching "
              f"the image's R version.")
        return False
    with open(os.path.join(outdir, "PROVENANCE.txt"), "w") as fh:
        fh.write(f"dataset : {key}\nmodule  : {d['module']}\n"
                 f"package : {d['pkg'] or 'base R (no download)'}\nsource  : {d['cite']}\n"
                 f"fetched : {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                 f"image   : {R_IMAGE}\n\n{textwrap.fill(d['note'], 76)}\n")
    print(f"  written to data/raw/{key} ({human(outdir)})")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Fetch the public datasets listed in data/README.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show available datasets")
    ap.add_argument("--get", nargs="+", metavar="KEY", help="datasets to fetch")
    ap.add_argument("--all", action="store_true", help="fetch everything (large)")
    ap.add_argument("--force", action="store_true", help="refetch if present")
    ap.add_argument("--dry-run", action="store_true", help="show, do not run")
    a = ap.parse_args()

    if a.list or not (a.get or a.all):
        do_list()
        return 0
    keys = list(DATASETS) if a.all else a.get
    bad = [k for k in keys if k not in DATASETS]
    if bad:
        print(f"unknown dataset(s): {', '.join(bad)}\n")
        do_list()
        return 2
    ok = all(fetch(k, a.force, a.dry_run) for k in keys)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
