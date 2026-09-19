# =============================================================================
# R teaching/analysis image for "Statistics for Bioinformatics".
#
# Build (from the repository root):
#     docker build -f docker/Dockerfile.r -t learn-stats-r:1.0 .
#
# Run a script:
#     docker run --rm -v "$PWD":/work -w /work learn-stats-r:1.0 \
#         Rscript statsR/core/01_study_design_and_estimands.R
#
# rocker/r-ver pins both R itself and a dated CRAN snapshot, so the image is
# reproducible: rebuilding it next year still gives you R 4.5.2 and the same
# package versions. It already contains every "recommended" R package
# (MASS, nlme, mgcv, survival, cluster, boot, Matrix, splines, lattice).
# =============================================================================
FROM rocker/r-ver:4.5.2

# pandoc is required by rmarkdown::render() to turn .Rmd into HTML.
RUN apt-get update && \
    apt-get install -y --no-install-recommends pandoc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY env/r-packages.R /tmp/r-packages.R
RUN Rscript /tmp/r-packages.R

CMD ["R", "--version"]
