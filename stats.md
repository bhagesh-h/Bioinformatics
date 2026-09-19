# Statistics and Mathematics for Bioinformatics: A Formal Companion

> **What this document is.** The sibling file
> [`CURRICULUM.md`](CURRICULUM.md)
> tells you *which decision* to make. This document tells you *what the mathematics
> underneath that decision actually says*: the estimator, its distribution, the
> assumptions that make it valid, and the algebra that connects it to everything
> else you already know.
>
> **What this document is not.** It is not a package manual. Every equation here
> is implemented, from scratch where instructive, in the numbered scripts under
> [`statsPy/`](statsPy/) and [`statsR/`](statsR/). Read a section, then run the
> matching module and watch the identity hold numerically.


## Table of contents

- [How to use this document](#how-to-use-this-document)
- [Notation](#notation)
- [Module map](#module-map)
- [Part 0 - Starting from zero](#part-0-starting-from-zero) *(no maths assumed)*
  - [0.1 What statistics is actually for](#01-what-statistics-is-actually-for)
  - [0.2 Variables: the things you measure](#02-variables-the-things-you-measure)
  - [0.3 Populations and samples](#03-populations-and-samples)
  - [0.4 Probability from scratch](#04-probability-from-scratch)
  - [0.5 Distributions](#05-distributions)
  - [0.6 Estimates, standard errors, and why averaging works](#06-estimates-standard-errors-and-why-averaging-works)
  - [0.7 Confidence intervals](#07-confidence-intervals)
  - [0.8 p-values: what they are, the kinds, and how thresholds get chosen](#08-p-values-what-they-are-the-kinds-and-how-thresholds-get-chosen)
  - [0.9 How statistics connects variables](#09-how-statistics-connects-variables)
  - [0.10 A plain-English glossary](#010-a-plain-english-glossary)
  - [0.11 Where to go next](#011-where-to-go-next)
- [Part I - Foundations](#part-i-foundations)
  - [Topic 1 - Statistical thinking for biological experiments](#topic-1-statistical-thinking-for-biological-experiments)
  - [Topic 2 - Data structures and measurement scales](#topic-2-data-structures-and-measurement-scales)
  - [Topic 3 - Exploratory data analysis](#topic-3-exploratory-data-analysis)
  - [Topic 4 - Probability models and sampling distributions](#topic-4-probability-models-and-sampling-distributions)
- [Part II - Inference](#part-ii-inference)
  - [Topic 5 - Estimation, effect sizes, and confidence intervals](#topic-5-estimation-effect-sizes-and-confidence-intervals)
  - [Topic 6 - Hypothesis tests and p-values](#topic-6-hypothesis-tests-and-p-values)
  - [Topic 7 - t-tests, rank tests, and permutation tests](#topic-7-t-tests-rank-tests-and-permutation-tests)
  - [Topic 8 - Multiple testing and selective inference](#topic-8-multiple-testing-and-selective-inference)
  - [Topic 9 - Power, sample size, and design optimisation](#topic-9-power-sample-size-and-design-optimisation)
- [Part III - Association and models](#part-iii-association-and-models)
  - [Topic 10 - Correlation and dependence](#topic-10-correlation-and-dependence)
  - [Topic 11 - Linear regression as the core framework](#topic-11-linear-regression-as-the-core-framework)
  - [Topic 12 - ANOVA, factorial designs, and contrasts](#topic-12-anova-factorial-designs-and-contrasts)
  - [Topic 13 - Generalised linear models](#topic-13-generalised-linear-models)
  - [Topic 14 - Mixed, multilevel, and repeated-measures models](#topic-14-mixed-multilevel-and-repeated-measures-models)
  - [Topic 15 - Missing data, censoring, and measurement limits](#topic-15-missing-data-censoring-and-measurement-limits)
- [Part IV - Multivariate statistics](#part-iv-multivariate-statistics)
  - [Topic 16 - Matrix algebra for high-dimensional biology](#topic-16-matrix-algebra-for-high-dimensional-biology)
  - [Topic 17 - Principal component analysis](#topic-17-principal-component-analysis)
  - [Topic 18 - Distances, clustering, and embeddings](#topic-18-distances-clustering-and-embeddings)
  - [Topic 19 - Batch effects and unwanted variation](#topic-19-batch-effects-and-unwanted-variation)
- [Part V - Modality-specific statistics](#part-v-modality-specific-statistics)
  - [Topic 20 - Bulk RNA-seq differential expression](#topic-20-bulk-rna-seq-differential-expression)
  - [Topic 21 - Single-cell RNA-seq: replicated inference](#topic-21-single-cell-rna-seq-replicated-inference)
  - [Topic 22 - Flow, mass, and imaging cytometry](#topic-22-flow-mass-and-imaging-cytometry)
  - [Topic 23 - DNA methylation and epigenomics](#topic-23-dna-methylation-and-epigenomics)
  - [Topic 24 - Genotypes, GWAS, and statistical genetics](#topic-24-genotypes-gwas-and-statistical-genetics)
  - [Topic 25 - Proteomics and metabolomics](#topic-25-proteomics-and-metabolomics)
  - [Topic 26 - Microbiome and compositional data](#topic-26-microbiome-and-compositional-data)
  - [Topic 27 - Spatial transcriptomics and spatial omics](#topic-27-spatial-transcriptomics-and-spatial-omics)
  - [Topic 28 - Time-course, longitudinal, and survival data](#topic-28-time-course-longitudinal-and-survival-data)
- [Part VI - Gene sets and systems-level inference](#part-vi-gene-sets-and-systems-level-inference)
  - [Topic 29 - Functional enrichment and pathway statistics](#topic-29-functional-enrichment-and-pathway-statistics)
  - [Topic 30 - Networks and multivariate integration](#topic-30-networks-and-multivariate-integration)
- [Part VII - Prediction and causal reasoning](#part-vii-prediction-and-causal-reasoning)
  - [Topic 31 - Statistical learning and biomarker prediction](#topic-31-statistical-learning-and-biomarker-prediction)
  - [Topic 32 - Causal inference for observational bioinformatics](#topic-32-causal-inference-for-observational-bioinformatics)
  - [Topic 33 - Bayesian reasoning and hierarchical shrinkage](#topic-33-bayesian-reasoning-and-hierarchical-shrinkage)
- [Part VIII - Reliability and reproducibility](#part-viii-reliability-and-reproducibility)
  - [Topic 34 - Model diagnostics and sensitivity analysis](#topic-34-model-diagnostics-and-sensitivity-analysis)
  - [Topic 35 - Reproducible statistical workflows](#topic-35-reproducible-statistical-workflows)
- [Appendix A - Distribution reference sheet](#appendix-a-distribution-reference-sheet)
- [Appendix B - Identities worth memorising](#appendix-b-identities-worth-memorising)
- [Appendix C - Symbol-to-code dictionary](#appendix-c-symbol-to-code-dictionary)
- [References](#references)


## How to use this document

Each topic follows the same seven-part skeleton, so you can navigate by role
rather than by reading linearly:

| Heading | What it gives you |
|---|---|
| **Module** | The script/notebook that implements this topic in Python and R |
| **The question** | The scientific question the mathematics is built to answer |
| **Model** | The probability statement that defines the estimand |
| **Key equations** | Numbered, derived, and explained term by term |
| **Assumptions** | What must be true, and what breaks when it is not |
| **Diagnostics** | How to check the assumptions on real data |
| **Decision rules** | The practical "if...then" summary |

Equations are numbered `(topic.index)`. Equation `(11.4)` is the fourth
equation of Topic 11. The scripts refer to these numbers in their comments, so
`# implements stats.md eq. (11.4)` always resolves.

**A reading order that works.** Do not read Topics 1-35 in sequence on a first
pass. Follow the five stages in the decision-centred curriculum:

1. **Core inference**: Topics 1-9. Stop when you can state an estimand.
2. **Unified modelling**: Topics 10-15. Stop when a *t*-test looks like `lm()`.
3. **High-dimensional analysis**: Topics 16-19, 29, 35.
4. **Modality tracks**: pick three of Topics 20-28.
5. **Advanced reasoning**: Topics 30-34.


## Notation

| Symbol | Meaning |
|---|---|
| $n$ | Number of independent experimental units (**not** number of cells or reads) |
| $p$ | Number of parameters in a model; also number of predictors |
| $G$ | Number of features (genes, CpGs, proteins, taxa, variants) tested |
| $i, j$ | Index over units $i = 1,\dots,n$ and features $g = 1,\dots,G$ |
| $Y, y$ | Random outcome variable; its realised value |
| $\mathbf{y}$ | The $n \times 1$ outcome vector |
| $\mathbf{X}$ | The $n \times p$ design matrix |
| $\boldsymbol\beta$ | The $p \times 1$ coefficient vector; $\hat{\boldsymbol\beta}$ its estimate |
| $\boldsymbol\varepsilon$ | Residual/error vector |
| $\theta$ | A generic parameter; $\hat\theta$ a generic estimator |
| $\mathbb{E}[\cdot]$, $\operatorname{Var}(\cdot)$ | Expectation and variance |
| $\operatorname{Cov}(\cdot,\cdot)$ | Covariance |
| $\sim$ | "is distributed as" |
| $\mathcal{N}(\mu, \sigma^2)$ | Normal with mean $\mu$, variance $\sigma^2$ |
| $\stackrel{d}{\to}$ | Convergence in distribution |
| $\ell(\theta)$ | Log-likelihood $\log L(\theta)$ |
| $\mathcal{I}(\theta)$ | Fisher information |
| $\mathbf{I}_n$ | $n \times n$ identity matrix |
| $\mathbf{1}_n$ | $n \times 1$ vector of ones |
| $\|\mathbf{v}\|$ | Euclidean norm $\sqrt{\mathbf{v}^\top\mathbf{v}}$ |
| $\mathbb{1}\{A\}$ | Indicator: 1 if $A$ is true, 0 otherwise |
| $\alpha$ | Type I error rate; $1-\beta$ is power |
| $\text{logit}(\pi)$ | $\log\!\big(\pi/(1-\pi)\big)$ |

**Vectors are columns.** $\mathbf{x}^\top\mathbf{y}$ is a scalar (inner product);
$\mathbf{x}\mathbf{y}^\top$ is a matrix (outer product). Getting this wrong is
the commonest source of dimension errors in omics code.


## Module map

### Foundations: start here if you are new to statistics

`Part 0` of this document has no prerequisites and assumes no mathematics.
These six modules are its runnable companions.

| Part 0 section | Python | R |
|---|---|---|
| §0.1-§0.3 variables, populations, samples | `statsPy/foundations/F1_variables_populations_samples.py` | `statsR/foundations/F1_variables_populations_samples.R` |
| §0.4 probability and independence | `statsPy/foundations/F2_probability_basics.py` | `statsR/foundations/F2_probability_basics.R` |
| §0.5 distributions | `statsPy/foundations/F3_distributions.py` | `statsR/foundations/F3_distributions.R` |
| §0.6 estimates, standard errors, the CLT | `statsPy/foundations/F4_estimates_and_standard_errors.py` | `statsR/foundations/F4_estimates_and_standard_errors.R` |
| §0.7-§0.8 confidence intervals and p-values | `statsPy/foundations/F5_confidence_intervals_and_pvalues.py` | `statsR/foundations/F5_confidence_intervals_and_pvalues.R` |
| §0.9 connecting variables, the four causal roles | `statsPy/foundations/F6_connecting_variables.py` | `statsR/foundations/F6_connecting_variables.R` |

### Core and applied

| stats.md topic | Python module | R module |
|---|---|---|
| 1 Study design | `statsPy/core/01_study_design_and_estimands.py` | `statsR/core/01_study_design_and_estimands.R` |
| 2 Data structures | `statsPy/core/02_data_structures_and_scales.py` | `statsR/core/02_data_structures_and_scales.R` |
| 3 EDA | `statsPy/core/03_exploratory_data_analysis.py` | `statsR/core/03_exploratory_data_analysis.R` |
| 4 Probability | `statsPy/core/04_probability_and_sampling.py` | `statsR/core/04_probability_and_sampling.R` |
| 5 Estimation | `statsPy/core/05_estimation_and_intervals.py` | `statsR/core/05_estimation_and_intervals.R` |
| 6 p-values | `statsPy/core/06_hypothesis_tests_and_pvalues.py` | `statsR/core/06_hypothesis_tests_and_pvalues.R` |
| 7 t / rank / permutation | `statsPy/core/07_ttests_ranks_permutation.py` | `statsR/core/07_ttests_ranks_permutation.R` |
| 8 Multiple testing | `statsPy/core/08_multiple_testing.py` | `statsR/core/08_multiple_testing.R` |
| 9 Power | `statsPy/core/09_power_and_sample_size.py` | `statsR/core/09_power_and_sample_size.R` |
| 10 Correlation | `statsPy/core/10_correlation_and_dependence.py` | `statsR/core/10_correlation_and_dependence.R` |
| 11 Linear models | `statsPy/core/11_linear_models.py` | `statsR/core/11_linear_models.R` |
| 12 ANOVA / contrasts | `statsPy/core/12_anova_and_contrasts.py` | `statsR/core/12_anova_and_contrasts.R` |
| 13 GLMs | `statsPy/core/13_generalized_linear_models.py` | `statsR/core/13_generalized_linear_models.R` |
| 14 Mixed models | `statsPy/core/14_mixed_models.py` | `statsR/core/14_mixed_models.R` |
| 15 Missing data | `statsPy/core/15_missing_data_and_censoring.py` | `statsR/core/15_missing_data_and_censoring.R` |
| 16 Matrix algebra | `statsPy/core/16_matrix_algebra.py` | `statsR/core/16_matrix_algebra.R` |
| 17 PCA | `statsPy/core/17_pca.py` | `statsR/core/17_pca.R` |
| 18 Clustering | `statsPy/core/18_distances_and_clustering.py` | `statsR/core/18_distances_and_clustering.R` |
| 19 Batch effects | `statsPy/core/19_batch_effects.py` | `statsR/core/19_batch_effects.R` |
| 28 Survival / longitudinal | `statsPy/core/20_survival_and_longitudinal.py` | `statsR/core/20_survival_and_longitudinal.R` |
| 31 Prediction | `statsPy/core/21_prediction_and_validation.py` | `statsR/core/21_prediction_and_validation.R` |
| 32 Causal inference | `statsPy/core/22_causal_inference.py` | `statsR/core/22_causal_inference.R` |
| 33 Bayes / shrinkage | `statsPy/core/23_bayesian_and_shrinkage.py` | `statsR/core/23_bayesian_and_shrinkage.R` |
| 34-35 Diagnostics, reproducibility | `statsPy/core/24_diagnostics_and_reproducibility.py` | `statsR/core/24_diagnostics_and_reproducibility.R` |
| 20 Bulk RNA-seq | `statsPy/bioinformatics/30_bulk_rnaseq_differential_expression.py` | `statsR/bioinformatics/30_bulk_rnaseq_differential_expression.R` |
| 21 Single-cell | `statsPy/bioinformatics/31_single_cell_pseudobulk.py` | `statsR/bioinformatics/31_single_cell_pseudobulk.R` |
| 22 Cytometry | `statsPy/bioinformatics/32_cytometry_differential_abundance.py` | `statsR/bioinformatics/32_cytometry_differential_abundance.R` |
| 23 Methylation | `statsPy/bioinformatics/33_methylation_epigenomics.py` | `statsR/bioinformatics/33_methylation_epigenomics.R` |
| 24 GWAS | `statsPy/bioinformatics/34_gwas_association.py` | `statsR/bioinformatics/34_gwas_association.R` |
| 25 Proteomics | `statsPy/bioinformatics/35_proteomics_missing_values.py` | `statsR/bioinformatics/35_proteomics_missing_values.R` |
| 26 Microbiome | `statsPy/bioinformatics/36_microbiome_compositional.py` | `statsR/bioinformatics/36_microbiome_compositional.R` |
| 27 Spatial | `statsPy/bioinformatics/37_spatial_omics.py` | `statsR/bioinformatics/37_spatial_omics.R` |
| 28 Clinical survival | `statsPy/bioinformatics/38_survival_biomarkers.py` | `statsR/bioinformatics/38_survival_biomarkers.R` |
| 29 Enrichment | `statsPy/bioinformatics/39_gene_set_enrichment.py` | `statsR/bioinformatics/39_gene_set_enrichment.R` |
| 30 Networks / multi-omics | `statsPy/bioinformatics/40_networks_and_multiomics.py` | `statsR/bioinformatics/40_networks_and_multiomics.R` |

### Integrative exercises

Each exercise combines several topics into one realistic problem, with the
worked solution commented out beneath every question.

| Topics combined | Python | R |
|---|---|---|
| 1, 7, 9, 14, 19 - design audit | `statsPy/exercises/E1_design_audit.py` | `statsR/exercises/E1_design_audit.R` |
| 8, 13, 20, 29 - RNA-seq end to end | `statsPy/exercises/E2_rnaseq_end_to_end.py` | `statsR/exercises/E2_rnaseq_end_to_end.R` |
| 18, 21, 31 - prediction audit | `statsPy/exercises/E3_prediction_audit.py` | `statsR/exercises/E3_prediction_audit.R` |
| 11, 15, 32, 34 - causal question | `statsPy/exercises/E4_causal_question.py` | `statsR/exercises/E4_causal_question.R` |

See `README.md` for how to run these, and `tools/check_solutions.sh` for the
runner that executes every shipped solution.


# Part 0: Starting from zero

> **Read this part if you have never studied statistics.** It assumes no
> mathematics beyond arithmetic. Every symbol is defined the first time it
> appears, and every idea is introduced in plain English before any notation.
> If you already know what a standard error is, skip to
> [Part I](#part-i-foundations).
>
> The runnable companions are `statsPy/foundations/` and `statsR/foundations/`
> (modules `F1`-`F6`). Read a section here, then run the matching module and
> watch the idea happen to actual numbers.


## 0.1 What statistics is actually for

You did an experiment on six mice. You want to say something about mice in
general.

That sentence contains the whole problem. You measured **six** animals, but you
want to make a claim about **all** animals of that kind - including the ones
you did not measure and never will. Statistics is the set of tools for doing
that honestly: for saying how far a conclusion drawn from the few reaches
toward the many, and for being explicit about how often that reach fails.

Three things follow, and they shape everything else in this document.

**Your data is not the answer. It is evidence about the answer.** The average
of your six mice is not "the effect". It is one noisy glimpse of the effect. Run
the experiment again with six different mice and you get a different number.
Statistics is largely the study of *how different* that second number would be.

**Uncertainty is not failure.** A result reported without uncertainty is not a
stronger result, it is a less honest one. "The treatment lowered the marker by
2.3 units" is worth much less than "by 2.3 units, and the data are consistent
with anywhere from 0.4 to 4.2".

**Most errors are errors of judgement, not arithmetic.** Computers do the
arithmetic. What goes wrong is deciding what counts as an independent
observation, what question you are actually answering, and what you decided
*after* looking at the data. Those are the subject of this course.

### Two jobs statistics does

| Job | The question | Example |
|---|---|---|
| **Description** | What happened in the data I have? | "The treated mice averaged 12.4 units." |
| **Inference** | What does that imply about data I do *not* have? | "Treatment changes the marker by 2.3 units (95% CI 0.4 to 4.2)." |

Description is arithmetic and is never wrong. Inference is a *leap* - from the
measured to the unmeasured - and it is the leap that can fail. Almost every
technique in this document exists to make one particular leap safer.


## 0.2 Variables: the things you measure

A **variable** is anything you record that can differ between the things you
measure. Weight is a variable. Treatment group is a variable. So is the count of
reads mapping to a gene.

A **unit** (or *observation*, or *case*) is the thing being measured - a mouse,
a patient, a cell, a tumour sample. Deciding what your unit is turns out to be
the most consequential decision in experimental biology, and Topic 1 is
devoted to it.

Data is usually laid out with one row per unit and one column per variable:

| mouse | treatment | weight_g | marker | genotype |
|---|---|---|---|---|
| m1 | control | 24.1 | 12.6 | wild-type |
| m2 | control | 22.8 | 11.9 | wild-type |
| m3 | drug | 23.5 | 10.2 | mutant |

### The types of variable, and why the type matters

The type determines which arithmetic is *meaningful*, which determines which
methods are valid. This is not pedantry - it is the source of a large fraction
of real analysis errors.

**Continuous**: can take any value in a range, and differences are meaningful.
Weight, concentration, temperature, log-expression. You can average it.

**Count**: a whole number of things: reads, colonies, cells, events. Counts are
never negative, and their *variability grows with their size* - a gene averaging
10,000 reads varies by hundreds, one averaging 10 reads varies by two or three.
This is why RNA-seq cannot be analysed with methods designed for continuous
data, and it is the whole subject of Topics 13 and 20.

**Binary**: exactly two possible values: responded / did not, alive / dead,
mutated / wild-type. Usually coded 0 and 1, which makes the average of the
column the *proportion* of 1s - a fact used constantly.

**Categorical (nominal)**: several unordered labels: cell type, treatment arm,
sequencing centre. There is no meaningful average of "B cell" and "T cell".
Software must be told these are labels, not numbers; forgetting is a classic
error, and one that produces no warning.

**Ordinal**: ordered labels with no fixed spacing: tumour grade I/II/III, "mild
/ moderate / severe". Grade III is worse than grade II, but it is not "one unit"
worse in any defensible sense, and the gap between I and II need not equal the
gap between II and III. Treating ordinal data as continuous is sometimes
defensible and always a decision you should make consciously.

**Proportion / compositional**: parts of a whole that must sum to 1: cell-type
fractions, microbiome relative abundances. These carry a hidden trap. If one
component goes up, the others *must* go down, whether or not anything happened
to them. Topics 22 and 26 exist because of this.

**Time-to-event (censored)**: how long until something happens, where for some
units it has not happened yet when the study ends. You know patient 7 survived
*at least* 400 days, not how long they will survive. That partial information is
precious and discarding it is a serious error. This is Topic 28.

> **Why this matters immediately.** Nearly every "which test should I use?"
> question is answered by naming the variable type of the outcome and the
> structure of the design. Get the type right and the method usually follows.


## 0.3 Populations and samples

The **population** is everything you want to draw a conclusion about: all
laboratory mice of this strain, all patients with this tumour subtype, all cells
in this tissue. It is usually infinite, hypothetical, and unobservable.

The **sample** is what you actually measured.

A number describing the population is a **parameter**. It is fixed, unknown, and
what you would like to know. A number computed from your sample is a
**statistic** (or an **estimate**). It is known, and it varies from sample to
sample.

The convention throughout this document, and in essentially all of statistics:

| | Population (unknown, fixed) | Sample (known, varies) |
|---|---|---|
| Mean | $\mu$ ("mu") | $\bar{x}$ ("x-bar") |
| Standard deviation | $\sigma$ ("sigma") | $s$ |
| Proportion | $\pi$ or $p$ | $\hat{p}$ ("p-hat") |
| Any parameter | $\theta$ ("theta") | $\hat{\theta}$ ("theta-hat") |

**Greek letters are truth; Latin letters and hats are guesses.** That one rule
makes most statistical notation readable.

### Random sampling, and what it buys you

A **simple random sample** means every member of the population had an equal
chance of being picked, independently of the others. This is the assumption that
licenses the leap from sample to population. It is also almost never literally
true in biology - you used the mice the facility sent, the patients who consented
at your hospital.

The practical question is never "was this a perfect random sample?" (no) but
"is the way it departed from random *related to what I am measuring*?" If your
control mice came from one cage and your treated mice from another, the
departure is related to the comparison, and your experiment is compromised. If
they were randomly assigned across cages, it is not. That difference is the
subject of Topic 1, and the reason **randomisation** is worth more than any
analysis method.

### The bit everyone finds surprising

> Precision depends almost entirely on the **size of your sample**, not on what
> fraction of the population it represents.

A sample of 1,000 people tells you about as much about the United States
(330 million) as about Iceland (400 thousand). Your intuition says the second
should be far more precise because it covers a much larger share. It is not.
What matters is $n$, not $n/N$. This is why "we sequenced 90% of the cells in
the sample" is not the reassurance it sounds like - if those cells all came from
three patients, your $n$ is three.


## 0.4 Probability from scratch

Probability is the mathematics of *things that could have come out otherwise*.
You need it because your sample could have come out otherwise.

### The rules, in full

A probability is a number between 0 and 1. Zero means impossible, one means
certain, 0.5 means it happens half the time in the long run.

**Everything sums to one.** If you list every possible outcome, their
probabilities add to 1. Something must happen.

$$\sum_{\text{all outcomes}} P(\text{outcome}) = 1 \tag{0.1}$$

The symbol $\sum$ ("sigma", capital) means "add up all of these". It is
shorthand for a list, nothing more.

**Either/or adds**: when two outcomes cannot both happen:

$$P(A \text{ or } B) = P(A) + P(B) \qquad \text{(if } A, B \text{ are mutually exclusive)} \tag{0.2}$$

The chance a die shows 1 or 2 is $1/6 + 1/6 = 1/3$.

**Both-of adds only if independent**: two events are **independent** when
knowing one happened tells you nothing about the other:

$$P(A \text{ and } B) = P(A) \times P(B) \qquad \text{(if } A, B \text{ are independent)} \tag{0.3}$$

Two coin flips: $P(\text{two heads}) = 0.5 \times 0.5 = 0.25$.

> **Independence is the assumption that fails most often in biology**, and
> eq. (0.3) is where the damage starts. Twenty cells from the same mouse are not
> twenty independent observations of the treatment - they share that mouse's
> genetics, cage, handling and mood. Treat them as independent and you multiply
> probabilities that should not be multiplied, and the resulting p-value can be
> wrong by many orders of magnitude. This is **pseudoreplication**, and it is the
> single most common fatal flaw in published biology. Topic 1 quantifies it;
> exercise `E1` shows a *p* = 2e-12 that should have been *p* = 0.06.

### Conditional probability: probability given that you know something

$P(A \mid B)$, read "the probability of $A$ **given** $B$", is the chance of $A$
in the restricted world where $B$ is known to have happened.

$$P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)} \tag{0.4}$$

This is the most misread idea in applied statistics, because

$$P(A \mid B) \neq P(B \mid A).$$

The probability that someone is wet given that it is raining is high. The
probability that it is raining given that someone is wet is much lower - they
might be swimming.

That asymmetry is exactly the mistake people make with p-values. A p-value is
$P(\text{data this extreme} \mid \text{no real effect})$. It is *not*
$P(\text{no real effect} \mid \text{data})$. Confusing the two is the
**prosecutor's fallacy**, and §0.8 returns to it.

### Random variables, and expectation

A **random variable** is a measurement whose value depends on chance - written
with a capital letter, $X$. Before you run the experiment, the weight of the
next mouse is a random variable. After you weigh it, it is just a number.

The **expected value** $E[X]$ is the long-run average - the number you would
converge to if you could repeat forever. For a population it *is* $\mu$.

$$E[X] = \mu \tag{0.5}$$

The **variance** measures how spread out the values are: take each value's
distance from the mean, square it (so that being below the mean counts the same
as being above), and average:

$$\operatorname{Var}(X) = E\big[(X - \mu)^2\big] = \sigma^2 \tag{0.6}$$

Squaring makes the units wrong - squared grams - so we usually take the square
root and get back to the original units. That is the **standard deviation**:

$$\sigma = \sqrt{\operatorname{Var}(X)} \tag{0.7}$$

**Standard deviation is the natural measure of "how spread out".** If mouse
weights average 24 g with a standard deviation of 2 g, most mice weigh roughly
22-26 g.

Two facts about variance do a great deal of work later:

$$\operatorname{Var}(aX) = a^2 \operatorname{Var}(X) \tag{0.8}$$

$$\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) \qquad \text{(if } X, Y \text{ independent)} \tag{0.9}$$

Eq. (0.9) is why noise partially cancels when you average, and therefore why
larger samples are more precise. It is also why eq. (0.9) failing - because
observations are *not* independent - is so destructive. Everything in §0.6
depends on it.


## 0.5 Distributions

A **distribution** describes which values a variable takes and how often. It is
the full picture, of which the mean and standard deviation are two summaries.

You can see a distribution in your data as a histogram. The idealised,
infinite-population version is a mathematical curve, and a handful of these
curves describe most biological measurements. **You do not need to memorise
their formulas.** You need to recognise which situation produces which shape,
because that determines the right method.

### The ones that matter

**Normal (Gaussian, "bell curve").** Symmetric, centred at $\mu$, with spread
$\sigma$. Written $X \sim N(\mu, \sigma^2)$ - the "$\sim$" means "is distributed
as". Roughly 68% of values fall within one $\sigma$ of the mean, 95% within two.

It appears constantly for a deep reason (§0.6), not because biology is
intrinsically bell-shaped. Heights, log-expression values and measurement errors
are often close to normal. Raw counts and concentrations usually are not.

**Binomial.** The number of successes in $n$ independent yes/no trials each with
probability $p$: how many of 50 patients respond, how many of 200 cells are
positive. Mean $np$, variance $np(1-p)$.

**Poisson.** The number of events in a fixed window when events happen
independently at a constant rate: reads mapping to a gene, mutations per
genome, colonies per plate. Its defining and very restrictive property:

$$\text{mean} = \text{variance} = \lambda \tag{0.10}$$

**Negative binomial.** A Poisson whose rate itself varies between samples. Real
biological count data is almost always more variable than Poisson allows -
because mice, patients and cultures genuinely differ - and this extra spread is
called **overdispersion**. The negative binomial adds one parameter to absorb
it, and is the workhorse of RNA-seq (Topic 20).

**Uniform.** Every value in a range equally likely. Mostly it matters for one
reason: **a p-value from a correct test, when there is genuinely no effect, is
uniform between 0 and 1.** That single fact is the basis of almost every
diagnostic in this course - if your p-values are not uniform under the null,
something is wrong, and you can see it in a histogram.

**t, chi-squared, F.** These describe not measurements but *test statistics* -
numbers computed from samples. You meet them when testing, not when measuring.
The **t distribution** is the normal's heavier-tailed cousin, used when $\sigma$
is estimated rather than known; with small samples it has fatter tails, which is
precisely what stops small studies from being over-confident.

### Why shape matters

Reporting a mean and standard deviation for a strongly skewed variable is
misleading: most values sit below the mean, and the mean is dragged by a few
large ones. Gene expression, concentrations and survival times are typically
right-skewed, which is why so much of biology is analysed on a **log scale** -
the logarithm turns multiplicative, skewed variation into additive, symmetric
variation, where the standard toolkit works.


## 0.6 Estimates, standard errors, and why averaging works

You compute $\bar{x} = 12.4$ from six mice. What is that number *worth*?

An **estimator** is the recipe ("take the average of the sample"). An
**estimate** is the number that recipe produced this time (12.4). The
distinction matters because the recipe has properties - the number does not.

### The sampling distribution

Here is the idea everything else rests on. Imagine repeating your entire
experiment thousands of times, each with six fresh mice, computing $\bar{x}$ each
time. Those thousands of averages form a distribution - the **sampling
distribution** of the mean.

You never observe it. You get one draw from it. But its *properties* are exactly
what "how much should I trust 12.4?" means, and remarkably, they can be worked
out from a single sample.

**Two theorems do all the work.**

**The law of large numbers:** as $n$ grows, $\bar{x}$ converges to $\mu$. Bigger
samples give estimates closer to the truth.

**The central limit theorem:** as $n$ grows, the sampling distribution of
$\bar{x}$ becomes **normal**: *whatever shape the original data had*. Averages
of skewed, lumpy, ugly data are still approximately bell-shaped. This is why the
normal distribution is everywhere in inference, and why methods based on it work
far more often than they have any right to.

### The standard error

The standard deviation *of the sampling distribution* is called the **standard
error**. For a sample mean:

$$\mathrm{SE}(\bar{x}) = \frac{\sigma}{\sqrt{n}} \tag{0.11}$$

and since $\sigma$ is unknown, we use the sample's own spread $s$:

$$\widehat{\mathrm{SE}}(\bar{x}) = \frac{s}{\sqrt{n}} \tag{0.12}$$

> **Standard deviation and standard error are different things and are
> constantly confused.**
>
> * **$s$ (standard deviation)** describes how much *individual mice* vary. It
>   does not shrink as you collect more data - mice are as variable as they are.
> * **$\mathrm{SE}$ (standard error)** describes how much *your estimate* would
>   vary across repeat experiments. It shrinks as $\sqrt{n}$.
>
> Use SD to describe your sample. Use SE to say how precisely you know a
> quantity. Error bars in papers are often unlabelled, which makes them
> uninterpretable - always say which one you plotted.

Eq. (0.11) comes straight from eq. (0.9): averaging $n$ independent things
divides the variance by $n$, and therefore the standard deviation by $\sqrt{n}$.

**The $\sqrt{n}$ is the tyranny of statistics.** To halve your uncertainty you
must *quadruple* your sample. To improve tenfold you need a hundred times the
data. This is why powerful studies are expensive, and why gains from better
design usually beat gains from more samples.

And note the condition on eq. (0.9): *independent*. If your $n$ observations are
not independent - twenty cells per mouse - the division by $n$ is not earned.
You get an SE that is too small, a confidence interval that is too narrow, and a
p-value that is too small. Topic 1 shows exactly how much too small.


## 0.7 Confidence intervals

A single number is a poor answer. "The difference was 2.3 units" hides whether
the data could equally well support 0.1 or 5.0.

A **confidence interval** is a range of values consistent with your data. The
standard form, once the central limit theorem applies:

$$\text{estimate} \;\pm\; (\text{critical value}) \times \mathrm{SE} \tag{0.13}$$

For a 95% interval based on the normal distribution the critical value is 1.96
(often rounded to 2), because 95% of a normal distribution lies within 1.96
standard deviations of its centre:

$$\bar{x} \pm 1.96 \times \frac{s}{\sqrt{n}} \tag{0.14}$$

With small samples you use the $t$ distribution's slightly larger critical value
instead, which widens the interval appropriately.

### What "95% confident" actually means

This is worth getting exactly right, because the natural reading is wrong.

> **Correct:** if you repeated the whole experiment many times and computed an
> interval each time, 95% of *those intervals* would contain the true value.

The **procedure** succeeds 95% of the time. Your particular interval either
contains the truth or does not - there is no probability left in it, because the
truth is fixed and your interval is fixed.

> **Incorrect:** "there is a 95% probability the true value lies in my
> interval."

This is a statement about the truth given the data - and as §0.4 established,
that is the reverse conditional. Getting a statement of that form requires
Bayesian methods and a prior (Topic 33), which give you a *credible* interval.
In practice people read confidence intervals the Bayesian way and are usually
not badly misled, but the distinction becomes real whenever the two disagree.

### How to read one

The **width** is the information content, and it is the part to read first.

* Narrow interval -> precise estimate, regardless of where it sits.
* Wide interval -> your data is consistent with many different answers.
* Interval excludes zero -> "significant" at the matching level. But this is the
  *least* interesting thing about it.

Compare these two results, both "non-significant":

| Result | Reading |
|---|---|
| difference 0.1, CI (-0.2, 0.4) | genuinely no meaningful effect - you have *ruled out* anything large |
| difference 2.5, CI (-0.5, 5.5) | uninformative - a large effect is entirely consistent with this data |

Both have $p > 0.05$. They mean completely different things, and only the
interval tells you which you have. **This is the single strongest practical
argument for reporting intervals rather than p-values**, and it is why
"no significant difference" should never be written as "no difference".


## 0.8 p-values: what they are, the kinds, and how thresholds get chosen

### The logic

A hypothesis test is an argument by contradiction with the contradiction
softened into a probability.

1. Assume, for the sake of argument, that **nothing is going on**. This is the
   **null hypothesis** $H_0$: the treatment does nothing, the difference is
   zero, the gene is unchanged.
2. Ask: *if that were true*, how likely is data at least as extreme as what I
   saw?
3. That probability is the **p-value**. If it is small, the data is hard to
   reconcile with "nothing is going on".

Formally:

$$p = P\big(\text{a test statistic at least as extreme as observed} \mid H_0 \text{ true}\big) \tag{0.15}$$

**Note what is on which side of that bar.** The p-value is the probability of
*the data* given the *hypothesis*. It is not the probability of the hypothesis
given the data. A p-value of 0.03 does **not** mean a 3% chance the effect is
not real (§0.4).

### What a p-value is not

* **Not** the probability the null hypothesis is true.
* **Not** the probability your result was a fluke.
* **Not** a measure of effect size. With 10,000 samples, a biologically
  meaningless difference produces a tiny p-value. With 4 samples, a huge effect
  may not reach significance. *p tells you about evidence against a hypothesis,
  never about importance.*
* **Not** a statement about replication. $p = 0.04$ does not mean a repeat
  experiment has a 96% chance of working.

### The kinds of p-value

This is the part most introductions skip, and it is where practical confusion
concentrates.

**By how the tail is counted**

* **Two-sided**: "different in either direction". The default, and the right
  choice almost always. Counts extremes at both ends.
* **One-sided**: "larger", specifically. Half the two-sided p-value, and
  therefore easier to make significant, which is exactly why it is abused.
  Legitimate only when a difference in the other direction would be *acted on
  identically to no difference*, and only when declared before seeing the data.
  Switching to one-sided after a two-sided test gives $p = 0.06$ is a form of
  cheating with a name: **p-hacking**.

**By how the null distribution is obtained**

* **Parametric**: assumes a distributional form (normal, negative binomial) and
  reads the p-value off that curve. Fast and powerful when the assumption
  roughly holds. Examples: $t$-test, ANOVA, most GLMs. (Topic 7)
* **Exact**: enumerates every possible arrangement of the data. No
  distributional assumption at all, but only feasible for small samples or
  simple designs. Example: Fisher's exact test. (Topic 29)
* **Permutation**: shuffles the labels many times to build the null
  distribution empirically. Assumes only **exchangeability**: that under the
  null, the labels could equally well have been attached to any unit. Robust and
  widely applicable. *But you must shuffle within the structure of your design*:
  with batches or paired samples, free shuffling produces a null that is simply
  wrong (exercise `E2` measures this). (Topic 7)
* **Rank-based**: replaces values by their ranks, then tests. Immune to
  outliers and to monotone transformations. Examples: Wilcoxon, Mann-Whitney.
  Answers a slightly different question than the $t$-test, which is a feature
  when your data is skewed and a trap when you wanted a mean difference.
  (Topic 7)
* **Bootstrap**: resamples your data with replacement to approximate the
  sampling distribution directly. Most natural for confidence intervals, and
  invaluable for statistics with no clean formula. (Topic 5)

**By what has been done to it afterwards**

* **Raw (nominal)**: as computed, for one test.
* **Adjusted / corrected**: modified to account for having done many tests.
  Testing 20,000 genes at $p < 0.05$ yields about 1,000 false positives if
  nothing is going on at all. Two philosophies:
  * **FWER** (family-wise error rate; Bonferroni, Holm) controls the probability
    of *even one* false positive. Appropriate when a single false claim is
    costly. Very conservative with many tests.
  * **FDR** (false discovery rate; Benjamini-Hochberg) controls the *expected
    proportion* of your discoveries that are false. The right default in
    genomics: you accept that 5% of your 200-gene list is wrong in exchange for
    finding the list at all. Adjusted values are called **q-values**.
  * This is Topic 8, and it is unavoidable in any genome-scale analysis.

### Where does 0.05 come from?

**It was a convention, chosen by one person, for convenience.** Ronald Fisher
suggested in the 1920s that one-in-twenty was a reasonable line for deciding a
result deserved a second look. It was never a law of nature, and Fisher
explicitly treated it as a rule of thumb rather than a standard of proof.

It persists because it is a shared default, and shared defaults have real value -
they stop people from choosing a threshold after seeing the answer. But it has
no special claim to correctness, and the right threshold depends on what the
two kinds of error cost:

* **Type I error**: a false positive. You claim an effect that is not there.
  The threshold $\alpha$ *is* your false positive rate.
* **Type II error**: a false negative. You miss an effect that is there. Its
  probability is $\beta$, and $1-\beta$ is the **power**.

Lowering $\alpha$ buys fewer false positives at the cost of more false
negatives. So the threshold should follow the consequences: a screening assay
where follow-up is cheap can tolerate $\alpha = 0.1$; a genome-wide association
study conventionally uses $5 \times 10^{-8}$, because it performs about a
million independent tests and a single false claim would be pursued expensively
for years (Topic 24).

**The modern consensus**: the American Statistical Association's 2016 statement
and the 2019 follow-up - is that "statistically significant" should not be
treated as a conclusion at all. Report the estimate, the interval, and the
p-value; describe the evidence; do not let a threshold do your thinking. This
document follows that position throughout.

### The dangerous habits, named

* **p-hacking**: trying analyses until one gives $p < 0.05$: dropping outliers,
  adding covariates, switching tests, testing subgroups. The reported p-value is
  then meaningless, because the *procedure* that produced it had a false
  positive rate far above 5%.
* **HARKing**: Hypothesising After the Results are Known. Presenting a finding
  you discovered in the data as though you had predicted it.
* **Optional stopping**: peeking at the data and continuing to collect until
  significance appears. This reaches $p < 0.05$ eventually with probability 1,
  even when nothing is happening.
* **The threshold cliff**: treating $p = 0.049$ and $p = 0.051$ as different
  findings. They are the same finding.

The protection against all four is the same: **decide the analysis before seeing
the outcome**, and report everything you tried. Topics 8 and 34 make this
concrete, and Module 24 shows what a pre-specified analysis plan looks like.


## 0.9 How statistics connects variables

Most real questions are not "what is the average?" but "does this relate to
that?" - does the drug change the marker, does expression predict survival.

### The vocabulary of roles

A variable's *type* (§0.2) is a fact about how it was measured. Its **role** is
a decision you make about the question you are asking. The same column can be an
outcome in one analysis and a covariate in another.

* **Outcome** (dependent variable, response, $Y$) - what you are trying to
  explain or predict. Expression level, survival time, response status.
* **Predictor** (independent variable, explanatory variable, covariate, feature,
  $X$) - what you use to explain it. Treatment group, dose, age, genotype.
* **Covariate**: a predictor that is not the focus but is included anyway,
  usually to improve precision or to adjust for a difference between groups.

### Four roles that look identical in the data and are not

This is the most important idea in §0.9, and the reason Topic 32 exists.
Suppose you are studying whether a drug (D) changes a marker (M).

**Confounder**: a variable that causes *both* the predictor and the outcome.

```
        severity
        /      \
       v        v
     drug ---> marker
```

Sicker patients are more likely to be given the drug *and* have worse markers.
The crude drug-marker comparison mixes the drug's effect with the fact that
treated patients were sicker to begin with. **You must adjust for confounders**,
and failing to do so is the classic reason observational studies mislead -
exercise `E4` contains one that flips the sign of the answer.

**Mediator**: a variable on the causal path *from* the predictor *to* the
outcome.

```
     drug ---> enzyme ---> marker
```

The drug works *by* changing the enzyme. Adjusting for the enzyme removes part
of the very effect you are trying to measure. **Do not adjust for mediators**
unless you specifically want the direct effect.

**Collider**: a variable caused by *both* the predictor and the outcome (or by
the predictor and something that causes the outcome).

```
     drug ---> toxicity <--- frailty ---> marker
```

Here is the counter-intuitive part: adjusting for a collider **creates** a
spurious association that was not there. Conditioning on a common effect makes
its causes appear related. **Adjusting for more variables is not safer.** In
`E4` this bias is large, and every statistical signal - significance, $R^2$,
residual variance - says to include the variable.

**Precision variable**: causes the outcome but not the predictor. Adjusting for
it does not change the estimate but reduces its noise. Free precision; include
it.

> **Nothing in your data distinguishes these four.** A confounder and a collider
> can look statistically identical. The distinction comes from knowledge of how
> the data were generated - from biology and from the time order of events -
> which is why you draw the causal diagram *before* you fit anything, and why
> automatic variable selection is not a substitute for thinking.
>
> One reliable rule: **never adjust for anything measured after the
> treatment**, because it is a mediator or a collider or both.

### The one model behind almost everything

Nearly every method in this course is a variation on one line:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \varepsilon \tag{0.16}$$

In words: *the outcome equals a baseline, plus a contribution from each
predictor, plus what is left over.*

* $\beta_0$ ("beta-nought") - the **intercept**: the outcome when all predictors
  are zero.
* $\beta_1$ - the **slope** or **coefficient**: how much $Y$ changes per one-unit
  increase in $X_1$, **holding the other predictors fixed**. That last clause is
  what "adjusting for" means, and it is the whole of confounder control.
* $\varepsilon$ ("epsilon") - the **error** or **residual**: everything the
  model does not explain.

Once you can read eq. (0.16), a surprising amount of statistics becomes the same
thing wearing different clothes:

| Method | Is really eq. (0.16) with... |
|---|---|
| Two-sample $t$-test | one binary predictor |
| ANOVA | one categorical predictor |
| Multiple regression | several predictors |
| ANCOVA | a predictor of interest plus covariates |
| Logistic regression | a binary outcome, modelled on the log-odds scale |
| Poisson / negative binomial regression | a count outcome, modelled on the log scale |
| Mixed model | plus a term for each group, to handle non-independence |
| Cox regression | applied to the hazard of an event over time |

That is why Topic 11 calls the linear model "the core framework" and why it
repays more study than any individual test. Learning twenty tests is memorising.
Learning eq. (0.16) and its extensions is understanding.

### Association is not causation

A coefficient in eq. (0.16) is an **association**: when $X$ differs by one unit,
$Y$ tends to differ by $\beta$. Whether that reflects $X$ *causing* $Y$ depends
on the design, not on the arithmetic.

* **Randomised experiment**: you assigned $X$ yourself, so nothing can have
  caused both $X$ and $Y$. Confounding is eliminated *by design*. This is why
  randomisation is worth more than any amount of statistical adjustment.
* **Observational study**: $X$ happened for reasons you did not control, and
  those reasons may also affect $Y$. You can adjust for confounders you measured
  and thought of. You cannot adjust for the ones you did not. Every causal claim
  from observational data rests on an assumption that is not checkable from the
  data (Topic 32).


## 0.10 A plain-English glossary

| Term | In one sentence |
|---|---|
| **Population** | Everything you want to conclude about. |
| **Sample** | What you actually measured. |
| **Unit** | The thing measured - and the thing that must be independent. |
| **Parameter** | A fixed unknown number describing the population ($\mu$, $\sigma$). |
| **Statistic / estimate** | A number computed from your sample ($\bar{x}$, $s$). |
| **Estimator** | The recipe that produces the estimate. |
| **Mean** | The average. |
| **Median** | The middle value; unaffected by extremes. |
| **Variance** | Average squared distance from the mean. |
| **Standard deviation (SD)** | Typical distance of an *observation* from the mean. |
| **Standard error (SE)** | Typical distance of an *estimate* from the truth; shrinks as $\sqrt{n}$. |
| **Distribution** | Which values occur and how often. |
| **Normal distribution** | The bell curve; what averages look like (CLT). |
| **Overdispersion** | More variability than the assumed model allows. |
| **Independence** | Knowing one observation tells you nothing about another. |
| **Pseudoreplication** | Counting non-independent measurements as independent. |
| **Sampling distribution** | How an estimate would vary across repeat experiments. |
| **Confidence interval** | A range of values consistent with the data; the *procedure* covers the truth 95% of the time. |
| **Null hypothesis** | The assumption that nothing is going on. |
| **p-value** | P(data this extreme \| nothing going on). Not P(nothing going on \| data). |
| **Type I error** | False positive; its rate is $\alpha$. |
| **Type II error** | False negative; its rate is $\beta$. |
| **Power** | $1-\beta$; the chance of detecting a real effect. |
| **Effect size** | How big the difference is - the thing a p-value does *not* tell you. |
| **FWER** | Probability of *any* false positive across many tests. |
| **FDR** | Expected *proportion* of your discoveries that are false. |
| **q-value** | A p-value adjusted for multiple testing under FDR. |
| **Outcome** | What you are explaining ($Y$). |
| **Predictor / covariate** | What you explain it with ($X$). |
| **Confounder** | Causes both predictor and outcome - **adjust for it**. |
| **Mediator** | Sits on the path from predictor to outcome - **do not adjust**. |
| **Collider** | Caused by both - **adjusting creates false associations**. |
| **Coefficient ($\beta$)** | Change in $Y$ per unit of $X$, holding others fixed. |
| **Residual** | What the model failed to explain. |
| **Randomisation** | Assigning treatment by chance; eliminates confounding by design. |


## 0.11 Where to go next

You now have the vocabulary for the rest of this document.

| If you want to... | Read | Run |
|---|---|---|
| See these ideas in code | - | `statsPy/foundations/F1`-`F6` |
| Understand why $n$ is the hardest question | [Topic 1](#topic-1-statistical-thinking-for-biological-experiments) | `01_study_design_and_estimands` |
| Go deeper on probability | [Topic 4](#topic-4-probability-models-and-sampling-distributions) | `04_probability_and_sampling` |
| Go deeper on intervals | [Topic 5](#topic-5-estimation-effect-sizes-and-confidence-intervals) | `05_estimation_and_intervals` |
| Go deeper on p-values | [Topic 6](#topic-6-hypothesis-tests-and-p-values) | `06_hypothesis_tests_and_pvalues` |
| Handle thousands of tests | [Topic 8](#topic-8-multiple-testing-and-selective-inference) | `08_multiple_testing` |
| Master eq. (0.16) | [Topic 11](#topic-11-linear-regression-as-the-core-framework) | `11_linear_models` |
| Reason about causes | [Topic 32](#topic-32-causal-inference-for-observational-bioinformatics) | `22_causal_inference` |

A reasonable first pass: read this Part, run `F1`-`F6`, then read Topic 1 and
run its module. Topic 1 is where the beginner material and the professional
material meet, and it is the topic that matters most.


# Part I: Foundations

## Topic 1: Statistical thinking for biological experiments

**Module:** `01_study_design_and_estimands`, **Prerequisite for:** everything

### The question

Before any formula: *what number, defined on what population, would answer the
biological question if we could measure everyone?* That number is the
**estimand**. Everything else - estimator, standard error, p-value - is
machinery for approximating it from a finite, noisy sample.

### Model

Let the population be a probability distribution $P$ over units. A study draws
$n$ units. For unit $i$ we observe an outcome $Y_i$, an exposure or treatment
$A_i \in \{0,1\}$, and covariates $\mathbf{Z}_i$.

Under the **potential-outcomes** notation, $Y_i(1)$ and $Y_i(0)$ are the
outcomes unit $i$ *would* show under treatment and control. Only one is ever
observed - the **fundamental problem of causal inference**:

$$Y_i = A_i Y_i(1) + (1 - A_i) Y_i(0). \tag{1.1}$$

### Key equations

**Average treatment effect (ATE).** The canonical experimental estimand:

$$\tau = \mathbb{E}\big[Y(1) - Y(0)\big]. \tag{1.2}$$

Randomisation makes $A \perp\!\!\!\perp \big(Y(1), Y(0)\big)$, which turns the
unobservable $(1.2)$ into an observable contrast of group means:

$$\tau = \mathbb{E}[Y \mid A=1] - \mathbb{E}[Y \mid A=0]. \tag{1.3}$$

This one line is why randomisation matters. Without it, $(1.3)$ estimates an
*association*, and the gap between $(1.2)$ and $(1.3)$ is confounding bias.

**Variance components and the unit of inference.** Suppose unit $i$ (a donor)
has true value $\mu_i$, and we take $m$ technical measurements $Y_{ij}$ of it:

$$Y_{ij} = \mu + b_i + e_{ij}, \qquad
b_i \sim (0, \sigma_b^2), \quad e_{ij} \sim (0, \sigma_e^2), \tag{1.4}$$

with $b_i$ (biological) and $e_{ij}$ (technical) independent. The mean of all
$nm$ observations, $\bar{Y}_{\cdot\cdot}$, has variance

$$\operatorname{Var}(\bar{Y}_{\cdot\cdot})
= \frac{\sigma_b^2}{n} + \frac{\sigma_e^2}{nm}. \tag{1.5}$$

**Read equation (1.5) carefully - it is the most consequential formula in this
entire document.** The biological variance term $\sigma_b^2/n$ is divided by
$n$, the number of *donors*, and is completely unaffected by $m$. As
$m \to \infty$ the second term vanishes but the first does not:

$$\lim_{m\to\infty}\operatorname{Var}(\bar{Y}_{\cdot\cdot}) = \frac{\sigma_b^2}{n}. \tag{1.6}$$

Sequencing one donor a million times, or profiling 50,000 cells from three
mice, buys you precision about *those three mice* and nothing about the
population of mice. Pseudoreplication is the error of putting $nm$ where
equation (1.5) demands $n$; the standard error is then too small by a factor of
roughly $\sqrt{1 + (m-1)\rho}$, where

$$\rho = \frac{\sigma_b^2}{\sigma_b^2 + \sigma_e^2} \tag{1.7}$$

is the **intraclass correlation (ICC)**. The quantity

$$\mathrm{DE} = 1 + (m-1)\rho \tag{1.8}$$

is the **design effect**: the factor by which clustering inflates the true
variance relative to the naive independent-sample formula. With $m = 500$ cells
per donor and a modest $\rho = 0.05$, $\mathrm{DE} \approx 25.95$, so naive
cell-level standard errors are about $\sqrt{25.95} \approx 5.1$ times too
small. That single number explains most false discoveries in early single-cell
differential-expression literature.[^sc1][^sc2]

**Effective sample size.** Clustering reduces $nm$ observations to

$$n_{\text{eff}} = \frac{nm}{1 + (m-1)\rho}. \tag{1.9}$$

With $n = 3$ donors, $m = 500$ cells, $\rho = 0.05$: $n_{\text{eff}} \approx 57.8$
- not 1500. And note it can never usefully exceed $n/\rho$.

### Assumptions

| Assumption | Meaning | What breaks it |
|---|---|---|
| SUTVA | One unit's treatment does not affect another's outcome | Shared media, cross-contamination, co-housed animals |
| Exchangeability | Treated and control units comparable before treatment | Non-random allocation, batch-confounded groups |
| Positivity | $0 < \Pr(A=1 \mid \mathbf{Z}) < 1$ | A covariate stratum in which everyone got treatment |
| Consistency | The observed $Y$ equals $Y(a)$ for the received $a$ | Ill-defined or heterogeneous "treatment" |

### Diagnostics

- Draw the data-generating hierarchy as a tree before writing code. Count the
  branch points, not the leaves; $n$ lives at the highest randomised level.
- Tabulate design factors against batch. If `table(condition, batch)` is
  block-diagonal, the effect is **unidentifiable**, and no software can fix it.
- Estimate the ICC from pilot data and compute (1.8) before trusting any
  cell-level or read-level p-value.

### Decision rules

1. Write the estimand as an equation before opening the data.
2. $n$ = the number of units that were independently randomised or independently
   sampled from the population of interest. Cells, reads, wells and technical
   replicates are *measurements of* units, not units.
3. If a factor was used to block or pair, it must appear in the model. Pairing
   that is designed in but analysed out throws away the precision you paid for.
4. Confounded designs are a design failure, not an analysis problem.


## Topic 2: Data structures and measurement scales

**Module:** `02_data_structures_and_scales`

### The question

Two matrices of the same shape can require entirely different likelihoods. What
does a *number* in this assay physically represent, and what constraints does
that impose?

### Key equations

**The assay triple.** Every omics dataset is a triple
$(\mathbf{Y}, \mathbf{R}, \mathbf{C})$ where $\mathbf{Y}$ is $G \times n$
(features x samples), $\mathbf{R}$ is $G \times q$ feature metadata, and
$\mathbf{C}$ is $n \times r$ sample metadata. The binding contract is

$$\text{colnames}(\mathbf{Y}) = \text{rownames}(\mathbf{C}), \qquad
\text{rownames}(\mathbf{Y}) = \text{rownames}(\mathbf{R}). \tag{2.1}$$

`SummarizedExperiment` exists to enforce (2.1) under subsetting and reordering;
a bare matrix plus a separate metadata frame does not.[^se1][^se2] Silent
violation of (2.1) - a sorted metadata table, a dropped sample - produces
analyses that run without error and are entirely wrong.

**Scale determines support, and support determines likelihood.**

| Measurement | Support | Natural model |
|---|---|---|
| Read counts | $\{0,1,2,\dots\}$ | Poisson / negative binomial |
| Counts out of a known total | $\{0,\dots,N\}$ | Binomial / beta-binomial |
| Log intensity | $\mathbb{R}$ | Normal |
| Methylation beta | $[0,1]$ | Beta, or logit-transformed normal |
| Relative abundance | Simplex $\mathcal{S}^{D-1}$ | Log-ratio / Dirichlet-multinomial |
| Time to event | $[0,\infty)$ with censoring | Survival / hazard model |
| Pathology grade | Ordered categories | Ordinal (proportional odds) |

**The three standard transformations, and exactly what they do.**

*Log with pseudocount* stabilises multiplicative noise:

$$y \mapsto \log_2(y + c). \tag{2.2}$$

The pseudocount $c$ is not cosmetic. For counts with mean $\mu$ and
negative-binomial variance $\mu + \phi\mu^2$, the delta method gives

$$\operatorname{Var}\!\big(\log_2(Y+c)\big) \approx
\frac{1}{(\ln 2)^2}\cdot\frac{\mu + \phi\mu^2}{(\mu+c)^2}, \tag{2.3}$$

which for large $\mu$ tends to $\phi/(\ln 2)^2$ - constant, which is the point
- but for small $\mu$ is dominated by $c$. Small $c$ inflates the variance of
low-count features; that is why filtering low counts and choosing $c$ are the
same decision.

*Logit (M-value)* maps a bounded proportion to the whole real line:

$$M = \log_2\!\left(\frac{\beta}{1-\beta}\right), \qquad \beta\in(0,1). \tag{2.4}$$

Methylation beta values are heteroscedastic - variance is compressed near 0 and
1 - which violates linear-model assumptions; M-values are approximately
homoscedastic and are the recommended scale for *modelling*, while beta values
remain the scale for *reporting* because they mean "fraction methylated".[^mm1]

*Inverse hyperbolic sine (arcsinh)* behaves like a linear function near zero and
a log far from it, which is what cytometry intensities need:

$$y \mapsto \operatorname{arcsinh}\!\left(\frac{y}{c}\right)
= \ln\!\left(\frac{y}{c} + \sqrt{\frac{y^2}{c^2} + 1}\right). \tag{2.5}$$

The cofactor $c$ sets the width of the linear region: $c = 5$ is conventional
for mass cytometry, $c = 150$ for fluorescence flow. Unlike $\log$, (2.5) is
defined for the negative values that compensation produces.

**Compositional closure.** If only relative abundance is observed, the data live
on the simplex

$$\mathcal{S}^{D-1} = \Big\{ (x_1,\dots,x_D) : x_d > 0, \ \sum_{d=1}^{D} x_d = 1 \Big\}. \tag{2.6}$$

The constraint in (2.6) has a hard consequence: the covariance of any closed
composition satisfies

$$\sum_{d=1}^{D}\operatorname{Cov}(x_c, x_d) = 0 \quad\text{for every } c, \tag{2.7}$$

so at least one covariance with every component must be negative. **Negative
correlations between taxa are guaranteed by arithmetic, not by ecology.** This
is developed in Topic 26.

### Decision rules

1. Ask what the number counts or measures before choosing a test.
2. Keep counts as counts for inference; use transformed values for
   visualisation, distances, and PCA.
3. Model on the scale where the variance is stable; report on the scale a
   biologist can interpret. These need not be the same (beta vs M is the
   canonical example).
4. Verify (2.1) programmatically at the top of every script. Assert, do not assume.


## Topic 3: Exploratory data analysis

**Module:** `03_exploratory_data_analysis`

### The question

Does the data match what the assay was supposed to produce - and is any
structure I see technical rather than biological?

### Key equations

**Order statistics and quantiles.** Sort the sample to get
$y_{(1)} \le \dots \le y_{(n)}$. The empirical CDF is

$$\hat{F}_n(t) = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{y_i \le t\}, \tag{3.1}$$

and the quantile function is its generalised inverse
$\hat{F}_n^{-1}(q) = \inf\{t : \hat{F}_n(t) \ge q\}$. The Dvoretzky-Kiefer-Wolfowitz
inequality bounds how far $\hat F_n$ can stray from the truth:

$$\Pr\Big(\sup_t |\hat{F}_n(t) - F(t)| > \epsilon\Big) \le 2e^{-2n\epsilon^2}. \tag{3.2}$$

Equation (3.2) is why an ECDF plot from $n = 6$ tells you very little about
distributional shape, and why "the data look non-normal" is a weak basis for
switching tests in small samples.

**Robust spread.** The median absolute deviation,

$$\mathrm{MAD} = \operatorname{median}_i\big(|y_i - \operatorname{median}(y)|\big), \tag{3.3}$$

is rescaled by $1.4826$ to be a consistent estimator of $\sigma$ under
normality, because $\Phi^{-1}(0.75) \approx 0.6745$ and $1/0.6745 \approx 1.4826$:

$$\hat\sigma_{\mathrm{MAD}} = 1.4826 \times \mathrm{MAD}. \tag{3.4}$$

MAD has a breakdown point of 50% (half the data can be arbitrarily corrupted
before it fails) versus 0% for the standard deviation. This is why MAD-based
outlier thresholds are the scverse-recommended default for single-cell QC:

$$\text{flag cell } i \ \text{if} \ |y_i - \operatorname{median}(y)| > k \cdot \hat\sigma_{\mathrm{MAD}}, \quad k \in \{3,5\}. \tag{3.5}$$

**The mean-variance relationship is the assay's fingerprint.** For each feature
$g$ compute $(\bar{y}_g, s_g^2)$ and plot. Then:

$$s_g^2 \approx \bar{y}_g \ \Rightarrow \text{Poisson}; \qquad
s_g^2 \approx \bar{y}_g + \phi \bar{y}_g^2 \ \Rightarrow \text{negative binomial}; \qquad
s_g^2 \approx \text{const} \ \Rightarrow \text{Gaussian-ready}. \tag{3.6}$$

Fitting a curve to (3.6) and reading off $\phi$ is exactly what `edgeR` and
`DESeq2` do internally, and what `voom` converts into precision weights.

**Skewness and kurtosis.** With $m_k = n^{-1}\sum_i (y_i - \bar y)^k$:

$$g_1 = \frac{m_3}{m_2^{3/2}}, \qquad g_2 = \frac{m_4}{m_2^{2}} - 3. \tag{3.7}$$

Under normality $\operatorname{Var}(g_1) \approx 6/n$, so with $n = 10$ the
standard error of skewness is about $0.77$ - a sample skewness of 1.0 is
entirely unremarkable noise. Do not "test for normality and then choose a test";
the test has no power when you need it and excessive power when you do not.

### Diagnostics worth running every time

| Plot | Detects |
|---|---|
| Library size / total intensity per sample | Failed or outlier libraries |
| Features detected vs library size | Depth-driven detection bias |
| Mean-variance scatter (log-log) | The correct likelihood family (3.6) |
| Sample-sample correlation heatmap | Swaps, batches, outlier specimens |
| PCA coloured by every metadata column | Which technical factor dominates |
| Missingness vs mean abundance | MNAR/left-censoring (Topic 15, 25) |
| Per-batch boxplots of a QC metric | Acquisition drift |

### Decision rules

1. Explore to diagnose data quality and choose a *likelihood*, not to choose a
   *hypothesis*. Choosing the hypothesis after seeing the outcome is HARKing and
   invalidates the p-value you later report.
2. Never delete an outlier without a documented, outcome-independent reason.
   Prefer robust methods or a reported sensitivity analysis (Topic 34).
3. Any preprocessing step that uses the outcome - filtering by group means,
   outcome-aware normalisation - must be either abandoned or moved inside the
   resampling loop (Topic 31).


## Topic 4: Probability models and sampling distributions

**Module:** `04_probability_and_sampling`

### The question

Where does a standard error come from? Every reported uncertainty is a
consequence of an assumed data-generating process; name it.

### Key equations

**Expectation and variance rules.** For constants $a, b$ and random $X, Y$:

$$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y], \tag{4.1}$$

$$\operatorname{Var}(aX + bY) = a^2\operatorname{Var}(X) + b^2\operatorname{Var}(Y) + 2ab\operatorname{Cov}(X,Y). \tag{4.2}$$

Equation (4.2) is the reason paired designs are powerful: for a difference
$D = Y_{\text{post}} - Y_{\text{pre}}$ with correlation $r$ between the members
of a pair,

$$\operatorname{Var}(D) = \sigma^2_{\text{post}} + \sigma^2_{\text{pre}} - 2r\,\sigma_{\text{post}}\sigma_{\text{pre}}
\;\xrightarrow{\ \sigma_{\text{pre}}=\sigma_{\text{post}}=\sigma\ }\; 2\sigma^2(1-r). \tag{4.3}$$

With $r = 0.8$, pairing cuts the variance of the estimated effect by a factor of
5 relative to an unpaired comparison of the same $2n$ measurements.

**The law of total variance** decomposes hierarchy:

$$\operatorname{Var}(Y) = \underbrace{\mathbb{E}\big[\operatorname{Var}(Y \mid Z)\big]}_{\text{within-group}}
+ \underbrace{\operatorname{Var}\big(\mathbb{E}[Y \mid Z]\big)}_{\text{between-group}}. \tag{4.4}$$

This is the formal statement behind (1.4)-(1.5) and behind every variance
component in a mixed model.

**Central limit theorem.** For i.i.d. $Y_i$ with mean $\mu$ and finite variance
$\sigma^2$,

$$\sqrt{n}\,\frac{\bar{Y}_n - \mu}{\sigma} \stackrel{d}{\to} \mathcal{N}(0,1). \tag{4.5}$$

The CLT is a statement about the *sampling distribution of the mean*, not about
the data. Normality of residuals is never required for the validity of a
large-sample t-interval; it is required for exactness in small samples.
The Berry-Esseen theorem quantifies the approximation error:

$$\sup_t \Big| \Pr\Big(\tfrac{\sqrt{n}(\bar Y_n - \mu)}{\sigma} \le t\Big) - \Phi(t)\Big|
\le \frac{C\,\mathbb{E}|Y - \mu|^3}{\sigma^3\sqrt{n}}, \quad C < 0.4748. \tag{4.6}$$

Convergence is $O(n^{-1/2})$ and degrades with skewness - which is precisely
the situation for low-count genes, cytokine concentrations, and rare taxa.

**The key discrete distributions.**

*Poisson*, the model for counts with no extra variability:

$$\Pr(Y = y) = \frac{\lambda^y e^{-\lambda}}{y!}, \quad
\mathbb{E}[Y] = \operatorname{Var}(Y) = \lambda. \tag{4.7}$$

*Negative binomial* as a gamma-Poisson mixture. If $Y \mid \Lambda \sim \text{Poisson}(\Lambda)$
and $\Lambda \sim \text{Gamma}(\text{shape}=1/\phi, \text{scale}=\mu\phi)$, then
marginally

$$\Pr(Y=y) = \frac{\Gamma(y + 1/\phi)}{\Gamma(1/\phi)\, y!}
\left(\frac{1}{1+\mu\phi}\right)^{1/\phi}\!\left(\frac{\mu\phi}{1+\mu\phi}\right)^{y}, \tag{4.8}$$

with the mean-variance relationship that defines all of count-based omics:

$$\mathbb{E}[Y] = \mu, \qquad \operatorname{Var}(Y) = \mu + \phi\mu^2. \tag{4.9}$$

Deriving (4.9) from (4.4) takes one line and is worth doing once:
$\operatorname{Var}(Y) = \mathbb{E}[\operatorname{Var}(Y\mid\Lambda)] + \operatorname{Var}(\mathbb{E}[Y\mid\Lambda])
= \mathbb{E}[\Lambda] + \operatorname{Var}(\Lambda) = \mu + \phi\mu^2$.
The parameter $\phi$ (the **dispersion**; `edgeR` reports $\sqrt{\phi}$ as the
biological coefficient of variation) captures donor-to-donor variability that
Poisson sampling cannot produce. Technical resequencing of one library is
Poisson; biological replicates are not.

*Binomial and beta-binomial.* For $y$ successes in $N$ trials with probability
$\pi$: $\operatorname{Var}(Y) = N\pi(1-\pi)$. Allowing $\pi \sim \text{Beta}(a,b)$
gives overdispersion with $\operatorname{Var}(Y) = N\pi(1-\pi)\big(1 + (N-1)\rho\big)$,
the same design-effect structure as (1.8). Bisulfite-sequencing methylation
counts and "positive cells out of total cells" both need this.

**Likelihood, score, information.** With log-likelihood $\ell(\theta)$:

$$U(\theta) = \frac{\partial \ell}{\partial\theta}, \qquad
\mathcal{I}(\theta) = -\mathbb{E}\!\left[\frac{\partial^2 \ell}{\partial\theta^2}\right], \tag{4.10}$$

and under regularity the MLE satisfies

$$\sqrt{n}(\hat\theta - \theta) \stackrel{d}{\to} \mathcal{N}\big(0, \mathcal{I}_1(\theta)^{-1}\big),
\qquad \widehat{\operatorname{Var}}(\hat\theta) \approx \mathcal{I}(\hat\theta)^{-1}. \tag{4.11}$$

Every Wald standard error printed by `glm`, `lm`, `coxph`, `DESeq2` and
`statsmodels` is (4.11) evaluated at the estimate. The **Cramér-Rao bound**
says no unbiased estimator can beat $\mathcal{I}(\theta)^{-1}$.

**Monte Carlo and permutation.** If $T$ is a statistic and $\pi$ ranges over a
group of label permutations under which the null makes the data exchangeable,
the exact p-value from $B$ random permutations is

$$\hat{p} = \frac{1 + \sum_{b=1}^{B}\mathbb{1}\{T(\pi_b(\mathbf{y})) \ge T_{\text{obs}}\}}{B + 1}. \tag{4.12}$$

The $+1$ in both numerator and denominator is not a fudge: it includes the
observed labelling in the reference set and is what makes $\hat p$ a valid
p-value ($\Pr(\hat p \le \alpha) \le \alpha$) for finite $B$. Omitting it can
produce $\hat p = 0$, which is never a valid p-value. The minimum attainable
p-value is $1/(B+1)$ - so $B \ge 999$ if you want to reach $0.001$.

### Decision rules

1. Name the distribution and the source of randomness before computing a
   standard error.
2. Check overdispersion: if $s^2 \gg \bar y$ for counts, Poisson inference will
   be anticonservative - often dramatically.
3. Prefer permutation when the design justifies exchangeability and $n$ is too
   small for (4.5); prefer likelihood when a parametric mean-variance model is
   credible and $G$ is large enough to borrow strength (Topic 33).


# Part II: Inference

## Topic 5: Estimation, effect sizes, and confidence intervals

**Module:** `05_estimation_and_intervals`

### The question

How large is the effect, in units a biologist can act on, and how precisely do
we know it? Contemporary statistical guidance is explicit that estimation, not
dichotomised testing, is the primary task.[^asa1][^asa2][^asa3]

### Key equations

**The bias-variance decomposition** of an estimator's mean squared error:

$$\mathrm{MSE}(\hat\theta) = \mathbb{E}\big[(\hat\theta - \theta)^2\big]
= \underbrace{\big(\mathbb{E}[\hat\theta] - \theta\big)^2}_{\text{bias}^2}
+ \underbrace{\operatorname{Var}(\hat\theta)}_{\text{variance}}. \tag{5.1}$$

Equation (5.1) licenses every shrinkage method in omics: `limma`'s moderated
variances, `DESeq2`'s `lfcShrink`, ridge regression and empirical-Bayes
dispersion estimation all accept a little bias to remove a lot of variance, and
by (5.1) that is a net win whenever the variance reduction exceeds the squared
bias introduced.

**Standard error of a mean, and the t-interval.**

$$\widehat{\operatorname{SE}}(\bar{Y}) = \frac{s}{\sqrt{n}}, \qquad
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(y_i - \bar{y})^2. \tag{5.2}$$

$$\bar{Y} \pm t_{1-\alpha/2,\,n-1}\cdot \frac{s}{\sqrt{n}}. \tag{5.3}$$

The $n-1$ in (5.2) is Bessel's correction: $\sum(y_i - \bar y)^2$ uses up one
degree of freedom estimating $\bar y$, so dividing by $n$ would underestimate
$\sigma^2$ by a factor $(n-1)/n$.

**Interpretation of a confidence interval.** A 95% CI is a procedure whose
intervals cover the true $\theta$ in 95% of repetitions. It is *not* a 95%
probability statement about $\theta$ given this dataset - that is the credible
interval of Topic 33. Operationally, a CI is the set of parameter values not
rejected at level $\alpha$:

$$\mathrm{CI}_{1-\alpha} = \{\theta_0 : p(\theta_0) > \alpha\}. \tag{5.4}$$

This **duality** in (5.4) is the cleanest way to see that a CI carries strictly
more information than the p-value for $\theta_0 = 0$.

**Standardised mean difference.** Cohen's $d$ with the pooled SD:

$$s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}, \qquad
d = \frac{\bar{y}_1 - \bar{y}_2}{s_p}. \tag{5.5}$$

$d$ is biased upward in small samples. Hedges' correction multiplies by

$$J(\nu) = \frac{\Gamma(\nu/2)}{\sqrt{\nu/2}\;\Gamma\big((\nu-1)/2\big)}
\approx 1 - \frac{3}{4\nu - 1}, \qquad \nu = n_1 + n_2 - 2, \tag{5.6}$$

giving $g = J(\nu)\,d$. With $n_1=n_2=4$ (so $\nu=6$), $J \approx 0.87$: an
uncorrected $d$ overstates the standardised effect by about 15%. Note the deeper caution: $d$ is a *ratio*, so a "large $d$" can mean a
small numerator over a tiny denominator. Always report the raw difference too.

**Effect measures for binary and time-to-event outcomes.** From a $2\times2$
table with cells $a,b,c,d$ (exposed/unexposed x event/no event):

$$\widehat{\mathrm{OR}} = \frac{ad}{bc}, \qquad
\widehat{\operatorname{Var}}\big(\log \widehat{\mathrm{OR}}\big)
= \frac{1}{a}+\frac{1}{b}+\frac{1}{c}+\frac{1}{d}, \tag{5.7}$$

$$\widehat{\mathrm{RR}} = \frac{a/(a+b)}{c/(c+d)}, \qquad
\widehat{\operatorname{Var}}\big(\log\widehat{\mathrm{RR}}\big)
= \frac{1}{a} - \frac{1}{a+b} + \frac{1}{c} - \frac{1}{c+d}. \tag{5.8}$$

Both are computed on the **log scale**, where the sampling distribution is far
closer to normal, and then exponentiated - never symmetrically on the ratio
scale. The hazard ratio (Topic 28) follows the same pattern.

**The delta method** is the engine behind (5.7)-(5.8). For a smooth $g$:

$$\operatorname{Var}\big(g(\hat\theta)\big) \approx \big[g'(\theta)\big]^2 \operatorname{Var}(\hat\theta). \tag{5.9}$$

Multivariate version, needed whenever you transform a vector of coefficients
(e.g. converting a logistic model's linear predictor to a probability):

$$\operatorname{Var}\big(g(\hat{\boldsymbol\theta})\big) \approx
\nabla g(\boldsymbol\theta)^\top \boldsymbol\Sigma \,\nabla g(\boldsymbol\theta). \tag{5.10}$$

**Log fold change.** In omics, effect size is almost always a log ratio:

$$\mathrm{LFC}_g = \log_2 \frac{\mu_{g,\text{treated}}}{\mu_{g,\text{control}}}
= \log_2 \mu_{g,\text{treated}} - \log_2 \mu_{g,\text{control}}. \tag{5.11}$$

A difference of coefficients on the log scale *is* a log fold change - which is
why GLMs with a log link (Topic 13) give differential expression for free.

**Bootstrap intervals.** Resample $n$ units with replacement $B$ times, recompute
$\hat\theta^{*(b)}$.

*Percentile:* $\big(\hat\theta^*_{(\alpha/2)},\ \hat\theta^*_{(1-\alpha/2)}\big)$.

*Basic (reverse percentile), which corrects for bias in the wrong direction from
what most people guess:*

$$\big(2\hat\theta - \hat\theta^*_{(1-\alpha/2)},\ \ 2\hat\theta - \hat\theta^*_{(\alpha/2)}\big). \tag{5.12}$$

*BCa*, adjusting for bias and skewness, uses

$$\hat{z}_0 = \Phi^{-1}\!\left(\frac{\#\{\hat\theta^{*(b)} < \hat\theta\}}{B}\right), \qquad
\hat{a} = \frac{\sum_i \big(\hat\theta_{(\cdot)} - \hat\theta_{(i)}\big)^3}
{6\Big[\sum_i \big(\hat\theta_{(\cdot)} - \hat\theta_{(i)}\big)^2\Big]^{3/2}}, \tag{5.13}$$

where $\hat\theta_{(i)}$ is the jackknife (leave-one-out) estimate, then reads
percentiles at adjusted levels
$\alpha_1 = \Phi\big(\hat z_0 + \frac{\hat z_0 + z_\alpha}{1 - \hat a(\hat z_0 + z_\alpha)}\big)$.

**Critical bootstrap rule for bioinformatics:** resample *experimental units*,
never cells, reads or repeated measurements. Resampling the wrong level
reproduces the pseudoreplication of (1.5) inside the resampling loop.

**Robust (sandwich) standard errors.** When the variance model is wrong but the
mean model is right:

$$\widehat{\operatorname{Var}}_{\mathrm{HC}}(\hat{\boldsymbol\beta})
= (\mathbf{X}^\top\mathbf{X})^{-1}
\left(\sum_{i=1}^{n} \omega_i\, e_i^2\, \mathbf{x}_i\mathbf{x}_i^\top\right)
(\mathbf{X}^\top\mathbf{X})^{-1}, \tag{5.14}$$

with $\omega_i = 1$ (HC0), $n/(n-p)$ (HC1), $1/(1-h_{ii})$ (HC2), or
$1/(1-h_{ii})^2$ (HC3). **Use HC3 when $n < 250$**: HC0 is badly
anticonservative in small samples, which is most bioinformatics experiments.

**Equivalence testing (TOST).** To claim "no meaningful difference", specify a
margin $\Delta$ and run two one-sided tests:

$$H_{01}: \theta \le -\Delta \quad\text{and}\quad H_{02}: \theta \ge +\Delta. \tag{5.15}$$

Reject both at level $\alpha$ $\iff$ the $100(1-2\alpha)\%$ CI lies entirely
within $(-\Delta, +\Delta)$. This is the *only* correct way to argue for
absence of an effect; $p > 0.05$ is not evidence of no effect.

### Decision rules

1. Report direction, magnitude, interval and the units. "Upregulated,
   $p = 0.003$" is not a result; "$\mathrm{LFC} = 1.8$, 95% CI $[1.1, 2.5]$,
   i.e. a 3.5-fold increase" is.
2. Build intervals on the scale where the sampling distribution is symmetric
   (log for ratios, Fisher $z$ for correlations, logit for proportions), then
   back-transform.
3. To claim equivalence, pre-specify $\Delta$ and use (5.15).
4. Bootstrap the experimental unit, and keep every learned preprocessing step
   inside the resample.


## Topic 6: Hypothesis tests and p-values

**Module:** `06_hypothesis_tests_and_pvalues`

### The question

What exactly is the probability that a p-value reports? Misinterpretation here
propagates into every downstream claim.[^asa1][^asa4]

### Key equations

**Definition.** Given a test statistic $T$ with observed value $t_{\text{obs}}$
and a null distribution $F_0$, the (upper-tail) p-value is

$$p = \Pr\big(T \ge t_{\text{obs}} \;\big|\; H_0 \text{ and all model assumptions}\big). \tag{6.1}$$

Three things follow immediately from (6.1) and are constantly forgotten:

- $p$ is conditional on $H_0$ **and the entire model**: distribution, independence,
  the design matrix, the filtering rule. A small $p$ indicts the *conjunction*,
  not just $H_0$.
- $p$ is *not* $\Pr(H_0 \mid \text{data})$. Reversing the conditioning is the
  prosecutor's fallacy.
- $p$ carries no information about magnitude. With $n$ large, any
  $\theta \ne 0$ gives $p \to 0$.

**Uniformity under the null.** If $T$ is continuous with null CDF $F_0$, then
$p = 1 - F_0(T)$ satisfies

$$p \mid H_0 \ \sim\ \mathrm{Uniform}(0,1). \tag{6.2}$$

Equation (6.2) is the foundation of all multiple-testing machinery in Topic 8,
and it is the most useful diagnostic you own: plot the histogram of all
$G$ p-values. Flat with a spike at 0 = healthy. A hump in the middle = wrong
null or correlated features. A slope toward 1 = conservative/discrete test. A
spike at 1 = a discreteness or filtering artefact.

For discrete statistics (small-count Fisher tests, permutation p-values),
$\Pr(p \le \alpha) \le \alpha$ with strict inequality - the test is
*conservative*, and lumping discrete with continuous p-values distorts FDR
estimation.

**Error rates.**

$$\alpha = \Pr(\text{reject} \mid H_0), \qquad
\beta = \Pr(\text{fail to reject} \mid H_1), \qquad
\text{power} = 1 - \beta. \tag{6.3}$$

**False positive risk.** The quantity researchers actually care about - the
probability that a "significant" result is a false alarm - depends on the prior
probability $\pi$ that $H_1$ is true:

$$\mathrm{PPV} = \Pr(H_1 \mid \text{reject})
= \frac{(1-\beta)\pi}{(1-\beta)\pi + \alpha(1-\pi)}. \tag{6.4}$$

Substitute a realistic screening scenario: $\pi = 0.10$, power $= 0.5$
(typical for underpowered biology), $\alpha = 0.05$. Then
$\mathrm{PPV} = (0.5)(0.1)/[(0.5)(0.1) + (0.05)(0.9)] = 0.05/0.095 = 0.526$.
**Nearly half of "significant" findings in that setting are false**, even with
perfect statistical practice. Equation (6.4) is why power (Topic 9) and
multiplicity (Topic 8) are not bureaucracy.

**Type S and Type M errors.** Under low power, conditioning on significance
selects for exaggerated estimates. Define, for true effect $\theta$:

$$\text{Type S} = \Pr\big(\operatorname{sign}(\hat\theta) \ne \operatorname{sign}(\theta) \;\big|\; |\hat\theta| > c\big), \qquad
\text{Type M} = \mathbb{E}\big[|\hat\theta| \;\big|\; |\hat\theta| > c\big] \big/ |\theta|. \tag{6.5}$$

The Type M ratio (the **winner's curse**) routinely exceeds 2-3 in underpowered
studies: published effect sizes are systematically too big, which is why
replication "fails" even when the effect is real.

**One-sided vs two-sided.** A one-sided test at $\alpha$ has the power of a
two-sided test at $2\alpha$ in the predicted direction and *zero* power against
the opposite direction. Choose the side from the design, before seeing data, or
do not use it.

**Combining independent p-values.** Fisher's method:

$$X^2 = -2\sum_{k=1}^{K}\log p_k \ \sim\ \chi^2_{2K} \quad\text{under } H_0. \tag{6.6}$$

Stouffer's weighted $z$ method is often preferable because it preserves direction:

$$Z = \frac{\sum_k w_k \Phi^{-1}(1 - p_k)}{\sqrt{\sum_k w_k^2}}, \qquad w_k \propto \sqrt{n_k}. \tag{6.7}$$

### Decision rules

1. Report the exact p-value alongside the estimate and interval. Never report
   only stars, and never report "$p = \text{n.s.}$".
2. "$p > 0.05$" means the data are compatible with $H_0$ - and also with many
   non-zero effects. Use (5.15) if you want to claim equivalence.
3. Decide sidedness, stopping rule, covariates, filtering and the multiplicity
   family **before** the analysis. Everything decided afterwards is exploratory.
4. Check the p-value histogram (6.2) before adjusting anything.


## Topic 7: t-tests, rank tests, and permutation tests

**Module:** `07_ttests_ranks_permutation`

### The question

Two groups differ - by how much, under which assumptions, and is a *t*-test even
a different thing from a regression? (It is not.)

### Key equations

**One-sample and paired.** With $d_i$ the within-pair differences:

$$t = \frac{\bar{d} - \mu_0}{s_d/\sqrt{n}} \ \sim\ t_{n-1} \quad \text{under } H_0. \tag{7.1}$$

A paired t-test *is* a one-sample t-test on differences. By (4.3) it is the
right analysis whenever pairing exists in the design, regardless of how the data
look.

**Pooled-variance two-sample:**

$$t = \frac{\bar{y}_1 - \bar{y}_2}{s_p\sqrt{\tfrac{1}{n_1} + \tfrac{1}{n_2}}}
\ \sim\ t_{n_1+n_2-2}, \tag{7.2}$$

with $s_p$ from (5.5). This assumes $\sigma_1^2 = \sigma_2^2$.

**Welch's test** drops that assumption:

$$t_W = \frac{\bar{y}_1 - \bar{y}_2}{\sqrt{\tfrac{s_1^2}{n_1} + \tfrac{s_2^2}{n_2}}}, \tag{7.3}$$

$$\nu = \frac{\left(\tfrac{s_1^2}{n_1} + \tfrac{s_2^2}{n_2}\right)^{2}}
{\dfrac{\left(s_1^2/n_1\right)^2}{n_1-1} + \dfrac{\left(s_2^2/n_2\right)^2}{n_2-1}}
\quad\text{(Welch-Satterthwaite).} \tag{7.4}$$

The df in (7.4) is generally non-integer and satisfies
$\min(n_1,n_2)-1 \le \nu \le n_1+n_2-2$. **Welch should be the default.** It
costs almost nothing when variances are equal and protects substantially when
they are not - especially under unequal group sizes, where the pooled test is
anticonservative if the smaller group has the larger variance. This is why
`t.test()` in R defaults to `var.equal = FALSE`. Do not "pre-test" variances
with Levene/F and then choose: the two-stage procedure has a distorted Type I
error rate.

**The t-test is a linear model.** Coding $x_i = \mathbb{1}\{\text{group 2}\}$,

$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \qquad
\hat\beta_1 = \bar{y}_2 - \bar{y}_1, \tag{7.5}$$

and the t-statistic for $H_0: \beta_1 = 0$ is *numerically identical* to (7.2).
The scripts verify this to machine precision. Once you see (7.5), ANOVA,
covariate adjustment, blocking, interactions and `limma` are all the same object.

**Mann-Whitney / Wilcoxon rank-sum.** Rank all $N = n_1+n_2$ observations; let
$R_1$ be the rank sum in group 1:

$$U_1 = R_1 - \frac{n_1(n_1+1)}{2}, \qquad
\mathbb{E}[U_1] = \frac{n_1 n_2}{2}, \qquad
\operatorname{Var}(U_1) = \frac{n_1 n_2 (N+1)}{12}. \tag{7.6}$$

The key interpretive identity - the statistic estimates a **probabilistic
index**, not a median difference:

$$\frac{U_1}{n_1 n_2} = \widehat{\Pr}(Y_1 > Y_2) + \tfrac{1}{2}\widehat{\Pr}(Y_1 = Y_2)
= \widehat{\mathrm{AUC}}. \tag{7.7}$$

Equation (7.7) says the Mann-Whitney statistic *is* the ROC area under the curve
(Topic 31). It tests medians **only** under the extra assumption of a pure
location shift with identical shapes. Under unequal variances or different
shapes, a significant Wilcoxon result can occur with identical medians. The
common claim "use Wilcoxon to compare medians" is false in general.

The rank-biserial correlation is a clean effect size:
$r_{rb} = 2U_1/(n_1n_2) - 1 \in [-1,1]$.

**Permutation test.** Under the null of exchangeability (guaranteed by
randomisation), every relabelling is equally likely, so (4.12) gives an exact
p-value. For paired data the exchangeable group is sign flips within pairs,
giving $2^n$ possible assignments. Permutation tests are **exact** for Type I
error regardless of the distribution - but they still assume exchangeability,
which batch effects and clustering destroy. Permuting labels across donors when
observations are cells is invalid for the same reason (1.5) is: it breaks the
dependence structure that the null must preserve.

**Choosing among them.**

| Situation | Use |
|---|---|
| Design is paired/blocked | Paired t or within-pair permutation |
| Approximately symmetric, unequal variance | Welch (7.3) |
| Estimand is a mean difference on a meaningful scale | t on that scale |
| Heavy tails, ordinal outcome, estimand is $\Pr(Y_1>Y_2)$ | Mann-Whitney (7.7) |
| Tiny $n$, randomised design | Permutation (4.12) |
| Many features, small $n$ | Moderated t / empirical Bayes (Topic 33) |

### Decision rules

1. Pairing comes from the design, never from a normality test.
2. Default to Welch for independent two-group comparisons.
3. Use a rank test when the *estimand* is rank-based, not as a reflex against
   non-normality - and then report (7.7), not a median difference.
4. Normality applies to the sampling distribution of the estimate, not to the
   raw values. With $n \ge 30$ per group and no extreme skew, (4.5) covers you.


## Topic 8: Multiple testing and selective inference

**Module:** `08_multiple_testing`

### The question

You tested $G = 20{,}000$ genes. Which error rate matches the claim you want to
make, and over exactly which family?

### Setting

Of $G$ hypotheses, $G_0$ are true nulls. After thresholding:

|  | Not rejected | Rejected | Total |
|---|---|---|---|
| True null | $U$ | $V$ (false positives) | $G_0$ |
| True alternative | $T$ | $S$ | $G - G_0$ |
| Total | $G - R$ | $R$ | $G$ |

### Key equations

**Family-wise error rate**: the probability of *even one* false positive:

$$\mathrm{FWER} = \Pr(V \ge 1). \tag{8.1}$$

**False discovery rate**: the expected *proportion* of rejections that are false:

$$\mathrm{FDR} = \mathbb{E}\!\left[\frac{V}{\max(R,1)}\right]. \tag{8.2}$$

The $\max(R,1)$ defines $V/R := 0$ when nothing is rejected. FDR is the right
target for discovery screens (you will follow up a list, and you can tolerate a
known fraction of duds); FWER is right for a small number of confirmatory
claims, each of which must stand alone.

**Bonferroni.** Reject $H_g$ if $p_g \le \alpha/G$. By the union bound,

$$\mathrm{FWER} \le \sum_{g \in \mathcal{H}_0}\Pr(p_g \le \alpha/G) \le G_0\frac{\alpha}{G} \le \alpha, \tag{8.3}$$

with **no independence assumption whatsoever**: its great virtue. Šidák,
$\alpha_{\text{Šid}} = 1-(1-\alpha)^{1/G}$, is slightly less conservative but
requires independence.

**Holm's step-down** is uniformly more powerful than Bonferroni and equally
assumption-free. Sort $p_{(1)} \le \dots \le p_{(G)}$; reject
$H_{(1)},\dots,H_{(k-1)}$ where

$$k = \min\Big\{ j : p_{(j)} > \frac{\alpha}{G - j + 1} \Big\}. \tag{8.4}$$

**There is never a reason to use Bonferroni instead of Holm** when you want
FWER: Holm rejects everything Bonferroni does, and sometimes more.

**Benjamini-Hochberg.** Find

$$k = \max\Big\{ j : p_{(j)} \le \frac{j}{G}\alpha \Big\} \tag{8.5}$$

and reject $H_{(1)},\dots,H_{(k)}$. Under independence or positive regression
dependence (PRDS - which covers the common case of features positively
correlated through shared biology),

$$\mathrm{FDR} \le \frac{G_0}{G}\alpha \le \alpha. \tag{8.6}$$

Equation (8.6) shows BH is *conservative by the factor* $\pi_0 = G_0/G$. If only
20% of genes are truly null, BH at $\alpha = 0.05$ actually delivers
$\mathrm{FDR} \approx 0.01$ - you are leaving discoveries on the table, which
motivates Storey's adaptive version below.

The adjusted p-values (what `p.adjust(method="BH")` returns) enforce monotonicity:

$$\tilde{p}_{(j)} = \min_{m \ge j}\left\{ \min\left( \frac{G}{m} p_{(m)},\ 1 \right) \right\}. \tag{8.7}$$

**Benjamini-Yekutieli** is valid under *arbitrary* dependence, at the cost of a
harmonic-number penalty:

$$p_{(j)} \le \frac{j}{G \cdot H_G}\alpha, \qquad H_G = \sum_{m=1}^{G}\frac{1}{m} \approx \ln G + 0.5772. \tag{8.8}$$

For $G = 20{,}000$, $H_G \approx 10.4$ - a tenfold price. Use BY only when
negative dependence is genuinely plausible; for expression data, BH is the
standard and defensible choice.

**Storey's q-value.** Estimate the null proportion from the flat part of the
p-value histogram:

$$\hat\pi_0(\lambda) = \frac{\#\{p_g > \lambda\}}{G(1-\lambda)}, \qquad \lambda \approx 0.5, \tag{8.9}$$

then

$$\hat{q}(p_{(j)}) = \min_{t \ge p_{(j)}} \frac{\hat\pi_0 \, G \, t}{\#\{p_m \le t\}}. \tag{8.10}$$

The q-value of a feature is the minimum FDR at which that feature would be
called. It gains power exactly when $\pi_0 < 1$.

**Local FDR** is the per-feature posterior null probability, the quantity you
actually want for a single gene:

$$\mathrm{fdr}(z) = \frac{\pi_0 f_0(z)}{f(z)}, \tag{8.11}$$

where $f = \pi_0 f_0 + (1-\pi_0)f_1$ is the observed mixture. FDR is the
*average* of $\mathrm{fdr}$ over the rejection region - so a gene sitting just
past the threshold has a local fdr much worse than the nominal 0.05.

**Independent filtering.** Removing features by a statistic $A$ that is
independent of the p-value *under the null* but correlated with power (e.g.
mean expression) increases discoveries without breaking FDR control, because
$\pi_0$ falls while null p-values stay uniform. Filtering by anything
outcome-dependent (a group-wise fold change, a t-statistic) **invalidates**
everything downstream.

**Defining the family.** This is a scientific decision, not a software default.
If you test 20,000 genes x 8 cell types x 3 contrasts, the number of hypotheses
is 480,000. Adjusting within each gene-list separately answers "among the genes
I called in this cell type and contrast, what fraction are wrong?" - a legitimate
but *different* claim from a global one. State which you mean.

### Decision rules

1. FDR (BH) for discovery screens; Holm for a handful of confirmatory claims.
2. Look at the p-value histogram (6.2) first. Adjusting a pathological histogram
   produces confident nonsense.
3. Filter only on outcome-independent statistics, and pre-specify the rule.
4. Rank the surviving list by effect size and its interval, not by adjusted
   p-value. A gene with $q = 0.001$ and $\mathrm{LFC} = 0.05$ is precise and
   uninteresting.


## Topic 9: Power, sample size, and design optimisation

**Module:** `09_power_and_sample_size`

### The question

Given a design, what effects could it reliably detect? Asked *before* the
experiment, this is power analysis. Asked afterwards, using the observed effect,
it is "post-hoc power" - a deterministic function of the p-value that adds no
information and should never be reported.

### Key equations

**Two-sample z-approximation.** For a difference $\delta$ with common SD
$\sigma$ and $n$ per group, the power of a two-sided level-$\alpha$ test is

$$1 - \beta = \Phi\!\left( \frac{\delta}{\sigma\sqrt{2/n}} - z_{1-\alpha/2} \right)
+ \Phi\!\left( \frac{-\delta}{\sigma\sqrt{2/n}} - z_{1-\alpha/2} \right), \tag{9.1}$$

whose second term is negligible unless power is very low. Inverting the first
term gives the sample size per group:

$$n = \frac{2\big(z_{1-\alpha/2} + z_{1-\beta}\big)^2 \sigma^2}{\delta^2}
 = \frac{2\big(z_{1-\alpha/2} + z_{1-\beta}\big)^2}{d^{2}}, \qquad d = \delta/\sigma. \tag{9.2}$$

**Lehr's rule of thumb.** For $\alpha = 0.05$ two-sided and 80% power,
$(1.96 + 0.8416)^2 \approx 7.85$, so

$$n \approx \frac{16}{d^2} \ \text{per group}. \tag{9.3}$$

$d = 1$ needs 16/group; $d = 0.5$ needs 64/group; $d = 0.2$ needs 400/group.
Commit (9.3) to memory - it instantly exposes the implausibility of "we found a
small but significant effect with $n = 4$".

**Exact t-based power** replaces the normal with a *non-central* $t$:

$$1-\beta = \Pr\big(|T| > t_{1-\alpha/2,\,\nu}\big), \qquad
T \sim t_{\nu}(\lambda), \quad \lambda = \frac{\delta}{\sigma}\sqrt{\frac{n}{2}}, \tag{9.4}$$

where $\lambda$ is the non-centrality parameter. (9.2) underestimates $n$ by 1-2
per group at small sizes; use (9.4) or `pwr::pwr.t.test` for real planning.

**Paired designs.** Substituting (4.3), the effective SD becomes
$\sigma\sqrt{2(1-r)}$, so the required number of *pairs* is

$$n_{\text{pairs}} = \frac{\big(z_{1-\alpha/2}+z_{1-\beta}\big)^2 \cdot 2(1-r)\sigma^2}{\delta^2}. \tag{9.5}$$

At $r = 0.7$, pairing needs 30% of the subjects of the unpaired design.

**Clustered designs.** Multiply by the design effect (1.8):

$$n_{\text{clustered}} = n_{\text{indep}} \times \big[1 + (m-1)\rho\big]. \tag{9.6}$$

The optimal allocation question for single-cell studies falls straight out of
(1.5): with cost $C = n(c_{\text{donor}} + m\,c_{\text{cell}})$, the variance
$\sigma_b^2/n + \sigma_e^2/(nm)$ is minimised subject to fixed cost at

$$m^{*} = \sqrt{\frac{\sigma_e^2}{\sigma_b^2}\cdot\frac{c_{\text{donor}}}{c_{\text{cell}}}}. \tag{9.7}$$

Equation (9.7) is the formal answer to "more donors or more cells per donor?"
Because $\sigma_b^2$ (donor variability) is typically large in human studies,
$m^*$ is modest and **more donors nearly always wins**.

**Power for negative-binomial counts.** For log fold change $\lambda = \log(\mathrm{FC})$,
mean count $\mu$, dispersion $\phi$, $n$ per group, the variance of the estimated
log fold change is approximately

$$\operatorname{Var}(\hat\lambda) \approx \frac{2}{n}\left(\frac{1}{\mu} + \phi\right), \tag{9.8}$$

so

$$n \approx \frac{2\big(z_{1-\alpha/2} + z_{1-\beta}\big)^2 \left(\tfrac{1}{\mu} + \phi\right)}{\lambda^2}. \tag{9.9}$$

Equation (9.9) contains the entire depth-versus-replicates argument: increasing
sequencing depth raises $\mu$ and shrinks $1/\mu$, but **cannot touch $\phi$**.
Once $\mu \gg 1/\phi$, extra reads buy nothing and only extra samples help. With
a typical human $\phi \approx 0.16$ (BCV $= 0.4$), the floor is reached around
$\mu \approx 6$ counts - which most expressed genes clear easily.

**Multiplicity burden.** Genome-wide testing requires replacing $\alpha$ with the
effective threshold. For FWER control, $\alpha/G$; substituting into (9.2), the
required $n$ grows like $(z_{1-\alpha/(2G)} + z_{1-\beta})^2$, which is only
*logarithmic* in $G$ - a relief. For $G = 10^6$ (GWAS, $\alpha = 5\times10^{-8}$),
$z \approx 5.45$ versus $1.96$, so $n$ must grow by a factor of about
$\big((5.45+0.84)/(1.96+0.84)\big)^2 \approx 5.0$.

**Simulation-based power** is the general tool, and the only honest one for
mixed models, GLMMs and multi-stage pipelines:

$$\widehat{\text{power}} = \frac{1}{B}\sum_{b=1}^{B}\mathbb{1}\{\text{analysis of simulated dataset } b \text{ rejects}\}. \tag{9.10}$$

Simulate from the *assumed* generative model, run the *exact* analysis pipeline
you will use (including filtering and multiplicity), and count rejections. Its
Monte Carlo SE is $\sqrt{\hat\pi(1-\hat\pi)/B}$, so $B = 1000$ gives about
$\pm 1.5\%$ - adequate.

### Decision rules

1. Power analysis is prospective. Report the *minimum detectable effect* for the
   design you actually ran, not post-hoc power.
2. State every input ($\delta$, $\sigma$ or $\phi$, $\rho$, $\alpha$, $G$) and
   show a sensitivity curve, because pilot variance estimates are optimistic and
   biased downward.
3. Spend on biological replicates before depth, cells, or technical replicates -
   equations (1.6), (9.7) and (9.9) all say the same thing.
4. For any design with clustering or a non-standard model, simulate (9.10).


# Part III: Association and models

## Topic 10: Correlation and dependence

**Module:** `10_correlation_and_dependence`

### The question

Two measurements move together. On what scale, under what dependence structure,
and does "together" mean linear, monotone, or merely non-independent?

### Key equations

**Covariance and Pearson correlation.**

$$\operatorname{Cov}(X,Y) = \mathbb{E}\big[(X-\mu_X)(Y-\mu_Y)\big], \qquad
\rho = \frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y} \in [-1,1]. \tag{10.1}$$

$$\hat\rho = r = \frac{\sum_i (x_i-\bar x)(y_i - \bar y)}
{\sqrt{\sum_i (x_i - \bar x)^2}\sqrt{\sum_i (y_i-\bar y)^2}}. \tag{10.2}$$

$r$ measures **linear** association only. $r = 0$ does not imply independence
($Y = X^2$ with symmetric $X$ gives $r = 0$), and $r$ is not invariant to
monotone transformation - $r(\log x, \log y) \ne r(x,y)$. Since log
transformation is ubiquitous in omics, always state the scale.

**Fisher's z-transform** makes the sampling distribution approximately normal
with a variance that does not depend on $\rho$:

$$z = \operatorname{artanh}(r) = \tfrac12\ln\frac{1+r}{1-r},
\qquad z \ \dot\sim\ \mathcal{N}\!\left(\operatorname{artanh}(\rho), \frac{1}{n-3}\right). \tag{10.3}$$

Build the CI for $z$ using $\pm z_{1-\alpha/2}/\sqrt{n-3}$ and map back with
$r = \tanh(z)$. Never build a symmetric interval on $r$ directly - the interval
would escape $[-1,1]$.

**Spearman** is Pearson on ranks; **Kendall's** $\tau$ counts concordant ($C$)
minus discordant ($D$) pairs:

$$\tau = \frac{C - D}{\binom{n}{2}}, \qquad
\operatorname{Var}_0(\tau) = \frac{2(2n+5)}{9n(n-1)}. \tag{10.4}$$

$\tau$ has a direct probabilistic reading:
$\tau = \Pr(\text{concordant}) - \Pr(\text{discordant})$. It handles ties better
than Spearman and is preferable for small $n$ and ordinal data.

**Partial correlation**: association after linearly removing $Z$:

$$r_{XY\cdot Z} = \frac{r_{XY} - r_{XZ}r_{YZ}}{\sqrt{(1-r_{XZ}^2)(1-r_{YZ}^2)}}. \tag{10.5}$$

The multivariate generalisation is the **precision matrix**
$\boldsymbol\Omega = \boldsymbol\Sigma^{-1}$:

$$r_{ij \cdot \text{rest}} = \frac{-\Omega_{ij}}{\sqrt{\Omega_{ii}\Omega_{jj}}}. \tag{10.6}$$

Equation (10.6) is the basis of Gaussian graphical models (Topic 30): a zero in
the *precision* matrix means conditional independence, whereas a zero in the
covariance matrix means marginal independence. These are very different claims.

**Simpson's paradox and repeated measures.** Pooling correlated observations
from the same donors can *reverse* the sign of an association. The
repeated-measures correlation removes between-subject variation by fitting a
common within-subject slope:

$$r_{rm} = \operatorname{sign}(\hat\beta)\sqrt{\frac{SS_{\text{model}}}{SS_{\text{model}} + SS_{\text{error}}}}
\quad \text{from } \ y \sim x + \text{factor(subject)}. \tag{10.7}$$

**Correlation is not agreement.** Two assays can correlate at $r = 0.99$ and
disagree by a constant factor of two. Use Bland-Altman limits of agreement on
the differences $d_i = y_{1i} - y_{2i}$:

$$\mathrm{LoA} = \bar d \pm 1.96\, s_d, \tag{10.8}$$

and Lin's concordance correlation coefficient, which penalises both scatter and
systematic shift:

$$\rho_c = \frac{2\sigma_{XY}}{\sigma_X^2 + \sigma_Y^2 + (\mu_X - \mu_Y)^2}. \tag{10.9}$$

Note $\rho_c = \rho$ only when means and variances match exactly; otherwise
$\rho_c < \rho$.

**Spurious correlation from closure.** For compositional data, apply
$\log$-ratios (Topic 26) before correlating. As a concrete demonstration: draw
three *independent* absolute abundances and close them to proportions; the
resulting proportions show strong negative correlations that were not present in
the absolute data. Equation (2.7) guarantees it.

**Correlation under multiplicity.** A $G \times G$ correlation matrix contains
$G(G-1)/2$ tests. For $G = 1000$ that is 499,500 hypotheses; at $\alpha = 0.05$
you expect ~25,000 false positives. Co-expression networks built from unadjusted
correlations are mostly noise. Shrinkage estimators (Ledoit-Wolf) stabilise the
matrix:

$$\hat{\boldsymbol\Sigma}_{\text{shrunk}} = (1-\lambda)\mathbf{S} + \lambda \mathbf{T}, \tag{10.10}$$

with $\mathbf{T}$ a structured target (e.g. $\bar{s}^2\mathbf{I}$) and $\lambda$
chosen to minimise expected squared error. This is essential when $G > n$, where
$\mathbf{S}$ is singular and $\mathbf{S}^{-1}$ in (10.6) does not exist.

### Decision rules

1. Pearson for linear association on a justified scale; Spearman/Kendall for
   monotone association or ordinal data; distance correlation when you need to
   detect any dependence.
2. Handle donor structure explicitly - (10.7) or a mixed model - before
   interpreting a correlation across repeated measurements.
3. Use Fisher's $z$ (10.3) for intervals and tests.
4. For method comparison, report (10.8)-(10.9), not $r$.
5. Apply log-ratio transformation before correlating compositions.


## Topic 11: Linear regression as the core framework

**Module:** `11_linear_models`

### The question

Almost every classical method in this curriculum is $\mathbf{y} = \mathbf{X}\boldsymbol\beta + \boldsymbol\varepsilon$
with a different $\mathbf{X}$. Learn the algebra once.

### Model

$$\mathbf{y} = \mathbf{X}\boldsymbol\beta + \boldsymbol\varepsilon, \qquad
\mathbb{E}[\boldsymbol\varepsilon] = \mathbf{0}, \quad
\operatorname{Var}(\boldsymbol\varepsilon) = \sigma^2\mathbf{I}_n. \tag{11.1}$$

### Key equations

**Least squares.** Minimising $\|\mathbf{y}-\mathbf{X}\boldsymbol\beta\|^2$ gives
the normal equations $\mathbf{X}^\top\mathbf{X}\boldsymbol\beta = \mathbf{X}^\top\mathbf{y}$, hence

$$\hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}. \tag{11.2}$$

Geometrically, $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ is the orthogonal
projection of $\mathbf{y}$ onto the column space of $\mathbf{X}$, where

$$\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \tag{11.3}$$

is the **hat matrix**: symmetric, idempotent ($\mathbf{H}^2=\mathbf{H}$), with
$\operatorname{tr}(\mathbf{H}) = p$. In practice never invert
$\mathbf{X}^\top\mathbf{X}$ numerically; use the QR or SVD decomposition, which
is what `lm()` and `numpy.linalg.lstsq` do. Forming $\mathbf{X}^\top\mathbf{X}$
squares the condition number and destroys precision on collinear designs.

**Sampling distribution.**

$$\operatorname{Var}(\hat{\boldsymbol\beta}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1},
\qquad \hat\sigma^2 = \frac{\|\mathbf{y}-\hat{\mathbf{y}}\|^2}{n-p} = \frac{\mathrm{RSS}}{n-p}. \tag{11.4}$$

$$t_j = \frac{\hat\beta_j}{\widehat{\operatorname{SE}}(\hat\beta_j)} \ \sim\ t_{n-p}
\quad\text{under } H_0:\beta_j = 0. \tag{11.5}$$

By the **Gauss-Markov theorem**, $\hat{\boldsymbol\beta}$ is the best linear
unbiased estimator under (11.1) - note that this requires neither normality nor
any distributional assumption beyond the first two moments. Normality is needed
only for the *exactness* of (11.5) in small samples.

**Interpretation of a coefficient.** $\beta_j$ is the expected change in $y$ per
unit change in $x_j$ **holding all other predictors in the model fixed**. Change
the covariate set and $\beta_j$ changes meaning - this is not instability, it is
a different estimand. It is also why adding covariates without a causal argument
(Topic 32) is not "being careful".

**Nested-model comparison (partial F).**

$$F = \frac{(\mathrm{RSS}_0 - \mathrm{RSS}_1)/(p_1 - p_0)}{\mathrm{RSS}_1/(n-p_1)}
\ \sim\ F_{p_1-p_0,\; n-p_1}. \tag{11.6}$$

**Variance explained.**

$$R^2 = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}}, \qquad
R^2_{\text{adj}} = 1 - \frac{\mathrm{RSS}/(n-p)}{\mathrm{TSS}/(n-1)}. \tag{11.7}$$

$R^2$ never decreases when predictors are added, so it cannot be used for model
selection; $R^2_{\text{adj}}$ penalises $p$ but is still not a validation metric
(Topic 31).

**Two intervals that are constantly confused.** For a new point $\mathbf{x}_0$:

$$\text{CI for the mean: } \hat{y}_0 \pm t_{1-\alpha/2,n-p}\,\hat\sigma\sqrt{\mathbf{x}_0^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_0}, \tag{11.8}$$

$$\text{PI for an observation: } \hat{y}_0 \pm t_{1-\alpha/2,n-p}\,\hat\sigma\sqrt{1 + \mathbf{x}_0^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_0}. \tag{11.9}$$

The extra $1$ in (11.9) is the irreducible noise of a single new observation.
The prediction interval does **not** shrink to zero as $n \to \infty$.

**Leverage, residuals, influence.**

$$h_{ii} = \mathbf{x}_i^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_i,
\qquad r_i = \frac{e_i}{\hat\sigma\sqrt{1-h_{ii}}}, \tag{11.10}$$

$$D_i = \frac{r_i^2}{p}\cdot\frac{h_{ii}}{1-h_{ii}}
\quad\text{(Cook's distance).} \tag{11.11}$$

Rules of thumb: flag $h_{ii} > 2p/n$ and $D_i > 4/n$. Note the two are distinct -
high leverage means unusual *predictors*, large residual means unusual
*outcome*; influence requires both.

**Collinearity.**

$$\mathrm{VIF}_j = \frac{1}{1 - R_j^2}, \tag{11.12}$$

where $R_j^2$ is from regressing $x_j$ on the other predictors. $\mathrm{VIF}_j$
multiplies $\operatorname{Var}(\hat\beta_j)$. VIF $> 10$ ($R_j^2 > 0.9$) is
serious. Perfect collinearity makes $\mathbf{X}^\top\mathbf{X}$ singular - this
is exactly what happens when batch is confounded with condition, and it is the
algebraic form of "the design cannot answer the question".

**Nonlinearity via basis expansion.** Linear models are linear *in the
parameters*, not in the covariates. With a spline basis $\{B_k\}$:

$$\mathbb{E}[y \mid x] = \beta_0 + \sum_{k=1}^{K}\beta_k B_k(x). \tag{11.13}$$

Natural cubic splines constrain the fit to be linear beyond the boundary knots,
which controls the wild tail behaviour of unrestricted polynomials. This is the
right way to model dose-response and time courses (Topic 28).

**Centering changes interpretation, not fit.** In $y = \beta_0 + \beta_1 x_1 +
\beta_2 x_2 + \beta_3 x_1x_2$, $\beta_1$ is the effect of $x_1$ **when
$x_2 = 0$**. If $x_2 = 0$ is outside the observed range, $\beta_1$ is an
extrapolation. Centering $x_2$ at its mean makes $\beta_1$ the effect at the
average $x_2$ - usually what you meant.

### Assumptions, in order of how much they matter

1. **Independence**: violation is catastrophic and invisible in residual plots.
   Fix by design or with Topic 14.
2. **Correct mean structure (linearity)**: biases the estimate itself.
3. **Homoscedasticity**: biases standard errors; fix with (5.14) or weights.
4. **Normality of residuals**: least important; the CLT (4.5) handles it for
   inference on $\hat\beta$ at moderate $n$.

### Diagnostics

`plot(lm_fit)` gives all four you need: residuals vs fitted (linearity),
Q-Q (normality), scale-location (heteroscedasticity), residuals vs leverage with
Cook's contours (influence). Inspect the design matrix with `qr(X)$rank` before
fitting $G$ feature-wise models - a rank-deficient design fails identically for
all 20,000 genes.

### Decision rules

1. Write the design matrix and check its rank first.
2. Choose covariates from a causal diagram (Topic 32), not from stepwise
   selection, which invalidates all subsequent p-values.
3. Centre continuous predictors involved in interactions.
4. Prefer robust SEs (5.14) over dropping data when variance is non-constant.


## Topic 12: ANOVA, factorial designs, and contrasts

**Module:** `12_anova_and_contrasts`

### The question

With three or more groups, which comparisons actually encode the biology? The
omnibus "are they all equal?" is rarely the scientific question.

### Key equations

**The sum-of-squares decomposition.** With $k$ groups, $n_j$ per group,
$N = \sum n_j$:

$$\underbrace{\sum_{j}\sum_{i}(y_{ij}-\bar{y}_{\cdot\cdot})^2}_{\mathrm{SS}_{\text{total}}}
= \underbrace{\sum_j n_j(\bar{y}_j - \bar{y}_{\cdot\cdot})^2}_{\mathrm{SS}_{\text{between}}}
+ \underbrace{\sum_j\sum_i (y_{ij}-\bar{y}_j)^2}_{\mathrm{SS}_{\text{within}}}. \tag{12.1}$$

This is Pythagoras in $\mathbb{R}^N$: the two right-hand terms are orthogonal
projections. The omnibus test is their ratio of mean squares:

$$F = \frac{\mathrm{SS}_{\text{between}}/(k-1)}{\mathrm{SS}_{\text{within}}/(N-k)}
\ \sim\ F_{k-1,\,N-k}. \tag{12.2}$$

For $k=2$, $F = t^2$ exactly - ANOVA is the linear model of (7.5) again.

**Contrasts** are where the science lives. A contrast is a weighted combination
of group means with weights summing to zero:

$$L = \sum_{j=1}^{k} c_j \mu_j, \quad \sum_j c_j = 0; \qquad
\hat{L} = \sum_j c_j \bar{y}_j, \quad
\widehat{\operatorname{Var}}(\hat L) = \hat\sigma^2\sum_j \frac{c_j^2}{n_j}. \tag{12.3}$$

Then $t = \hat L/\widehat{\operatorname{SE}}(\hat L) \sim t_{N-k}$. The
zero-sum requirement is what makes $L$ invariant to the grand mean, so it
estimates a *difference* rather than a level.

Useful contrast vectors for $k=4$ ordered dose groups
$(0, 1, 2, 3)$:

| Purpose | $\mathbf{c}$ |
|---|---|
| Treated vs control | $(-1, 1/3, 1/3, 1/3)$ |
| Highest vs lowest | $(-1, 0, 0, 1)$ |
| Linear trend | $(-3, -1, 1, 3)$ |
| Quadratic trend | $(1, -1, -1, 1)$ |

Two contrasts are **orthogonal** when $\sum_j c_j d_j / n_j = 0$; orthogonal
contrasts partition $\mathrm{SS}_{\text{between}}$ into independent pieces and
their tests are uncorrelated.

**Coding schemes determine what the intercept means.**

- *Treatment (dummy) coding*, R's default: $\beta_0 = \mu_{\text{reference}}$
  and $\beta_j = \mu_j - \mu_{\text{reference}}$.
- *Sum-to-zero coding*: $\beta_0 = \bar\mu$ (unweighted grand mean) and
  $\beta_j = \mu_j - \bar\mu$. Required for Type III SS to be interpretable.
- *Helmert*: each level vs the mean of preceding levels.

The fitted values are identical under all codings; only the parameter meanings
change. Reporting "the effect of treatment" without stating the coding is
ambiguous.

**Factorial designs and interaction.** For factors $A$ (levels $a$) and $B$
(levels $b$):

$$\mathbb{E}[y_{ijk}] = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij}. \tag{12.4}$$

The interaction is a *difference of differences*:

$$(\alpha\beta)_{ij} \ \text{captures} \
\big(\mu_{11} - \mu_{12}\big) - \big(\mu_{21} - \mu_{22}\big). \tag{12.5}$$

**When an interaction is present, a "main effect" is an average over the levels
of the other factor**, which may correspond to no real condition. Report
simple (conditional) effects instead.

The efficiency argument for factorials: a $2\times2$ design with $N$ total units
estimates both main effects with the *same* precision as two separate
$N/2$-unit one-factor experiments - and gets the interaction for free.

**Types of sums of squares.** For unbalanced designs the order of terms matters:

- **Type I (sequential):** $\mathrm{SS}(A)$, then $\mathrm{SS}(B\mid A)$, then
  $\mathrm{SS}(AB \mid A,B)$. Depends on order. Appropriate when there is a
  genuine hierarchy (e.g. block before treatment).
- **Type II:** each main effect adjusted for the other main effects but not
  interactions. Most powerful when interactions are absent.
- **Type III:** each term adjusted for all others including interactions.
  Requires sum-to-zero contrasts to be meaningful; with treatment coding it
  tests hypotheses nobody wants.

The honest framing: stop arguing about SS types and instead write down the two
models you want to compare with (11.6), plus the contrasts you care about.

**Effect sizes.** $\eta^2 = \mathrm{SS}_{\text{between}}/\mathrm{SS}_{\text{total}}$
is biased upward; $\omega^2$ corrects it:

$$\omega^2 = \frac{\mathrm{SS}_{\text{between}} - (k-1)\mathrm{MS}_{\text{within}}}
{\mathrm{SS}_{\text{total}} + \mathrm{MS}_{\text{within}}}. \tag{12.6}$$

**Post-hoc multiplicity.** Tukey's HSD controls FWER across all $\binom{k}{2}$
pairwise comparisons using the studentised range distribution $q_{k,N-k}$:

$$\bar{y}_i - \bar{y}_j \pm \frac{q_{1-\alpha;\,k,\,N-k}}{\sqrt{2}}\,
\hat\sigma\sqrt{\frac{1}{n_i}+\frac{1}{n_j}}. \tag{12.7}$$

Dunnett's test is more powerful when all comparisons are against a single
control. For a handful of pre-specified contrasts, Holm (8.4) is simplest and
valid.

### Decision rules

1. Pre-specify contrasts. An omnibus F followed by all-pairs testing is rarely
   the best use of the data.
2. Check the interaction before interpreting main effects.
3. For ordered doses, a trend contrast has far more power than all-pairs tests.
4. Balance your design when you can; it makes contrasts orthogonal and makes the
   SS-type question disappear.


## Topic 13: Generalised linear models

**Module:** `13_generalized_linear_models`

### The question

The outcome is a count, a proportion, or a binary indicator. Its variance
depends on its mean and its range is bounded. How do we keep the linear-model
machinery?

### Model

A GLM has three components:

1. **Random:** $Y_i$ from an exponential-family distribution,

$$f(y;\theta,\varphi) = \exp\!\left\{\frac{y\theta - b(\theta)}{a(\varphi)} + c(y,\varphi)\right\}, \tag{13.1}$$

with the two facts that make everything work:

$$\mathbb{E}[Y] = b'(\theta) = \mu, \qquad
\operatorname{Var}(Y) = b''(\theta)\,a(\varphi) = V(\mu)\,a(\varphi). \tag{13.2}$$

2. **Systematic:** $\eta_i = \mathbf{x}_i^\top\boldsymbol\beta$.
3. **Link:** $g(\mu_i) = \eta_i$.

| Family | $V(\mu)$ | Canonical link | Coefficient means |
|---|---|---|---|
| Gaussian | $1$ | identity | mean difference |
| Binomial | $\mu(1-\mu)$ | logit | log odds ratio |
| Poisson | $\mu$ | log | log rate ratio |
| Neg. binomial | $\mu + \phi\mu^2$ | log | log fold change |
| Gamma | $\mu^2$ | inverse (log in practice) | log ratio of means |

### Key equations

**Fitting: iteratively reweighted least squares.** Define the working response
and weights

$$z_i = \eta_i + (y_i - \mu_i)\,g'(\mu_i), \qquad
w_i = \frac{1}{g'(\mu_i)^2 V(\mu_i)}, \tag{13.3}$$

and iterate weighted least squares to convergence:

$$\boldsymbol\beta^{(t+1)} = (\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{W}\mathbf{z}. \tag{13.4}$$

Equation (13.4) is Fisher scoring, and it shows a GLM is a *weighted* linear
model applied to a locally linearised response. At convergence,

$$\widehat{\operatorname{Var}}(\hat{\boldsymbol\beta}) = a(\varphi)(\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}. \tag{13.5}$$

**Logistic regression.**

$$\log\frac{\pi_i}{1-\pi_i} = \mathbf{x}_i^\top\boldsymbol\beta
\iff \pi_i = \frac{1}{1 + e^{-\mathbf{x}_i^\top\boldsymbol\beta}}. \tag{13.6}$$

$e^{\beta_j}$ is an **odds ratio** per unit increase in $x_j$, conditional on the
other covariates. Two warnings: (i) odds ratios are *not* risk ratios unless the
outcome is rare - for a 40% baseline risk, an OR of 2 is a risk ratio of about
1.43; (ii) unlike linear models, logistic coefficients are **non-collapsible**:
$\beta_j$ changes when you add a covariate even if that covariate is not a
confounder. Do not read a shift in the OR as evidence of confounding.

**Separation.** If a covariate perfectly predicts the outcome, the MLE diverges
($|\hat\beta| \to \infty$) and the SE explodes. This is common with small $n$ and
rare outcomes. The fix is **Firth's penalised likelihood**, which adds the
Jeffreys prior:

$$\ell^{*}(\boldsymbol\beta) = \ell(\boldsymbol\beta) + \tfrac12\log\big|\mathcal{I}(\boldsymbol\beta)\big|, \tag{13.7}$$

removing the first-order bias and always producing finite estimates.

**Poisson regression with an offset.** Counts observed over different exposures
(library size, callable bases, person-time, total cells) model a *rate*:

$$\log\mathbb{E}[Y_i] = \log t_i + \mathbf{x}_i^\top\boldsymbol\beta. \tag{13.8}$$

The $\log t_i$ term has a **fixed coefficient of 1**: that is what makes it an
offset rather than a covariate, and it is what converts counts into rates.
Putting $\log t_i$ in as a free covariate answers a different question.

**Overdispersion.** Estimate it from the Pearson statistic:

$$\hat\varphi = \frac{1}{n-p}\sum_{i=1}^{n}\frac{(y_i-\hat\mu_i)^2}{V(\hat\mu_i)}. \tag{13.9}$$

If $\hat\varphi \gg 1$, Poisson standard errors are too small by a factor
$\sqrt{\hat\varphi}$. Two principled responses:

- **Quasi-Poisson:** keep $\hat\mu$, multiply all variances by $\hat\varphi$.
  Variance is $\varphi\mu$ - linear in the mean.
- **Negative binomial:** a genuine likelihood with variance $\mu + \phi\mu^2$ -
  quadratic in the mean. This is what RNA-seq needs, and why `edgeR`/`DESeq2`
  are NB-based (Topic 20).

**Zero inflation.** A hurdle or zero-inflated model separates a structural-zero
process from the count process:

$$\Pr(Y=0) = \pi + (1-\pi)f(0;\mu), \qquad
\Pr(Y=y) = (1-\pi)f(y;\mu),\ y>0. \tag{13.10}$$

Fit one only if you can name the biological mechanism generating structural
zeros. In droplet scRNA-seq, UMI counts are generally well described by an NB
without zero inflation - the excess zeros are explained by low $\mu$ plus
sampling, and adding a spurious $\pi$ costs power.

**Three tests, one hypothesis.** For $H_0: \boldsymbol\beta_S = \mathbf{0}$:

$$\text{Wald: } \hat{\boldsymbol\beta}_S^\top\widehat{\operatorname{Var}}(\hat{\boldsymbol\beta}_S)^{-1}\hat{\boldsymbol\beta}_S, \quad
\text{LRT: } 2\big(\ell_1 - \ell_0\big), \quad
\text{Score: } U(\boldsymbol\beta_0)^\top\mathcal{I}^{-1}U(\boldsymbol\beta_0), \tag{13.11}$$

all $\ \dot\sim\ \chi^2_{|S|}$. They agree asymptotically but not in finite
samples: **prefer the LRT**, which is invariant to reparameterisation and far
better behaved under near-separation, where the Wald test can lose power
entirely (the Hauck-Donner effect).

**Deviance.**

$$D = 2\big(\ell_{\text{saturated}} - \ell_{\text{model}}\big). \tag{13.12}$$

Deviance residuals $d_i = \operatorname{sign}(y_i - \hat\mu_i)\sqrt{d_i^2}$ are
the right residuals to plot for a GLM; raw residuals are heteroscedastic by
construction.

### Decision rules

1. Choose the family from the outcome's support and mean-variance relationship
   (3.6), and the link from the estimand you want to report.
2. Counts with varying exposure require an offset (13.8), always.
3. Check (13.9). Overdispersed counts modelled as Poisson produce dramatically
   anticonservative p-values.
4. Report both scales: the odds/rate ratio *and* a predicted probability or rate
   at meaningful covariate values.


## Topic 14: Mixed, multilevel, and repeated-measures models

**Module:** `14_mixed_models`

### The question

Observations cluster - cells in donors, visits in patients, samples in centres,
mice in litters. How do we get standard errors that reflect the real number of
independent units (1.5)?

### Model

$$\mathbf{y} = \mathbf{X}\boldsymbol\beta + \mathbf{Z}\mathbf{b} + \boldsymbol\varepsilon,
\qquad \mathbf{b}\sim\mathcal{N}(\mathbf{0},\mathbf{G}),\quad
\boldsymbol\varepsilon\sim\mathcal{N}(\mathbf{0},\mathbf{R}), \quad \mathbf{b}\perp\boldsymbol\varepsilon. \tag{14.1}$$

$\mathbf{X}$ holds fixed effects (the population-level question), $\mathbf{Z}$
the random-effect design (which cluster each row belongs to).

### Key equations

**Marginal covariance**: the whole point of the model:

$$\operatorname{Var}(\mathbf{y}) = \mathbf{V} = \mathbf{Z}\mathbf{G}\mathbf{Z}^\top + \mathbf{R}. \tag{14.2}$$

For a random intercept with $\mathbf{G} = \sigma_b^2\mathbf{I}$ and
$\mathbf{R} = \sigma_e^2\mathbf{I}$, $\mathbf{V}$ is block-diagonal:
within a cluster every pair of observations has covariance $\sigma_b^2$, hence
correlation

$$\mathrm{ICC} = \rho = \frac{\sigma_b^2}{\sigma_b^2 + \sigma_e^2}, \tag{14.3}$$

which is exactly (1.7). A mixed model is the formal machinery that turns (1.5)
into correct standard errors automatically.

**Estimation.** Given $\mathbf{V}$, the fixed effects are generalised least
squares:

$$\hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{V}^{-1}\mathbf{y},
\qquad \operatorname{Var}(\hat{\boldsymbol\beta}) = (\mathbf{X}^\top\mathbf{V}^{-1}\mathbf{X})^{-1}. \tag{14.4}$$

Variance components are estimated by **REML**, which maximises the likelihood of
error contrasts $\mathbf{A}^\top\mathbf{y}$ with $\mathbf{A}^\top\mathbf{X}=\mathbf{0}$,
removing the downward bias that ML has for variance parameters (the same
$n$ vs $n-1$ issue as (5.2), generalised). **Consequence:** REML fits with
*different fixed effects* are not comparable by likelihood ratio. Refit with ML
before doing an LRT on fixed effects.

**Partial pooling and BLUPs.** The predicted random effect is

$$\hat{\mathbf{b}} = \mathbf{G}\mathbf{Z}^\top\mathbf{V}^{-1}(\mathbf{y}-\mathbf{X}\hat{\boldsymbol\beta}), \tag{14.5}$$

which for a simple random intercept with $n_j$ observations in cluster $j$
reduces to a shrinkage-weighted deviation:

$$\hat{b}_j = \underbrace{\frac{\sigma_b^2}{\sigma_b^2 + \sigma_e^2/n_j}}_{\text{shrinkage factor } \lambda_j}\;
\big(\bar{y}_j - \mathbf{x}_j^\top\hat{\boldsymbol\beta}\big). \tag{14.6}$$

Small clusters ($n_j$ small) are shrunk hard toward the population mean; large
clusters are trusted. This is empirical Bayes (Topic 33) in disguise, and it is
exactly the mechanism by which `limma` borrows strength across genes.

**Degrees of freedom.** The null distribution of a fixed-effect t-statistic is
not exactly $t$ with any integer df. Two corrections matter:

- **Satterthwaite:** matches moments of the estimated variance.
- **Kenward-Roger:** additionally inflates $\operatorname{Var}(\hat{\boldsymbol\beta})$
  to account for estimating $\mathbf{V}$; the better choice for small,
  unbalanced designs.

Without a correction, p-values from GLMM/LMM software using a $z$
approximation are **anticonservative with few clusters**. Rule of thumb: with
fewer than ~8 clusters per arm, a random-effect variance is poorly estimated;
consider a fixed-effect blocking factor instead.

**Subject-specific vs population-average.** In a GLMM with a nonlinear link, the
conditional (subject-specific) coefficient is **not** the marginal
(population-average) coefficient. For a logistic random-intercept model:

$$\beta^{\text{marginal}} \approx \frac{\beta^{\text{conditional}}}
{\sqrt{1 + 0.346\,\sigma_b^2}}. \tag{14.7}$$

GEE estimates the marginal effect directly; GLMM estimates the conditional one.
Say which you mean. (For identity and log links the two coincide, which is why
this trap is invisible in linear and Poisson models.)

**GEE.** Specify only the mean and a *working* correlation $\mathbf{R}(\alpha)$;
the sandwich estimator (5.14, cluster version) gives valid SEs even if the
working correlation is wrong:

$$\widehat{\operatorname{Var}}(\hat{\boldsymbol\beta}) = \mathbf{A}^{-1}
\left(\sum_{j=1}^{J}\mathbf{D}_j^\top\mathbf{V}_j^{-1}\mathbf{e}_j\mathbf{e}_j^\top\mathbf{V}_j^{-1}\mathbf{D}_j\right)\mathbf{A}^{-1}. \tag{14.8}$$

This robustness requires **many clusters** ($J \ge 40$ is the usual guidance);
with 6 donors the sandwich is badly biased and a mixed model is safer.

**Singular fits.** When $\hat\sigma_b^2 \to 0$ the fit is on the boundary. This
usually means: too few clusters, an over-specified random-effects structure
(random slopes you cannot identify), or genuinely negligible clustering.
Simplify the random structure; do not simply ignore the warning.

Note also that testing $H_0:\sigma_b^2=0$ puts the null on the boundary of the
parameter space, so the LRT statistic is **not** $\chi^2_1$ but a 50:50 mixture
$\tfrac12\chi^2_0 + \tfrac12\chi^2_1$; using $\chi^2_1$ makes the test
conservative by a factor of two in the p-value.

**The pseudobulk alternative.** For balanced designs, aggregating within cluster
and analysing the $J$ cluster-level summaries is simple, robust and often has
*identical* power to the mixed model. This is why pseudobulk aggregation is the
recommended default for multi-sample single-cell differential
expression.[^sc1][^sc2][^sc3]

### Decision rules

1. Enumerate every level of clustering before modelling. The random effects are
   determined by the design, not by model fit.
2. Fixed blocking effects when clusters are few and you only want to adjust;
   random effects when clusters are many, are a sample from a population, or you
   want to predict cluster values.
3. Use Satterthwaite/Kenward-Roger df, and never trust $z$-based GLMM p-values
   with few clusters.
4. When in doubt with a simple design, aggregate to the cluster level - it is
   transparent and nearly always valid.


## Topic 15: Missing data, censoring, and measurement limits

**Module:** `15_missing_data_and_censoring`

### The question

A blank cell in a proteomics matrix could mean "below detection", "peptide not
identified this run", or "sample failed". These require different models, and
`na.rm = TRUE` assumes the most optimistic one.[^md1]

### Key equations

Let $\mathbf{Y} = (\mathbf{Y}_{\text{obs}}, \mathbf{Y}_{\text{mis}})$ and let
$\mathbf{M}$ be the missingness indicator matrix.

$$\textbf{MCAR: } \Pr(\mathbf{M}\mid\mathbf{Y},\boldsymbol\psi) = \Pr(\mathbf{M}\mid\boldsymbol\psi). \tag{15.1}$$

$$\textbf{MAR: } \Pr(\mathbf{M}\mid\mathbf{Y},\boldsymbol\psi) = \Pr(\mathbf{M}\mid\mathbf{Y}_{\text{obs}},\boldsymbol\psi). \tag{15.2}$$

$$\textbf{MNAR: } \Pr(\mathbf{M}\mid\mathbf{Y},\boldsymbol\psi) \ \text{depends on } \mathbf{Y}_{\text{mis}}. \tag{15.3}$$

Under MCAR, complete-case analysis is unbiased but wasteful. Under MAR, it is
biased in general, but likelihood-based methods and multiple imputation that
*condition on the variables driving missingness* are valid. Under MNAR, **no
method is valid without an untestable assumption** - you can only do sensitivity
analysis. Critically, **MAR vs MNAR cannot be distinguished from the observed
data**; it is an assumption you argue for, not test.

**Multiple imputation and Rubin's rules.** Generate $M$ completed datasets,
analyse each, then combine:

$$\bar{Q} = \frac{1}{M}\sum_{m=1}^{M} \hat{Q}_m \qquad \text{(pooled estimate)}, \tag{15.4}$$

$$\bar{U} = \frac{1}{M}\sum_m U_m \ \text{(within-imputation variance)}, \qquad
B = \frac{1}{M-1}\sum_m (\hat{Q}_m - \bar{Q})^2 \ \text{(between)}, \tag{15.5}$$

$$T = \bar{U} + \left(1 + \frac{1}{M}\right)B. \tag{15.6}$$

The $(1+1/M)B$ term in (15.6) is exactly what single imputation omits - which
is why single imputation gives standard errors that are too small and CIs that
undercover. The fraction of missing information is
$\gamma = (1+1/M)B / T$, and the pooled df is

$$\nu = (M-1)\left(1 + \frac{\bar{U}}{(1+1/M)B}\right)^2. \tag{15.7}$$

**The imputation model must be at least as rich as the analysis model**: it
must include the outcome, all covariates, and any interactions or nonlinear
terms you will later fit. Omitting the outcome from the imputation model biases
associations toward zero.

**Left-censoring (limit of detection).** For proteomics/metabolomics, a value
below LOD $L$ is not missing - it is *known to be below $L$*. The correct
likelihood is a censored (Tobit) likelihood:

$$L(\mu,\sigma) = \prod_{i:\,y_i > L}\frac{1}{\sigma}\phi\!\left(\frac{y_i-\mu}{\sigma}\right)
\prod_{i:\,y_i \le L}\Phi\!\left(\frac{L-\mu}{\sigma}\right). \tag{15.8}$$

Substituting $L/2$ or $L/\sqrt{2}$ is a crude approximation to (15.8) that
biases variance estimates downward. The common proteomics practice of imputing
all missing values from a shifted low-abundance distribution (e.g.
`MinProb`/`QRILC`) assumes *all* missingness is censoring - which is false when
identification also fails stochastically. Modern practice separates the two:
MNAR-style imputation for features missing systematically in one condition, MAR
imputation for sporadic gaps.

**Diagnostic for the mechanism.** Plot the per-feature missingness rate against
the observed mean abundance. A strong negative trend is evidence of
abundance-dependent (MNAR/censoring) missingness; a flat cloud suggests MAR.
This single plot should be in every proteomics QC report.

**Inverse-probability weighting.** Model the probability of being observed,
$\pi_i = \Pr(M_i = 0 \mid \mathbf{Z}_i)$, and weight complete cases by $1/\pi_i$:

$$\hat\theta_{\mathrm{IPW}} = \frac{\sum_i \frac{\mathbb{1}\{M_i=0\}}{\hat\pi_i} h(Y_i)}
{\sum_i \frac{\mathbb{1}\{M_i=0\}}{\hat\pi_i}}. \tag{15.9}$$

Weights become unstable when $\hat\pi_i$ is near zero; truncate or stabilise them.

**Missingness in prediction pipelines.** Imputation is a *learned transformation*.
Fitting it on the full dataset before cross-validation leaks test information
into training and inflates apparent performance. It must go inside the fold
(Topic 31).

### Decision rules

1. Name the mechanism for each variable and justify it from how the assay works.
2. Never impute the outcome to create observations; model it instead.
3. Use $M \ge 20$ imputations when the fraction of missing information is
   substantial - the old "$M=5$" advice is outdated for efficiency.
4. Model detection limits as censoring (15.8), not as missingness.
5. Report a sensitivity analysis over plausible MNAR departures. This is not
   optional for proteomics.


# Part IV: Multivariate statistics

## Topic 16: Matrix algebra for high-dimensional biology

**Module:** `16_matrix_algebra`

### The question

A $20{,}000 \times 12$ expression matrix has more features than samples. What
does that do to the algebra, and which decomposition answers which question?

### Key equations

**Inner product, norm, projection.**

$$\mathbf{a}^\top\mathbf{b} = \sum_k a_k b_k = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta, \qquad
\|\mathbf{a}\| = \sqrt{\mathbf{a}^\top\mathbf{a}}. \tag{16.1}$$

The projection of $\mathbf{y}$ onto the span of $\mathbf{x}$:

$$\mathrm{proj}_{\mathbf{x}}(\mathbf{y}) = \frac{\mathbf{x}^\top\mathbf{y}}{\mathbf{x}^\top\mathbf{x}}\,\mathbf{x}. \tag{16.2}$$

Compare the scalar in (16.2) with (11.2): **simple regression is a projection**,
and $\hat\beta_1$ is the coordinate of $\mathbf{y}$ along $\mathbf{x}$. If
$\mathbf{x}$ and $\mathbf{y}$ are centred, the same scalar times
$\|\mathbf{x}\|/\|\mathbf{y}\|$ is the correlation (10.2). The three ideas are
one idea.

**Centering and the Gram matrix.** The centering matrix

$$\mathbf{C} = \mathbf{I}_n - \tfrac{1}{n}\mathbf{1}_n\mathbf{1}_n^\top \tag{16.3}$$

is symmetric and idempotent. With $\mathbf{X}$ an $n\times p$ sample-by-feature
matrix and $\tilde{\mathbf{X}} = \mathbf{C}\mathbf{X}$,

$$\mathbf{S} = \frac{1}{n-1}\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}} \quad (p\times p, \text{ covariance}),
\qquad \mathbf{K} = \tilde{\mathbf{X}}\tilde{\mathbf{X}}^\top \quad (n\times n, \text{ Gram}). \tag{16.4}$$

$\mathbf{S}$ and $\mathbf{K}$ share the same non-zero eigenvalues. **When
$p \gg n$, always decompose the smaller $n\times n$ matrix.** For 20,000 genes x
12 samples that is a $12\times12$ eigenproblem rather than a
$20{,}000\times20{,}000$ one.

**Rank and identifiability.**

$$\operatorname{rank}(\mathbf{X}) \le \min(n,p). \tag{16.5}$$

If $\operatorname{rank}(\mathbf{X}) < p$, then $\mathbf{X}^\top\mathbf{X}$ is
singular, (11.2) has no unique solution, and some coefficients are
**not identifiable**. This is the algebra of a confounded design: if batch is
nested perfectly inside condition, the corresponding columns are linearly
dependent, and no amount of software cleverness recovers the separate effects.

**Singular value decomposition - the most useful factorisation.**

$$\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top, \qquad
\mathbf{U}^\top\mathbf{U} = \mathbf{V}^\top\mathbf{V} = \mathbf{I},\quad
\mathbf{D} = \operatorname{diag}(d_1 \ge \dots \ge d_r > 0). \tag{16.6}$$

Consequences you will use constantly:

$$\mathbf{X}^\top\mathbf{X} = \mathbf{V}\mathbf{D}^2\mathbf{V}^\top, \qquad
\mathbf{X}\mathbf{X}^\top = \mathbf{U}\mathbf{D}^2\mathbf{U}^\top, \tag{16.7}$$

so SVD gives both eigendecompositions at once, and does so without ever forming
$\mathbf{X}^\top\mathbf{X}$ - which is why it is numerically superior.

**Eckart-Young theorem.** The best rank-$k$ approximation in Frobenius (and
spectral) norm is the truncated SVD:

$$\mathbf{X}_k = \sum_{j=1}^{k} d_j \mathbf{u}_j\mathbf{v}_j^\top
= \arg\min_{\operatorname{rank}(\mathbf{B})\le k}\|\mathbf{X}-\mathbf{B}\|_F, \qquad
\|\mathbf{X}-\mathbf{X}_k\|_F^2 = \sum_{j>k} d_j^2. \tag{16.8}$$

Equation (16.8) is the theoretical justification for PCA-based denoising, for
latent-factor models, and for the "keep the top $k$ PCs" step in every
single-cell pipeline.

**Moore-Penrose pseudoinverse** handles rank deficiency gracefully:

$$\mathbf{X}^{+} = \mathbf{V}\mathbf{D}^{-1}\mathbf{U}^\top, \qquad
\hat{\boldsymbol\beta}_{\text{min-norm}} = \mathbf{X}^{+}\mathbf{y}. \tag{16.9}$$

This returns the minimum-norm solution among infinitely many - useful, but never
silently: if you needed (16.9), your design was rank-deficient and you should
know why.

**Condition number** quantifies numerical fragility:

$$\kappa(\mathbf{X}) = \frac{d_1}{d_r}. \tag{16.10}$$

$\kappa > 10^3$ signals near-collinearity; $\kappa(\mathbf{X}^\top\mathbf{X}) = \kappa(\mathbf{X})^2$,
which is why forming the normal equations loses half your significant digits.

**Quadratic forms.** For $\mathbf{y}\sim\mathcal{N}(\boldsymbol\mu,\sigma^2\mathbf{I})$
and symmetric idempotent $\mathbf{A}$ of rank $r$:

$$\frac{\mathbf{y}^\top\mathbf{A}\mathbf{y}}{\sigma^2} \sim \chi^2_{r}\!\left(\frac{\boldsymbol\mu^\top\mathbf{A}\boldsymbol\mu}{\sigma^2}\right). \tag{16.11}$$

This is the engine behind every ANOVA table: $\mathrm{SS}$ terms are quadratic
forms in projection matrices, their df are the ranks, and Cochran's theorem
gives their independence.

### Decision rules

1. Decide which dimension you are decomposing - samples or features - before
   interpreting any output.
2. Use SVD/QR, not explicit inverses.
3. Check $\operatorname{rank}(\mathbf{X})$ and $\kappa(\mathbf{X})$ before
   fitting $G$ models with the same design.
4. Exploit $\mathbf{S} \leftrightarrow \mathbf{K}$ duality (16.4) when $p \gg n$.


## Topic 17: Principal component analysis

**Module:** `17_pca`

### The question

What are the dominant axes of variation, and is the largest one biology or
technology? PCA answers the first; only metadata answers the second.[^pca1][^pca2]

### Key equations

**Definition.** PCA finds orthonormal directions maximising projected variance:

$$\mathbf{v}_1 = \arg\max_{\|\mathbf{v}\|=1} \operatorname{Var}(\tilde{\mathbf{X}}\mathbf{v})
= \arg\max_{\|\mathbf{v}\|=1} \mathbf{v}^\top\mathbf{S}\mathbf{v}. \tag{17.1}$$

The Lagrangian $\mathbf{v}^\top\mathbf{S}\mathbf{v} - \lambda(\mathbf{v}^\top\mathbf{v}-1)$
differentiates to $\mathbf{S}\mathbf{v} = \lambda\mathbf{v}$: **the solution is
the leading eigenvector**, and the maximised variance is the eigenvalue. That
two-line derivation is all PCA is.

**Via SVD.** With $\tilde{\mathbf{X}} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$
($n$ samples x $p$ features, centred):

$$\underbrace{\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{V} = \mathbf{U}\mathbf{D}}_{\text{scores } (n\times k)},
\qquad \underbrace{\mathbf{V}}_{\text{loadings } (p\times k)}, \qquad
\lambda_j = \frac{d_j^2}{n-1}. \tag{17.2}$$

$$\mathrm{PVE}_j = \frac{d_j^2}{\sum_{m} d_m^2} = \frac{\lambda_j}{\sum_m \lambda_m}. \tag{17.3}$$

**Scores** are sample coordinates (plot these to see sample structure);
**loadings** are feature weights (read these to interpret what a PC means). The
reconstruction is $\tilde{\mathbf{X}} \approx \mathbf{Z}_k\mathbf{V}_k^\top$.

**Centering is mandatory; scaling is a decision.** Without centering, PC1 just
points at the mean vector. Scaling to unit variance means

$$\text{correlation PCA} \iff \text{covariance PCA on standardised features}. \tag{17.4}$$

Choose:
- **Scale** when features have different units or wildly different variances for
  technical reasons (raw clinical biomarkers: mg/dL vs cells/µL).
- **Do not scale** when features share units and the variance differences are the
  biology you want (log-expression, M-values, arcsinh cytometry). Scaling
  log-expression up-weights the noisiest low-expressed genes.

**How many components?** Three defensible answers:

*Broken stick / parallel analysis (Horn's):* keep $\lambda_j$ exceeding the
$95$th percentile of eigenvalues from permuted data (each feature permuted
independently to destroy correlation while preserving marginals).

*Random-matrix threshold (Marchenko-Pastur):* for pure noise with $p/n \to \gamma$,
eigenvalues of the sample correlation matrix are bounded above by

$$\lambda_{+} = \big(1+\sqrt{\gamma}\big)^2. \tag{17.5}$$

Components above $\lambda_+$ carry signal. With $p = 2000$, $n = 50$,
$\gamma = 40$, so $\lambda_+ = (1+6.32)^2 \approx 53.6$ - a sobering reminder of
how large noise eigenvalues get when $p \gg n$.

*Scree elbow:* useful, informal, and seed-independent - but do not over-read it.

**Sign and rotation indeterminacy.** If $(\mathbf{u}_j, \mathbf{v}_j)$ is a
solution so is $(-\mathbf{u}_j, -\mathbf{v}_j)$; PC signs are arbitrary and vary
between software versions. Never interpret the sign of a PC as direction of
biology without anchoring it to a known feature. Moreover, if
$\lambda_j \approx \lambda_{j+1}$, the individual eigenvectors are unstable
(only the subspace they span is well defined) - a genuine reproducibility trap.

**PCA is not batch correction and not a test.** Two specific errors:

1. *Overinterpreting separation.* Random data in high dimensions always separate
   in the top PCs of a small sample. Permute the group labels and recompute the
   between-group separation in PC space to get a null distribution.
2. *Circularity / double dipping.* Selecting features by a PC and then testing
   those features against the same PC or the same grouping guarantees
   significance. If you must do both, split the data or use selective-inference
   corrections.

**Associating PCs with metadata** is the most valuable PCA output in
bioinformatics. For each PC $j$ and each metadata variable $w$, regress
$\mathbf{z}_j$ on $w$ and record $R^2$ and a p-value:

$$R^2_{j,w} = \operatorname{cor}^2(\mathbf{z}_j, w) \quad\text{(continuous } w). \tag{17.6}$$

Tabulate PC x covariate $R^2$. If PC1 is 60% explained by processing date, you
have a batch problem (Topic 19), not a discovery.

**Robust, sparse, and probabilistic variants.**
- *Sparse PCA* adds an $\ell_1$ penalty on loadings for interpretability.
- *Probabilistic PCA* models $\mathbf{x} = \mathbf{W}\mathbf{z} + \boldsymbol\mu + \boldsymbol\epsilon$
  with $\boldsymbol\epsilon\sim\mathcal{N}(0,\sigma^2\mathbf{I})$, which handles
  missing values properly via EM instead of requiring imputation first.
- *GLM-PCA* replaces the Gaussian reconstruction loss with a count likelihood -
  the principled choice for raw UMI counts, avoiding the log(x+1) distortion.

### Decision rules

1. Always centre. Scale only with a stated reason.
2. Interpret PCs through loadings and metadata associations (17.6), never
   through the appearance of a scatter plot.
3. Report the PVE for every displayed component, and use equal aspect ratio so
   distances are not visually distorted.
4. Keep PCA used for QC separate from PCs used as model covariates, and never
   test features on the same data that defined the component.


## Topic 18: Distances, clustering, and embeddings

**Module:** `18_distances_and_clustering`

### The question

Clustering always returns clusters. Which of them would survive a different
distance, a different seed, or a different half of the data?

### Key equations

**Distances, and when each is right.**

$$d_{\text{Euclid}}(\mathbf{x},\mathbf{y}) = \sqrt{\sum_k (x_k-y_k)^2}, \qquad
d_{\text{Manh}} = \sum_k |x_k - y_k|. \tag{18.1}$$

$$d_{\text{corr}}(\mathbf{x},\mathbf{y}) = 1 - r_{\mathbf{x}\mathbf{y}}, \qquad
d_{\cos} = 1 - \frac{\mathbf{x}^\top\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}. \tag{18.2}$$

Correlation distance ignores magnitude - right for grouping genes by *shape* of
expression profile, wrong for samples whose overall level matters.

$$d_{\text{Jaccard}}(A,B) = 1 - \frac{|A\cap B|}{|A\cup B|} \quad\text{(presence/absence)}, \tag{18.3}$$

$$d_{\text{BC}}(\mathbf{x},\mathbf{y}) = \frac{\sum_k |x_k - y_k|}{\sum_k (x_k + y_k)} \quad\text{(abundance, ecology)}, \tag{18.4}$$

$$d_{\text{Aitchison}}(\mathbf{x},\mathbf{y}) = \big\|\operatorname{clr}(\mathbf{x}) - \operatorname{clr}(\mathbf{y})\big\|_2
\quad\text{(compositions; see Topic 26).} \tag{18.5}$$

There is an important identity linking Euclidean distance to correlation for
standardised vectors:

$$d^2_{\text{Euclid}}(\mathbf{z}_x,\mathbf{z}_y) = 2(p-1)\big(1 - r_{xy}\big), \tag{18.6}$$

so clustering z-scored features by Euclidean distance *is* clustering by
correlation. Knowing this prevents a lot of confused method-shopping.

**The curse of dimensionality.** As $p \to \infty$ with independent features,

$$\frac{\max_j d(\mathbf{x},\mathbf{x}_j) - \min_j d(\mathbf{x},\mathbf{x}_j)}{\min_j d(\mathbf{x},\mathbf{x}_j)} \to 0. \tag{18.7}$$

All points become equidistant, and "nearest neighbour" loses meaning. This is why
every scRNA-seq pipeline computes neighbours in the top ~30 PCs rather than in
20,000-dimensional gene space.

**Hierarchical clustering.** All standard linkages are special cases of the
Lance-Williams update, which gives the distance from a merged cluster
$(i\cup j)$ to $k$:

$$d_{(ij),k} = \alpha_i d_{ik} + \alpha_j d_{jk} + \beta d_{ij} + \gamma|d_{ik}-d_{jk}|. \tag{18.8}$$

| Linkage | Behaviour | Typical use |
|---|---|---|
| Single | Chaining; finds elongated shapes | Rarely useful for omics |
| Complete | Compact, equal-diameter clusters | Default for sample heatmaps |
| Average (UPGMA) | Between the two | Phylogenetics, microbiome |
| Ward.D2 | Minimises within-cluster SS | Spherical clusters; **requires Euclidean** |

Ward's criterion minimises the increase in error sum of squares:

$$\Delta_{ij} = \frac{n_i n_j}{n_i + n_j}\|\bar{\mathbf{x}}_i - \bar{\mathbf{x}}_j\|^2. \tag{18.9}$$

Using `ward.D` on a squared-distance object (or `ward.D2` on squared distances)
is a common and silent error; `ward.D2` expects ordinary (non-squared) distances.

**k-means** minimises within-cluster sum of squares:

$$\min_{C_1,\dots,C_K}\sum_{k=1}^{K}\sum_{i\in C_k}\|\mathbf{x}_i - \boldsymbol\mu_k\|^2,
\qquad \boldsymbol\mu_k = \frac{1}{|C_k|}\sum_{i\in C_k}\mathbf{x}_i. \tag{18.10}$$

Lloyd's algorithm converges to a *local* optimum, so results depend on
initialisation: use k-means++ and `nstart >= 25`. k-means implicitly assumes
spherical, equally-sized clusters and is only meaningful with Euclidean
distance. Use PAM/k-medoids for arbitrary dissimilarities.

**Choosing $K$.**

*Silhouette*, for point $i$ with mean within-cluster distance $a(i)$ and mean
distance to the nearest other cluster $b(i)$:

$$s(i) = \frac{b(i)-a(i)}{\max\{a(i),b(i)\}} \in [-1,1]. \tag{18.11}$$

*Gap statistic*, comparing to a null of no structure:

$$\mathrm{Gap}(K) = \mathbb{E}^{*}\big[\log W_K\big] - \log W_K, \tag{18.12}$$

with $W_K$ the pooled within-cluster dispersion and the expectation taken over
uniform reference data in the bounding box of the PCA-rotated data. Select the
smallest $K$ with $\mathrm{Gap}(K) \ge \mathrm{Gap}(K+1) - s_{K+1}$.

*Stability* is the most honest criterion for biology: subsample 80% of units
$B$ times, recluster, and measure the co-clustering frequency of each pair.
Unstable clusters should not be named cell types.

**Graph-based clustering** (Louvain/Leiden) maximises modularity

$$Q = \frac{1}{2m}\sum_{i,j}\left(A_{ij} - \frac{k_ik_j}{2m}\right)\delta(c_i,c_j), \tag{18.13}$$

which compares observed within-community edges to a degree-preserving null.
Resolution is a free parameter that *determines* the number of clusters; there is
no "correct" resolution, so report sensitivity across a range. Use Leiden over
Louvain: Louvain can return internally disconnected communities.

**t-SNE and UMAP are visualisations, not analyses.** t-SNE minimises

$$\mathrm{KL}(P\|Q) = \sum_{i\ne j} p_{ij}\log\frac{p_{ij}}{q_{ij}}, \qquad
q_{ij} \propto \big(1+\|\mathbf{y}_i-\mathbf{y}_j\|^2\big)^{-1}. \tag{18.14}$$

The asymmetric KL in (18.14) penalises putting near points far apart far more
than far points near, so **local structure is preserved and global distances are
not**. Consequences that are routinely violated in figures:

- Between-cluster distances are meaningless.
- Cluster sizes/areas are meaningless.
- Results depend on perplexity, seed, initialisation, and the number of PCs.
- Apparent gaps can be artefacts of the optimisation.

Report the seed and every parameter, and never quantify anything on embedding
coordinates.

**Double dipping.** Defining clusters from the data and then testing for
differences between those clusters using the same features and same data
produces small p-values even when the data are a single homogeneous Gaussian.
Remedies: data splitting (cluster on half, test on the other half), or
selective-inference tests that condition on the clustering event.

### Decision rules

1. Pick the distance from the measurement scale: (18.4)/(18.5) for compositions,
   (18.2) for profile shape, (18.3) for binary, Euclidean on a
   variance-stabilised scale otherwise.
2. Cluster in a reduced space (top PCs) for high-dimensional data (18.7).
3. Report $K$-selection evidence *and* stability, not just a coloured plot.
4. Never test the features that defined the clusters on the same data.


## Topic 19: Batch effects and unwanted variation

**Module:** `19_batch_effects`

### The question

Was the difference between groups produced by biology or by the week each group
was processed? Prevention beats correction; diagnostics beat faith.[^batch1]

### Key equations

**The additive-plus-multiplicative model** (the location-scale model that
`ComBat` formalises). For feature $g$, sample $j$ in batch $i$:

$$Y_{ijg} = \alpha_g + \mathbf{x}_j^\top\boldsymbol\beta_g + \gamma_{ig} + \delta_{ig}\varepsilon_{ijg}. \tag{19.1}$$

Here $\gamma_{ig}$ shifts the mean and $\delta_{ig}$ scales the variance of
feature $g$ in batch $i$. ComBat estimates $(\gamma_{ig}, \delta_{ig})$ with
empirical-Bayes shrinkage across features (Topic 33), which is what makes it
stable with few samples per batch, and produces

$$Y^{*}_{ijg} = \frac{\hat\sigma_g}{\hat\delta_{ig}}\Big(Y_{ijg} - \hat\alpha_g - \mathbf{x}_j^\top\hat{\boldsymbol\beta}_g - \hat\gamma_{ig}\Big)
+ \hat\alpha_g + \mathbf{x}_j^\top\hat{\boldsymbol\beta}_g. \tag{19.2}$$

**The single most important operational point about (19.2):** the biological
covariates $\mathbf{x}_j$ must be supplied to the correction. Correcting without
the design model removes batch *and* any biology correlated with batch. And
even done correctly, the corrected values have **understated uncertainty** -
the correction was estimated, but downstream tests treat $Y^*$ as observed.

**Identifiability.** Write the design with batch $\mathbf{B}$ and condition
$\mathbf{A}$:

$$\mathbf{X} = [\mathbf{1}\ \ \mathbf{A}\ \ \mathbf{B}]. \tag{19.3}$$

If every sample in batch 1 is control and every sample in batch 2 is treated,
the columns satisfy a linear dependency, $\operatorname{rank}(\mathbf{X}) < p$,
and by (16.5) the condition effect is **not identifiable**. No method - ComBat,
SVA, Harmony, MNN - can recover it. The correct response is to report the
confounding, not to correct and proceed.

**Partial confounding** is quantifiable: the variance inflation of the condition
coefficient is (11.12) with $R^2$ from regressing condition on batch. If
$R^2 = 0.8$, $\mathrm{VIF} = 5$ and your SE is $\sqrt{5} \approx 2.24\times$
larger - recoverable but expensive.

**Model-based adjustment (usually preferred over correcting the matrix).** Simply
include batch in the design:

$$\mathbb{E}[Y_{g}] = \beta_{0g} + \beta_{1g}\,\text{condition} + \sum_{b}\gamma_{bg}\,\mathbb{1}\{\text{batch}=b\}. \tag{19.4}$$

This propagates the uncertainty of the batch estimates into the standard error
of $\beta_{1g}$, which (19.2) does not. **Rule: correct the matrix for
visualisation and distance-based methods; adjust in the model for inference.**

**Latent unwanted variation.** When batch is unrecorded, estimate surrogate
variables. SVA decomposes the residuals after removing the biological model,

$$\mathbf{R} = \mathbf{Y} - \hat{\mathbf{Y}}_{\text{bio}} = \mathbf{U}\mathbf{D}\mathbf{V}^\top, \tag{19.5}$$

takes the top eigenvectors as surrogate variables, and adds them as covariates in
(19.4). RUV instead uses **negative control features**: genes assumed unaffected
by the biology (spike-ins, housekeeping genes):

$$\mathbf{Y}_{\text{ctl}} = \mathbf{W}\boldsymbol\alpha_{\text{ctl}} + \mathbf{E}
\ \Rightarrow\ \hat{\mathbf{W}} \ \text{from SVD of } \mathbf{Y}_{\text{ctl}}, \tag{19.6}$$

then includes $\hat{\mathbf{W}}$ in the design for all features. The validity of
RUV rests entirely on the control genes being genuinely unaffected - an
assumption to defend, not assume.

**Diagnostics, before and after.**

| Metric | What it shows | Want |
|---|---|---|
| PC x metadata $R^2$ table (17.6) | Which factor dominates variance | Batch $R^2$ drops, condition $R^2$ preserved |
| PVCA / variance partition | Fraction of variance per factor | Batch fraction falls |
| kBET / iLISI (neighbourhood mixing) | Local batch mixing | Batches mixed |
| cLISI / label preservation | Biological label purity | Preserved |
| Silhouette by batch vs by condition | Global separation | Batch silhouette $\to 0$; condition stable |
| Known positive/negative controls | Biological sanity | Positives retained |

Report at least one **mixing** metric and one **biology-preservation** metric.
Reporting only mixing rewards overcorrection, which is the dominant failure mode
of integration methods.

### Decision rules

1. Prevent: randomise samples across batches, balance conditions within batch,
   include bridging/reference samples in every batch, record every processing
   variable.
2. Diagnose before correcting: cross-tabulate condition x batch and build the
   PC x metadata table.
3. If completely confounded, stop and say so.
4. Adjust in the model (19.4) for inference; correct the matrix (19.2) only for
   visualisation, clustering and prediction - and then keep the correction
   inside cross-validation folds (Topic 31).
5. Always evaluate both batch removal and biology preservation.


# Part V: Modality-specific statistics

## Topic 20: Bulk RNA-seq differential expression

**Module:** `30_bulk_rnaseq_differential_expression`

### The question

Which genes change between conditions, given counts whose variance grows
quadratically with the mean and only a handful of replicates?[^rna1][^rna2][^rna3]

### Model

$$K_{gj} \sim \mathrm{NB}\big(\mu_{gj},\ \phi_g\big), \qquad
\log \mu_{gj} = \log s_j + \mathbf{x}_j^\top\boldsymbol\beta_g, \tag{20.1}$$

with $\operatorname{Var}(K_{gj}) = \mu_{gj} + \phi_g\mu_{gj}^2$ from (4.9). Note
the structure: this is exactly the GLM of (13.8) with a size-factor offset,
applied $G$ times, plus information sharing across $g$ to estimate $\phi_g$.

### Key equations

**Why raw counts and not TPM.** TPM/FPKM are ratios that have already divided out
library size; feeding them to (20.1) discards the count information that
determines the variance. A gene at 5 counts and a gene at 5000 counts can have
the same TPM but vastly different precision. Normalisation belongs in the
**offset** $\log s_j$, not in the response.

**Median-of-ratios size factors (DESeq2).**

$$s_j = \operatorname{median}_{g \,:\, K_{gv} > 0 \,\forall v}
\left( \frac{K_{gj}}{\big(\prod_{v=1}^{n} K_{gv}\big)^{1/n}} \right). \tag{20.2}$$

The reference is the geometric mean across samples, so (20.2) is robust: a few
very highly expressed genes cannot dominate, which plain total-count scaling
allows. **TMM (edgeR)** achieves the same goal by trimming M-values
$M_{gj} = \log_2(K_{gj}/N_j) - \log_2(K_{gr}/N_r)$ and A-values, then taking a
precision-weighted mean of the middle. Both correct *composition* bias: if one
gene takes 40% of the library in treated samples, every other gene appears
down-regulated unless corrected.

**Dispersion estimation with empirical Bayes.** With $n = 3$ per group, the
gene-wise $\hat\phi_g$ is hopeless on its own. The fix is to fit a mean-dispersion
trend across all genes and shrink toward it:

$$\log\hat\phi_g \ \big|\ \phi^{\text{trend}}(\bar\mu_g) \ \sim\
\mathcal{N}\big(\log\phi^{\text{trend}}(\bar\mu_g),\ \sigma_{\mathrm{lfc}}^2\big), \tag{20.3}$$

and take the posterior mode as $\hat\phi_g^{\mathrm{MAP}}$. Genes with extreme
gene-wise dispersion are shrunk least (they are flagged as outliers), genes with
little information are shrunk most. This is (5.1) in action: accepting bias to
crush variance.

**Quasi-likelihood F-test (edgeR).** Rather than assuming $\hat\phi_g$ is known,
a QL model adds a gene-specific quasi-dispersion $\sigma_g^2$ on top of the NB
trended dispersion, shrinks it with empirical Bayes, and tests with an
$F$-distribution that carries the uncertainty of that estimate:

$$F_g = \frac{\text{deviance-based statistic}}{\tilde\sigma_g^2}
\ \sim\ F_{d_1,\; d_2 + d_0}. \tag{20.4}$$

The extra $d_0$ df from the prior is why QL tests control the error rate better
than likelihood-ratio tests at small $n$ - this is the currently recommended
edgeR workflow.[^rna3]

**voom: turning counts into weighted Gaussians.** Compute
$y_{gj} = \log_2\!\big((K_{gj}+0.5)/(N_j+1)\times 10^6\big)$, fit a LOWESS curve
of residual standard deviation against mean log-count, and derive a **per
observation** precision weight

$$w_{gj} = \frac{1}{\big[\mathrm{lo}(\hat\mu_{gj})\big]^{4}}, \tag{20.5}$$

then run weighted `limma` linear models. voom's advantage is access to the whole
linear-model toolkit - arbitrary designs, `duplicateCorrelation` for repeated
measures, contrasts - while still respecting the mean-variance trend.

**Moderated t (limma's empirical Bayes), the workhorse of small-$n$ omics.**
Shrink each gene's residual variance toward a pooled prior:

$$\tilde{s}_g^2 = \frac{d_0 s_0^2 + d_g s_g^2}{d_0 + d_g},
\qquad
\tilde{t}_g = \frac{\hat\beta_g}{\tilde{s}_g\sqrt{\mathbf{c}^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{c}}}
\ \sim\ t_{d_g + d_0}. \tag{20.6}$$

$(d_0, s_0^2)$ are estimated from the distribution of all $G$ sample variances.
Equation (20.6) buys you $d_0$ extra degrees of freedom - often the equivalent of
several extra samples - and removes the pathology where a gene with an
accidentally tiny $s_g$ tops the list.

**Log fold-change shrinkage.** The MLE of $\beta_g$ is wildly noisy for
low-count genes. `apeglm`/`ashr` place a heavy-tailed prior on $\beta_g$ and
report the posterior:

$$\hat\beta_g^{\text{shrunk}} = \arg\max_{\beta}\ \big[\ell_g(\beta) + \log \pi(\beta)\big]. \tag{20.7}$$

**Use shrunken LFCs for ranking, visualisation and thresholds; use the
unshrunken test statistic for the hypothesis test.** Mixing them up (ranking by
raw LFC) puts noise at the top of your gene list.

**Independent filtering.** Remove genes with low counts *before* multiplicity
adjustment using an outcome-independent criterion - e.g. edgeR's
`filterByExpr`, which requires a minimum CPM in at least the size of the
smallest group. Because mean expression is independent of the p-value under the
null but strongly predicts power, this raises discoveries without breaking FDR
control (Topic 8).

**Testing against a threshold.** To find genes whose change exceeds a
biologically meaningful $\tau$, test $H_0: |\beta_g| \le \tau$ (DESeq2's
`lfcThreshold`, limma's `treat`) rather than filtering a
$\beta_g = 0$ result list by $|\hat\beta_g| > \tau$. The latter has no error
control for the threshold claim.

### Diagnostics

Dispersion-vs-mean plot (should show shrinkage toward a smooth trend), MA plot
(symmetric around zero; asymmetry means a normalisation problem), p-value
histogram (6.2), sample-distance heatmap and PCA coloured by every covariate,
and Cook's distance for count outliers.

### Decision rules

1. Raw integer counts into the count model; never rounded TPMs.
2. Put every known nuisance factor in the design (19.4) rather than correcting
   the count matrix.
3. Filter low counts by an outcome-independent rule, before FDR.
4. Report shrunken LFC + adjusted p-value + base mean, and verify your contrast
   encodes the comparison you meant.


## Topic 21: Single-cell RNA-seq: replicated inference

**Module:** `31_single_cell_pseudobulk`

### The question

You have 200,000 cells from 8 donors. What is $n$? The answer is 8, and
everything else follows.[^sc1][^sc2][^sc3][^sc4]

### Key equations

**The pseudoreplication inflation, restated for single cells.** Cells within a
donor share $b_i$ in (1.4). Treating $N = \sum_i m_i$ cells as independent
inflates the test statistic by roughly $\sqrt{1+(m-1)\rho}$ from (1.8). With
$m = 2000$ cells/donor and $\rho = 0.02$, that is $\sqrt{40.98} \approx 6.4$ -
so a true $t = 0.5$ is reported as $t = 3.2$. This is the mechanism behind the
documented false-discovery inflation of naive per-cell tests.[^sc2][^sc4]

**Pseudobulk aggregation**: the recommended default. For cell type $c$, sample
$i$, gene $g$:

$$K^{\text{pb}}_{gci} = \sum_{k \in \text{cells}(c,i)} K_{gk}. \tag{21.1}$$

Then analyse $\{K^{\text{pb}}_{gci}\}_i$ with the bulk machinery of Topic 20 -
DESeq2/edgeR/limma-voom - where $n$ is the number of samples, exactly as it
should be.

**Sum, not mean of normalised values.** Summing raw counts preserves the
count nature *and* the information about how many cells contributed, so that
sample-cell-type combinations with more cells naturally receive more weight
through the NB variance. Averaging normalised expression discards that and gives
a 3-cell pseudobulk the same influence as a 3000-cell one. Guard against
low-information pseudobulks with a minimum-cell rule (commonly $\ge 10$ cells
per sample x cell type) applied **before** looking at the outcome.

**Mixed model alternative.** Keep cell-level data and model the hierarchy:

$$\log \mathbb{E}[K_{gk}] = \log s_k + \mathbf{x}_{i(k)}^\top\boldsymbol\beta_g + u_{i(k)},
\qquad u_i \sim \mathcal{N}(0,\sigma_{u,g}^2), \tag{21.2}$$

a Poisson/NB GLMM with a donor random intercept. This retains
within-donor information (useful with very unbalanced cell numbers or
cell-level covariates) at much higher computational cost and with convergence
issues across 20,000 genes. For most two-condition designs, (21.1) and (21.2)
give similar answers, and (21.1) is easier to defend and audit.

**Differential expression vs differential state vs differential abundance.**

$$\underbrace{\text{DE}}_{\text{marker genes between cell types}} \ne
\underbrace{\text{DS}}_{\text{expression change within a cell type across conditions}} \ne
\underbrace{\text{DA}}_{\text{change in the proportion of that cell type}}. \tag{21.3}$$

Only DS and DA are condition-level claims requiring replication; marker-gene DE
between clusters defined on the same data is **circular** (Topic 18) and its
p-values are not valid.

**Differential abundance is compositional.** Cell-type proportions per sample sum
to 1, so (2.7) applies: a genuine expansion of one population forces apparent
contraction of the others. Analyse counts with a total-cell offset,

$$Y_{ci} \sim \mathrm{NB}(\mu_{ci}, \phi_c), \qquad
\log\mu_{ci} = \log(\text{total cells}_i) + \mathbf{x}_i^\top\boldsymbol\beta_c, \tag{21.4}$$

or use a compositional method (Topic 26). Never run a t-test on raw proportions
and report all significant populations as independently changed.

**Normalisation is task-specific.** Log-normalised or SCTransform values are for
visualisation, clustering and embedding. Count-based testing uses raw counts
with an offset. Do not feed log-normalised values into a negative-binomial model.

**Batch/integration caution.** Integration (Harmony, MNN, scVI) produces a
corrected *representation* for clustering. It does not produce valid corrected
expression values for inference. Keep an inferential path that returns to raw
counts with batch in the design (19.4).

### Decision rules

1. Donors are replicates. Cells are measurements. Always.
2. Pseudobulk by (cell type x sample) with sums; require a minimum cell count;
   analyse with bulk tools.
3. Separate DS from DA, and treat DA compositionally.
4. Multiplicity family = genes x cell types x contrasts. Say which you adjusted.
5. Never use cluster-marker p-values as evidence of a condition effect.


## Topic 22: Flow, mass, and imaging cytometry

**Module:** `32_cytometry_differential_abundance`

### The question

Millions of events, tens of markers, but a handful of specimens. The inferential
unit is the specimen.[^cyt1][^cyt2]

### Key equations

**Transformation.** Fluorescence and mass intensities span decades and can be
negative after compensation, so use arcsinh (2.5) with a modality-appropriate
cofactor: $c = 5$ for CyTOF, $c \approx 150$ for flow. Clustering and gating
happen on the transformed scale.

**From events to sample-level statistics.** After clustering, reduce to two
sample-level tables:

$$\text{counts } Y_{ci} \ (\text{cluster } c, \text{ sample } i), \qquad
\text{state } \bar{m}_{cji} = \operatorname{median}_{k \in (c,i)} m_{jk}. \tag{22.1}$$

**Differential abundance** uses a count model with a total-events offset - the
same structure as (21.4), which is why `diffcyt` uses edgeR internally:

$$Y_{ci} \sim \mathrm{NB}(\mu_{ci},\phi_c), \qquad
\log \mu_{ci} = \log(\text{total}_i) + \mathbf{x}_i^\top\boldsymbol\beta_c. \tag{22.2}$$

**Differential state** uses limma on the median (or a suitable summary) marker
intensity per cluster per sample:

$$\bar{m}_{cji} = \mathbf{x}_i^\top\boldsymbol\beta_{cj} + \varepsilon, \tag{22.3}$$

with empirical-Bayes moderation (20.6) across the cluster x marker grid.

**Rare populations.** With $Y_{ci}$ small, the NB variance is dominated by the
Poisson term. The relative standard error of a proportion estimate is
approximately

$$\mathrm{RSE} \approx \frac{1}{\sqrt{Y_{ci}}}, \tag{22.4}$$

so 100 events gives 10% RSE and 25 events gives 20%. Set a minimum-event
threshold from (22.4), pre-specified and outcome-independent.

**Batch/acquisition day** is a first-order effect in cytometry: include it in the
design (19.4), use bead normalisation and a bridging reference sample in every
run, and check marker-wise distributions per day.

**Multiplicity family** = clusters x markers x contrasts. A 20-cluster x 15-state
marker panel with 3 contrasts is 900 hypotheses.

### Decision rules

1. Specimen/donor is $n$; events are not replicates.
2. Counts with an offset for abundance; moderated linear models on medians for
   state.
3. Pre-specify minimum event counts and cluster resolution; report sensitivity
   to both.
4. Model acquisition batch; verify with reference samples.


## Topic 23: DNA methylation and epigenomics

**Module:** `33_methylation_epigenomics`

### The question

Methylation is a bounded proportion measured with strong probe-level and
cell-composition confounding. On which scale do we model, and what is the
confounder?[^meth1][^meth2]

### Key equations

**Beta and M-values.** From methylated and unmethylated intensities,

$$\beta = \frac{M_{\text{int}}}{M_{\text{int}} + U_{\text{int}} + \alpha}, \qquad
M = \log_2\frac{\beta}{1-\beta}. \tag{23.1}$$

The offset $\alpha$ (typically 100) stabilises low-intensity probes. The
variance of a beta value is intrinsically heteroscedastic - near 0 or 1 it is
compressed by the boundary - so linear models on $\beta$ have systematically
wrong standard errors for extreme probes. **Model on M, report on $\beta$.**[^meth1]
Convert an M-value effect back for reporting with
$\beta = 2^M/(2^M+1)$.

**Bisulfite sequencing** gives counts, not intensities: $y_{ij}$ methylated of
$N_{ij}$ total reads. Use a **beta-binomial** GLM:

$$y_{ij}\mid \pi_{ij} \sim \mathrm{Bin}(N_{ij},\pi_{ij}), \quad
\pi_{ij}\sim\mathrm{Beta}(a_{ij},b_{ij}), \quad
\operatorname{logit}\mathbb{E}[\pi_{ij}] = \mathbf{x}_j^\top\boldsymbol\beta_i, \tag{23.2}$$

whose variance $N\pi(1-\pi)[1+(N-1)\rho]$ carries the overdispersion that a
plain binomial misses. Coverage varies enormously across CpGs, and a binomial
model would treat a 200x site as 100x more certain than a 2x site without
accounting for biological variability.

**Cell composition is the dominant confounder.** Whole blood methylation mostly
measures *which cells are present*. The standard reference-based deconvolution
(Houseman) solves a constrained regression of the sample profile against a
reference matrix $\mathbf{R}$ of cell-type-specific methylation:

$$\hat{\mathbf{w}}_j = \arg\min_{\mathbf{w}\ge 0,\ \sum_c w_c = 1}
\big\|\mathbf{y}_j - \mathbf{R}\mathbf{w}\big\|_2^2. \tag{23.3}$$

Then decide, from biology, whether the estimated proportions $\hat{\mathbf{w}}$
are a **confounder** (adjust for them), a **mediator** (adjusting removes the
effect you wanted - see Topic 32), or the **outcome** itself.

**Probe-wise testing** is limma on M-values with empirical Bayes (20.6), followed
by BH across ~850,000 probes. Mandatory filtering, all outcome-independent:
detection p-value failures, cross-reactive probes, probes overlapping common
SNPs, and sex chromosomes (unless sex is the question).

**Regions, not just positions.** Neighbouring CpGs are spatially correlated, so
a DMR is more interpretable and better powered than isolated DMPs. Region-level
methods combine adjacent statistics with a kernel and assess significance by
permutation, accounting for the correlation that makes naive combination
anticonservative.

**Differential variability** is a distinct hypothesis, tested on
$|M_{ij} - \tilde{M}_{i\cdot}|$ (absolute deviations from the group median) with
a moderated test. Changes in *variance* without mean changes are a real
epigenetic phenomenon.

**Enrichment bias.** Genes are represented by unequal numbers of probes on the
array (promoter CpG islands are densely covered), so a naive gene-set test on
"genes with a significant CpG" is biased toward gene length / probe count.
Correct for probes-per-gene (this is what `missMethyl::gometh` does).[^meth1]

**Epigenetic clocks.** A clock is a penalised regression, so everything in
Topic 31 applies: normalisation and feature selection inside the fold, split by
donor, and validate externally. "Age acceleration" is the *residual* from
regressing predicted on chronological age; using raw predicted age as a
phenotype re-introduces the age signal you meant to remove.

### Decision rules

1. Model M, report $\beta$. Model bisulfite counts as beta-binomial.
2. Decide the causal role of cell composition before adjusting.
3. Filter probes by outcome-independent quality criteria first.
4. Use region-level inference for spatially correlated CpGs.
5. Correct gene-set tests for probe-number bias.


## Topic 24: Genotypes, GWAS, and statistical genetics

**Module:** `34_gwas_association`

### The question

A million variants, population structure, relatedness, and linkage
disequilibrium. What makes an association real?[^gwas1]

### Key equations

**Additive model.** Code the genotype as allele dosage $g \in \{0,1,2\}$:

$$\mathbb{E}[y_i \mid g_i] = \beta_0 + \beta_g g_i + \mathbf{z}_i^\top\boldsymbol\gamma. \tag{24.1}$$

$\beta_g$ is the per-allele effect. The additive coding has good power even when
the true model is dominant or recessive, which is why it is the default; test
non-additive models only with a reason.

**Hardy-Weinberg equilibrium** as a QC filter. With allele frequency $p$, the
expected genotype counts are $n p^2, 2np(1-p), n(1-p)^2$, and

$$X^2_{\mathrm{HWE}} = \sum_{k} \frac{(O_k - E_k)^2}{E_k} \ \sim\ \chi^2_1. \tag{24.2}$$

Strong HWE deviation in *controls* usually indicates genotyping error. Apply it
to controls only - a real disease association can cause HWE deviation in cases.

**Population structure.** Ancestry creates allele-frequency differences
correlated with phenotype, producing confounding at genome-wide scale. Two
standard fixes:

*Principal components of the genotype matrix* added as covariates in (24.1),
usually the top 10.

*Linear mixed model with a genetic relationship matrix*
$\mathbf{K} = \frac{1}{m}\tilde{\mathbf{G}}\tilde{\mathbf{G}}^\top$ (standardised
genotypes), which handles both structure and cryptic relatedness:

$$\mathbf{y} = \mathbf{X}\boldsymbol\beta + \mathbf{u} + \boldsymbol\varepsilon, \qquad
\mathbf{u}\sim\mathcal{N}(\mathbf{0},\sigma_g^2\mathbf{K}). \tag{24.3}$$

The SNP heritability follows directly:
$h^2_{\mathrm{SNP}} = \sigma_g^2/(\sigma_g^2+\sigma_e^2)$.

**Genomic inflation factor.** Under the null, the median of a $\chi^2_1$ is
$0.4549$, so

$$\lambda_{\mathrm{GC}} = \frac{\operatorname{median}(\chi^2_{\text{obs}})}{0.4549}. \tag{24.4}$$

$\lambda \approx 1$ means calibrated. $\lambda > 1.05$ suggests residual
structure - **but** polygenic traits genuinely inflate $\lambda$ with large $n$,
so use LD score regression to separate polygenicity (slope) from confounding
(intercept). A QQ plot of $-\log_{10}p$ is the visual version.

**Genome-wide significance.** The conventional threshold

$$\alpha_{\mathrm{GW}} = 5\times 10^{-8} \approx \frac{0.05}{10^6} \tag{24.5}$$

is a Bonferroni correction for the ~1 million effectively independent common
variants in European-ancestry LD structure. It is not universal: denser
sequencing data and other ancestries need a stricter threshold.

**Linkage disequilibrium.** For alleles $A$ and $B$:

$$D = p_{AB} - p_A p_B, \qquad r^2 = \frac{D^2}{p_A(1-p_A)p_B(1-p_B)}. \tag{24.6}$$

LD means significant variants come in correlated blocks. Clumping ($r^2 < 0.1$
within 250 kb) yields approximately independent signals; conditional analysis
and fine-mapping (computing posterior inclusion probabilities and 95% credible
sets) identify likely causal variants. **The lead SNP is usually not the causal
variant.**

**Rare variants.** Single-variant tests have no power at MAF $< 0.1\%$, so
aggregate within a gene. Burden tests assume one direction of effect; SKAT is a
variance-component score test that does not:

$$Q = \sum_{j=1}^{J} w_j S_j^2, \qquad S_j = \mathbf{g}_j^\top(\mathbf{y}-\hat{\boldsymbol\mu}), \tag{24.7}$$

with $Q$ following a mixture of $\chi^2_1$s. SKAT-O adaptively combines the two.

**Polygenic scores.**

$$\mathrm{PGS}_i = \sum_{j=1}^{m} \hat\beta_j g_{ij}. \tag{24.8}$$

Three failure modes: (i) weights fitted and evaluated on overlapping samples
(leakage); (ii) LD not accounted for (use LDpred/lassosum rather than raw
clumping when possible); (iii) **portability decay**: scores trained in
European-ancestry cohorts typically lose 50-80% of their predictive $R^2$ in
African-ancestry cohorts because LD patterns and allele frequencies differ.
Report the ancestry of both training and test data.

**Mendelian randomisation.** Using variants as instruments for exposure $X$ on
outcome $Y$, the inverse-variance-weighted estimate is

$$\hat\beta_{\mathrm{IVW}} = \frac{\sum_j \hat\beta_{Xj}\hat\beta_{Yj}/\sigma_{Yj}^2}
{\sum_j \hat\beta_{Xj}^2/\sigma_{Yj}^2}. \tag{24.9}$$

Valid only under three assumptions: relevance ($\beta_{Xj}\ne0$; check the
F-statistic $> 10$), independence (no confounding of instrument-outcome), and
**exclusion restriction** (no pathway from instrument to outcome except through
$X$). The third is untestable; MR-Egger's intercept and weighted-median
estimators are sensitivity analyses for horizontal pleiotropy, not proofs.

### Decision rules

1. Do QC first: call rates, MAF, HWE in controls, sex checks, relatedness,
   heterozygosity outliers.
2. Control structure with PCs or an LMM; verify with $\lambda_{\mathrm{GC}}$ and
   a QQ plot.
3. Clump before counting "independent loci"; fine-map before naming a gene.
4. Never evaluate a PGS in samples that contributed to its weights, and report
   ancestry transferability.


## Topic 25: Proteomics and metabolomics

**Module:** `35_proteomics_missing_values`

### The question

Continuous intensities with run-order drift, plex effects, and 20-50% missing
values whose mechanism depends on abundance.[^prot1][^prot2]

### Key equations

**Log transform first.** MS intensities are multiplicative-error, right-skewed,
and span orders of magnitude:

$$y = \log_2(\text{intensity}). \tag{25.1}$$

On the log scale, differences are log fold changes (5.11) and the variance is
approximately constant - both prerequisites for linear models.

**Normalisation.** Median (or quantile, or VSN) normalisation removes
sample-loading differences:

$$y^{*}_{gj} = y_{gj} - \operatorname{median}_g(y_{gj}) + \operatorname{median}_j\operatorname{median}_g(y_{gj}). \tag{25.2}$$

For metabolomics with pooled QC samples injected throughout the run, fit a
LOESS curve of QC intensity against injection order per feature and correct the
drift. Always plot intensity vs run order before and after.

**Peptide-to-protein summarisation.** Median polish or a robust linear model on
peptide-level data,

$$y_{pij} = \mu_i + \text{peptide}_p + \text{sample}_j + \varepsilon, \tag{25.3}$$

is far more robust than summing intensities, because peptides differ by orders
of magnitude in ionisation efficiency and a missing high-intensity peptide would
otherwise look like protein down-regulation. Shared (razor) peptides should be
excluded or explicitly modelled.

**Missingness is abundance-dependent.** Plot per-protein missingness rate vs
mean observed intensity; a strong negative slope is the signature of
left-censoring (Topic 15). The right treatment is a mixture:

- Proteins missing **systematically in one condition** -> MNAR; impute from a
  down-shifted distribution or model with (15.8).
- Proteins missing **sporadically** -> MAR; use a MAR-appropriate imputer (kNN,
  MissForest) or a model that tolerates missingness.

Applying one imputer to both classes is the standard mistake, and it creates
either false positives (down-shifting MAR values) or false negatives
(mean-imputing MNAR values).

**Testing with moderated linear models.** With few replicates, borrow strength
across proteins exactly as in (20.6). Include plex/batch/run in the design:

$$y_{gj} = \beta_{0g} + \beta_{1g}\,\text{condition}_j + \gamma_{g,\text{plex}(j)} + \varepsilon_{gj}. \tag{25.4}$$

For TMT designs, the plex is a **block**, and the within-plex comparison is the
precise one - design so that each plex contains a balanced set of conditions plus
a common reference channel.

### Decision rules

1. Log-transform, then normalise, then summarise to protein level - in that
   order, and inspect after each step.
2. Diagnose the missingness mechanism before choosing an imputer, and classify
   features into MNAR and MAR groups.
3. Model plex/batch/run-order structure explicitly.
4. Report how many proteins were quantified in how many samples; a "significant"
   protein observed in 3 of 20 samples is a QC finding, not a biological one.


## Topic 26: Microbiome and compositional data

**Module:** `36_microbiome_compositional`

### The question

Sequencing measures *relative* abundance. Which questions can relative data
answer, and which require absolute quantification?[^mb1][^mb2]

### Key equations

**The problem, stated precisely.** Observed counts are a multinomial sample of
unknown depth from unknown absolute abundances $\mathbf{A}$:

$$\mathbf{Y}_j \sim \mathrm{Multinomial}\big(N_j,\ \mathbf{A}_j/\textstyle\sum_d A_{dj}\big). \tag{26.1}$$

Only the *ratios* in $\mathbf{A}_j$ are identifiable. If one taxon blooms, every
other taxon's relative abundance falls even with constant absolute abundance.
By (2.7) the induced covariances must sum to zero.

**Log-ratio transformations** map the simplex to real space, where standard
methods are valid. With $g(\mathbf{x}) = \big(\prod_d x_d\big)^{1/D}$ the
geometric mean:

$$\operatorname{clr}(\mathbf{x}) = \left(\log\frac{x_1}{g(\mathbf{x})},\dots,\log\frac{x_D}{g(\mathbf{x})}\right), \tag{26.2}$$

$$\operatorname{alr}(\mathbf{x}) = \left(\log\frac{x_1}{x_D},\dots,\log\frac{x_{D-1}}{x_D}\right). \tag{26.3}$$

clr is symmetric across taxa but its components sum to zero, so the clr
covariance matrix is singular (rank $D-1$) - you cannot invert it for a
graphical model without a further step. alr breaks the symmetry by picking a
reference; ilr uses an orthonormal basis and is non-singular but not directly
interpretable per taxon. The **Aitchison distance** (18.5) is the Euclidean
distance between clr vectors and is the principled distance for compositions.

**Zeros are the practical obstacle.** $\log 0$ is undefined, and microbiome
tables are 70-90% zeros. Distinguish:
- *Sampling zeros* - present but not observed at this depth. A pseudocount or a
  Bayesian-multiplicative replacement is reasonable.
- *Structural zeros* - genuinely absent from that environment. These should be
  modelled, not filled.

Report the pseudocount and show that conclusions survive a change in it: a
sensitivity analysis over $c \in \{0.5, 1, \text{Bayesian}\}$ is cheap and
frequently changes results.

**ANCOM-BC's bias correction.** The model recognises that observed counts
reflect a sample-specific sampling fraction $d_j$:

$$\mathbb{E}\big[\log Y_{dj}\big] = \log A_{dj} + \log d_j, \tag{26.4}$$

and estimates the nuisance $\log d_j$ from the data (as a location shift common
to all taxa within a sample), then tests taxon-wise linear models on the
bias-corrected log abundances with covariates, repeated measures and
multiplicity control.[^mb1][^mb2] This is why ANCOM-BC2 controls FDR where
naive relative-abundance tests do not.

**Alpha diversity.** Within-sample summaries:

$$H = -\sum_{d} p_d \ln p_d \ \ (\text{Shannon}), \qquad
D_{\text{Simpson}} = 1 - \sum_d p_d^2, \qquad
\text{Chao1} = S_{\text{obs}} + \frac{f_1^2}{2f_2}. \tag{26.5}$$

All are **depth-dependent**: deeper sequencing finds more taxa. Rarefy to a
common depth *for diversity estimation* or use a coverage-based estimator -
this is one of the few places where rarefying is defensible, unlike for
differential abundance where it discards data.

**Beta diversity and PERMANOVA.** Test group differences on a distance matrix by
partitioning sums of squares (from the distances alone, via the Gower-centred
matrix) and permuting labels:

$$F = \frac{\mathrm{SS}_{\text{between}}/(a-1)}{\mathrm{SS}_{\text{within}}/(N-a)},
\qquad p \ \text{from } B \ \text{label permutations (4.12)}. \tag{26.6}$$

**The essential caveat:** PERMANOVA is sensitive to differences in *dispersion*
as well as in *location*. A significant result may mean "the groups have
different centroids" or "one group is more variable". Always run a dispersion
test (`betadisper`/PERMDISP) alongside, and report both.

**Correlation networks.** Pearson correlation on relative abundances is invalid
by (2.7). Use proportionality ($\varphi$, $\rho_p$) or SparCC/SPIEC-EASI, which
operate on log-ratios and estimate a sparse precision matrix (10.6).

### Decision rules

1. State whether the claim is about relative or absolute abundance. Only spike-in
   or qPCR-anchored data support absolute claims.
2. Transform with log-ratios before distance, ordination, correlation or PCA.
3. Handle zeros explicitly; report the choice and a sensitivity analysis.
4. Use bias-corrected differential abundance (26.4), and pair every PERMANOVA
   with a dispersion test.
5. Rarefy only for diversity, never for differential abundance.


## Topic 27: Spatial transcriptomics and spatial omics

**Module:** `37_spatial_omics`

### The question

Neighbouring spots are correlated by construction. What is a spatial claim, and
what is a patient-level claim?[^spa1]

### Key equations

**Spatial autocorrelation: Moran's I.** With spatial weights $w_{ij}$
(e.g. $k$-nearest-neighbour or a distance kernel) and $W = \sum_{i\ne j} w_{ij}$:

$$I = \frac{n}{W}\cdot\frac{\sum_{i}\sum_{j} w_{ij}(x_i-\bar{x})(x_j-\bar{x})}
{\sum_i (x_i-\bar{x})^2}, \qquad \mathbb{E}_0[I] = -\frac{1}{n-1}. \tag{27.1}$$

$I$ is a spatially weighted correlation: positive means nearby locations are
similar. Note $\mathbb{E}_0[I] = -1/(n-1)$, **not zero**: a common error in
reporting.

**Geary's C** is more sensitive to local differences:

$$C = \frac{(n-1)\sum_i\sum_j w_{ij}(x_i-x_j)^2}{2W\sum_i (x_i-\bar{x})^2},
\qquad \mathbb{E}_0[C] = 1. \tag{27.2}$$

Significance is best assessed by permuting values over locations, because the
normal approximation is poor for irregular lattices and skewed expression.

**Spatially variable genes.** The Gaussian-process formulation decomposes
expression variance into a spatial and a non-spatial part:

$$\mathbf{y}_g \sim \mathcal{N}\big(\mu_g\mathbf{1},\ \sigma_{s,g}^2\mathbf{K}(\boldsymbol\ell) + \sigma_{n,g}^2\mathbf{I}\big), \tag{27.3}$$

with $\mathbf{K}$ a kernel on the spatial coordinates (e.g. squared exponential
with length scale $\ell$). The test is $H_0: \sigma_{s,g}^2 = 0$ - a
variance-component test on the boundary, so the null is again a $\chi^2$ mixture
(Topic 14), not $\chi^2_1$.

**Counts remain counts.** Visium spots have widely varying total UMI, so use
an offset:

$$\log\mathbb{E}[Y_{gs}] = \log(\text{total}_s) + f_g(\text{location}_s). \tag{27.4}$$

**Deconvolution.** Each spot mixes several cells, so spot expression is
approximately a convex combination of cell-type profiles:

$$\mathbb{E}[\mathbf{y}_s] = \sum_{c} w_{cs}\boldsymbol\theta_c, \qquad
w_{cs}\ge0,\ \sum_c w_{cs} = 1. \tag{27.5}$$

The estimated $\hat{w}_{cs}$ carry substantial uncertainty that downstream
analyses almost always ignore. Propagate it (bootstrap or posterior draws) or
state the limitation.

**Colocalisation.** Whether two cell types are closer than chance is assessed
against a null of random labelling - permute cell-type labels while keeping
coordinates fixed, which preserves tissue geometry and density. Use a
multi-scale summary (a function of radius, such as Ripley's cross-$K$), because
colocalisation is scale-dependent: two types can be co-localised at 50 µm and
segregated at 500 µm.

**The replication trap, which is the central issue.** A tissue section contains
thousands of spots but is **one observation of one patient**. Spot-level
p-values describe that section. A condition-level claim needs multiple patients:
compute a per-section summary (mean domain composition, Moran's I, cell-type
proportion), then model those $n_{\text{patients}}$ summaries - or use a mixed
model with a patient random effect (Topic 14). This is exactly (1.5) again, and
recent spatial methods are explicitly moving toward replicated, sample-level
comparisons.[^spa1]

**Edge effects.** Locations near the tissue boundary have fewer neighbours,
biasing $I$ and $K$. Use a toroidal correction, a buffer zone, or an
edge-corrected estimator.

### Decision rules

1. Never treat spots or cells as patient replicates.
2. Build the neighbourhood graph deliberately ($k$, radius, kernel) and report
   sensitivity to it.
3. Use permutation nulls that preserve tissue geometry.
4. Keep counts as counts with a total-UMI offset (27.4).
5. Propagate deconvolution uncertainty or state that you did not.


## Topic 28: Time-course, longitudinal, and survival data

**Module:** `20_survival_and_longitudinal` (core) and `38_survival_biomarkers` (applied)

### The question

Time enters as a trajectory (repeated measurements) or as an outcome
(time-to-event with censoring). These need different machinery.[^surv1]

### Part A: longitudinal trajectories

**Do not analyse repeated visits as independent.** The model is (14.1) with a
random intercept and slope:

$$y_{ij} = \beta_0 + b_{0i} + (\beta_1 + b_{1i})t_{ij} + \mathbf{x}_i^\top\boldsymbol\gamma + \varepsilon_{ij}, \tag{28.1}$$

$$\begin{pmatrix} b_{0i}\\ b_{1i}\end{pmatrix} \sim
\mathcal{N}\!\left(\mathbf{0},\ \begin{pmatrix}\tau_0^2 & \tau_{01}\\ \tau_{01} & \tau_1^2\end{pmatrix}\right). \tag{28.2}$$

The **treatment x time interaction** is the estimand for "do the groups diverge
over time"; a main effect of treatment in (28.1) only describes the baseline
difference.

**Nonlinear trajectories.** Replace $t$ with a spline basis (11.13), and test
the difference between smooth curves with a partial F (11.6) comparing
`y ~ ns(t,df) + group` against `y ~ ns(t,df)*group`. This is the standard
approach for expression time courses.

**Cross-sectional $\ne$ longitudinal.** A cross-sectional age association mixes
between-person differences (cohort effects, survival bias) with within-person
change. Separate them explicitly by including both the person-mean and the
within-person deviation:

$$y_{ij} = \beta_B\,\bar{t}_i + \beta_W\,(t_{ij}-\bar{t}_i) + \dots \tag{28.3}$$

$\beta_W$ is the within-person effect - usually the biological question.

### Part B: survival

**Fundamental quantities.**

$$S(t) = \Pr(T>t), \qquad
h(t) = \lim_{\Delta\to0}\frac{\Pr(t\le T<t+\Delta \mid T\ge t)}{\Delta}
= -\frac{d}{dt}\log S(t), \tag{28.4}$$

$$S(t) = \exp\!\left(-\int_0^t h(u)\,du\right) = \exp\big(-H(t)\big). \tag{28.5}$$

**Censoring.** With right-censoring we observe $(\tilde{T}_i, \delta_i)$ where
$\tilde{T}_i = \min(T_i, C_i)$ and $\delta_i = \mathbb{1}\{T_i \le C_i\}$.
Validity requires **non-informative censoring**: $T \perp C$ given covariates.
Dropping censored subjects, or treating the censoring time as an event, both
bias the estimate badly.

**Kaplan-Meier.** At distinct event times $t_{(1)}<\dots<t_{(k)}$ with $d_i$
events among $n_i$ at risk:

$$\hat{S}(t) = \prod_{t_{(i)}\le t}\left(1 - \frac{d_i}{n_i}\right), \tag{28.6}$$

with **Greenwood's** variance

$$\widehat{\operatorname{Var}}\big(\hat{S}(t)\big) = \hat{S}(t)^2
\sum_{t_{(i)}\le t}\frac{d_i}{n_i(n_i-d_i)}. \tag{28.7}$$

Build CIs on the $\log(-\log S)$ scale so they stay inside $[0,1]$. Always
display the number-at-risk table: the right tail of a KM curve is estimated from
very few subjects and looks far more precise than it is.

**Log-rank test.** Summing over event times, with $E_{1i}$ the expected events in
group 1 under $H_0$ and $V_i$ the hypergeometric variance:

$$Z = \frac{\sum_i (d_{1i} - E_{1i})}{\sqrt{\sum_i V_i}} \ \dot\sim\ \mathcal{N}(0,1),
\qquad E_{1i} = d_i\frac{n_{1i}}{n_i},\quad
V_i = \frac{d_i(n_i-d_i)n_{1i}n_{2i}}{n_i^2(n_i-1)}. \tag{28.8}$$

The log-rank test is most powerful under **proportional hazards** and can have
near-zero power when curves cross.

**Cox proportional hazards.**

$$h_i(t) = h_0(t)\exp\big(\mathbf{x}_i^\top\boldsymbol\beta\big), \tag{28.9}$$

with $h_0$ left unspecified. $\boldsymbol\beta$ is estimated from the partial
likelihood, which conditions on the risk set at each event and so cancels $h_0$:

$$L(\boldsymbol\beta) = \prod_{i:\ \delta_i=1}
\frac{\exp(\mathbf{x}_i^\top\boldsymbol\beta)}{\sum_{j\in R(t_i)}\exp(\mathbf{x}_j^\top\boldsymbol\beta)}. \tag{28.10}$$

$e^{\beta_k}$ is a **hazard ratio**: a ratio of instantaneous event rates among
those still at risk, *not* a risk ratio and not a ratio of survival times.
Effective sample size for a Cox model is the **number of events**, not the
number of subjects - the usual guidance is $\ge 10$ events per covariate.

**Checking proportional hazards.** Scaled Schoenfeld residuals should have zero
slope against (transformed) time:

$$\mathbb{E}\big[s^{*}_{ik}\big] + \hat\beta_k \approx \beta_k(t). \tag{28.11}$$

If PH fails: stratify on the offending covariate, add a time-varying
coefficient $\beta_k(t)$, or switch estimand to restricted mean survival time,

$$\mathrm{RMST}(\tau) = \int_0^{\tau} S(t)\,dt, \tag{28.12}$$

whose difference between groups is interpretable in months of life gained and
requires no PH assumption. RMST is increasingly preferred when hazards are
non-proportional.

**Competing risks.** With a competing event, $1 - \hat{S}_{\mathrm{KM}}$
**overestimates** the probability of the event of interest. Use the cause-specific
cumulative incidence function:

$$\mathrm{CIF}_k(t) = \int_0^t S(u^-)\,h_k(u)\,du. \tag{28.13}$$

Model cause-specific hazards for aetiology; use Fine-Gray subdistribution
hazards for absolute risk prediction.

**Two biases to name explicitly.**

*Immortal time bias*: classifying subjects by an exposure that can only occur
after time zero (e.g. "responders") guarantees the exposed group survived long
enough to be classified. Fix with time-dependent covariates or landmark
analysis.

*Optimal cut-point bias*: dichotomising a continuous biomarker at the
"most significant" cut-point and reporting the resulting p-value is a
maximally-selected statistic with a badly inflated Type I error. Keep the
biomarker continuous (with splines if needed), or pre-specify the cut-point and
validate externally.

### Decision rules

1. Repeated measures -> mixed model with treatment x time (28.1); report the
   interaction, and separate between- from within-person effects (28.3).
2. Time-to-event -> Kaplan-Meier + Cox, with a PH check (28.11) and an
   at-risk table.
3. Count events, not subjects, when judging power.
4. Competing risks -> CIF (28.13), never $1-\mathrm{KM}$.
5. Never dichotomise a biomarker at a data-chosen optimal cut-point.


# Part VI: Gene sets and systems-level inference

## Topic 29: Functional enrichment and pathway statistics

**Module:** `39_gene_set_enrichment`

### The question

Your gene list is enriched for "immune response". Against what background, under
which null hypothesis, and does gene-gene correlation invalidate the p-value?
(Usually, yes.)[^gsea1][^gsea2]

### Key equations

**Over-representation analysis (ORA).** With $N$ genes in the universe, $K$ in
the pathway, $n$ in your list and $k$ in the overlap, the null of random
selection gives the hypergeometric distribution:

$$\Pr(X = k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}, \qquad
p = \Pr(X \ge k) = \sum_{x=k}^{\min(n,K)}\Pr(X=x). \tag{29.1}$$

(Equation (29.1) is Fisher's exact test on the $2\times2$ overlap table.) Three
ways ORA goes wrong:

1. **The universe $N$.** It must be the set of genes *that could have been
   detected and tested* - not all annotated genes in the genome. Using the whole
   genome when you tested 12,000 expressed genes makes every p-value too
   *small*: a larger $N$ shrinks the expected overlap $nK/N$, so the observed
   overlap looks more extreme than it is.
2. **The threshold.** A gene at $q = 0.051$ contributes nothing while one at
   $q = 0.049$ contributes fully. ORA discards the ranking.
3. **Detectability bias.** Long, highly expressed genes are more likely to reach
   significance. Pathways enriched for long genes (e.g. neuronal, ECM) come out
   significant in almost any RNA-seq screen unless you correct for it.

**Competitive vs self-contained nulls.** This distinction determines what your
p-value means:

$$H_0^{\text{self}}: \text{no gene in the set is associated with the phenotype}, \tag{29.2}$$

$$H_0^{\text{comp}}: \text{genes in the set are no more associated than genes outside it}. \tag{29.3}$$

Permuting **sample labels** tests (29.2); permuting **gene labels** tests (29.3).
They answer different questions and can disagree. GSEA's classic permutation is
over samples but its null distribution is built across gene sets, which makes it
a hybrid - one reason its p-values have been criticised.

**GSEA enrichment score.** With genes ranked by a statistic $r_g$, walk the
ranked list and accumulate:

$$\mathrm{ES} = \max_{i}\left|\ \sum_{g_j \in S,\ j\le i}\frac{|r_j|^{w}}{N_R}
\;-\; \sum_{g_j \notin S,\ j\le i}\frac{1}{N - N_S}\ \right|, \quad
N_R = \sum_{g_j\in S}|r_j|^{w}. \tag{29.4}$$

With $w = 0$ this is a Kolmogorov-Smirnov statistic; $w = 1$ (the default) weights
by effect magnitude. The ES is normalised (NES) by the mean of the null
distribution for sets of that size, because ES scales with set size.

**Inter-gene correlation destroys the naive null.** Genes in a pathway are
co-regulated, so they are not independent draws. The variance of the mean
statistic in a set of $m$ correlated genes is inflated by exactly the design
effect (1.8):

$$\mathrm{VIF} = 1 + (m-1)\bar\rho, \tag{29.5}$$

where $\bar\rho$ is the mean pairwise inter-gene correlation. With $m = 100$ and
a modest $\bar\rho = 0.05$, $\mathrm{VIF} \approx 5.95$ - the naive test
statistic is inflated by $\sqrt{5.95} \approx 2.4$. **`CAMERA` estimates
$\bar\rho$ from the data and divides it out**, which is the most important
reason to prefer it over an uncorrected competitive test. Note that (29.5) is
the *same* formula as (1.8) and (21.x): correlated units, whether cells in a
donor or genes in a pathway, always inflate variance the same way.

**Rotation tests (ROAST)** give a self-contained test valid for small samples by
rotating residual vectors instead of permuting labels, which works when there are
too few samples for permutation to have resolution ($n = 3$ per group gives only
$\binom{6}{3} = 20$ permutations, so the minimum p-value is $0.05$).

**Multiplicity across pathways.** GO has ~20,000 terms in a deeply nested
hierarchy, so tests are massively redundant - a significant child term drags its
parents along. BH across terms is the minimum; better, report leading-edge genes,
collapse redundant terms (semantic similarity clustering), and show the overlap
structure.

**Identifier and annotation versioning.** Gene symbols change (and Excel still
converts `SEPT7` to a date). Map to stable identifiers (Ensembl/Entrez), record
the annotation database version, and report how many genes failed to map -
unmapped genes silently shrink the universe.

### Decision rules

1. Prefer ranked/competitive tests using the full statistic vector over
   threshold-based ORA.
2. Define the universe as the tested, detectable genes.
3. Use a method that accounts for inter-gene correlation (29.5).
4. Correct for detectability bias (gene length, probe number, expression level).
5. Report leading-edge genes and set overlap, and record annotation versions.


## Topic 30: Networks and multivariate integration

**Module:** `40_networks_and_multiomics`

### The question

Which coordinated programmes exist across features and assays, and would they
appear again in an independent cohort?

### Key equations

**Co-expression networks (WGCNA).** Soft-threshold the correlation to
approximate a scale-free topology:

$$a_{ij} = \big|\operatorname{cor}(x_i,x_j)\big|^{\beta}. \tag{30.1}$$

Choosing $\beta$ by scale-free fit avoids the arbitrariness of a hard cutoff and
emphasises strong correlations. The topological overlap measure adds shared
neighbourhood information, which is more robust to noise than pairwise
correlation:

$$\mathrm{TOM}_{ij} = \frac{\sum_{u\ne i,j} a_{iu}a_{uj} + a_{ij}}
{\min(k_i,k_j) + 1 - a_{ij}}, \qquad k_i = \sum_u a_{iu}. \tag{30.2}$$

Modules come from hierarchical clustering on $1-\mathrm{TOM}$; the **module
eigengene** is the first PC of the module's expression:

$$\mathbf{e}_M = \text{PC1 of } \mathbf{X}_M. \tag{30.3}$$

Module-trait association then regresses $\mathbf{e}_M$ on the phenotype. Note the
sample-size trap: correlations estimated from $n < 20$ samples are so noisy that
modules are unstable. WGCNA's own guidance is $n \ge 15$, preferably $\ge 20$.

**Graphical models.** Marginal correlation confuses direct and indirect
association. The graphical lasso estimates a sparse precision matrix (10.6):

$$\hat{\boldsymbol\Theta} = \arg\max_{\boldsymbol\Theta \succ 0}
\Big\{\log\det\boldsymbol\Theta - \operatorname{tr}(\mathbf{S}\boldsymbol\Theta)
- \lambda\|\boldsymbol\Theta\|_1\Big\}. \tag{30.4}$$

An edge is absent iff $\Theta_{ij}=0$, meaning conditional independence given all
other features. The $\ell_1$ penalty makes (30.4) solvable when $p > n$, where
$\mathbf{S}$ is singular.

**Two-block integration.** Canonical correlation analysis finds paired directions
maximising cross-block correlation:

$$\max_{\mathbf{a},\mathbf{b}}\ \operatorname{cor}\big(\mathbf{X}\mathbf{a},\ \mathbf{Y}\mathbf{b}\big). \tag{30.5}$$

Classical CCA requires $n > p_X + p_Y$ and is useless in omics; **sparse CCA**
adds $\ell_1$ penalties on $\mathbf{a},\mathbf{b}$. PLS instead maximises
*covariance*, which is more stable because it does not divide by nearly-singular
within-block covariance matrices.

**Multi-block latent factors (MOFA-style).** Decompose $M$ assays sharing samples
into common factors plus assay-specific loadings:

$$\mathbf{Y}^{(m)} = \mathbf{Z}\mathbf{W}^{(m)\top} + \mathbf{E}^{(m)}, \qquad m = 1,\dots,M. \tag{30.6}$$

The interpretable output is the **variance explained by each factor in each
assay**: a factor loading on all assays is shared biology (or a shared technical
artefact); a factor confined to one assay is assay-specific. Always check factors
against technical metadata before calling them biology - assays processed in the
same batches share technical factors, which (30.6) will happily merge into a
"shared" component.

**Validation is the whole game.** Modules and factors are unsupervised summaries
and will always be produced. Minimum standards:

- **Preservation statistics** in an independent dataset (WGCNA's $Z_{\text{summary}}$).
- **Permutation null** for module-trait associations.
- **Stability** across subsamples and hyperparameters.
- **Cross-validation at the biological-unit level** if anything is predictive.

### Decision rules

1. Decide the goal - visualisation, latent biology, prediction, or mechanism -
   before picking a method; they have different validation standards.
2. Use partial correlation / graphical models for "direct" association claims.
3. Check that shared factors are not shared batches.
4. Report module preservation or factor reproducibility in independent data.


# Part VII: Prediction and causal reasoning

## Topic 31: Statistical learning and biomarker prediction

**Module:** `21_prediction_and_validation`

### The question

Will this classifier work on the next patient, from the next hospital, on the
next platform? Association and prediction are different objectives with
different validation standards.

### Key equations

**Risk and its decomposition.** For loss $L$ and a fitted $\hat{f}$:

$$R(\hat{f}) = \mathbb{E}_{(X,Y)\sim P}\big[L\big(Y,\hat{f}(X)\big)\big]. \tag{31.1}$$

For squared error the expected test error at $x_0$ decomposes as

$$\mathbb{E}\big[(y_0-\hat{f}(x_0))^2\big] =
\underbrace{\sigma^2}_{\text{irreducible}} +
\underbrace{\mathrm{Bias}^2\big[\hat{f}(x_0)\big]}_{\text{underfitting}} +
\underbrace{\operatorname{Var}\big[\hat{f}(x_0)\big]}_{\text{overfitting}}, \tag{31.2}$$

which is (5.1) plus the noise floor. Regularisation moves you along this
trade-off.

**Cross-validation.**

$$\widehat{\mathrm{CV}} = \frac{1}{K}\sum_{k=1}^{K}
\frac{1}{|F_k|}\sum_{i\in F_k} L\big(y_i, \hat{f}^{-k}(x_i)\big). \tag{31.3}$$

$K=5$ or $10$ balances bias and variance; leave-one-out has low bias but high
variance and is unstable for classification. Repeat CV with several random
partitions and report the spread - a single 5-fold CV estimate has substantial
Monte Carlo noise.

**Nested CV.** If you tune hyperparameters using CV and then report that same CV
score, the score is optimistically biased (you selected on it). The outer loop
estimates performance; the inner loop tunes:

$$\widehat{\mathrm{CV}}_{\text{nested}} = \frac{1}{K_{\text{out}}}\sum_{k}
L\Big(y_{F_k},\ \hat{f}^{-k}_{\hat\lambda^{-k}}(x_{F_k})\Big), \tag{31.4}$$

where $\hat\lambda^{-k}$ is tuned *within* the training part of fold $k$ only.

**Leakage: the dominant cause of irreproducible biomarkers.** Every learned
transformation must be fitted inside the training fold:

| Step | Leaks if fitted on all data |
|---|---|
| Normalisation / scaling / quantile | Yes |
| Imputation (Topic 15) | Yes |
| Batch correction (19.2) | Yes |
| Feature selection / filtering by outcome | Yes, severely |
| Hyperparameter tuning | Yes |
| Splitting cells/repeats of one donor across folds | Yes, severely |

The last row deserves emphasis: **group-aware splitting by donor is mandatory**.
Random splitting of a donor's cells puts near-duplicates in both train and test,
and the model memorises the donor. A classic demonstration of the
feature-selection leak: select the 100 features most correlated with a *random*
outcome using all $n$ samples, then cross-validate - apparent accuracy is far
above 50% on pure noise.

**Penalised regression.**

$$\hat{\boldsymbol\beta}_{\text{ridge}} = \arg\min_{\boldsymbol\beta}
\|\mathbf{y}-\mathbf{X}\boldsymbol\beta\|^2 + \lambda\|\boldsymbol\beta\|_2^2
= (\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}, \tag{31.5}$$

$$\hat{\boldsymbol\beta}_{\text{lasso}} = \arg\min_{\boldsymbol\beta}
\|\mathbf{y}-\mathbf{X}\boldsymbol\beta\|^2 + \lambda\|\boldsymbol\beta\|_1, \tag{31.6}$$

$$\text{elastic net: } \lambda\Big(\alpha\|\boldsymbol\beta\|_1 + \tfrac{1-\alpha}{2}\|\boldsymbol\beta\|_2^2\Big). \tag{31.7}$$

The $\lambda\mathbf{I}$ in (31.5) makes the inverse exist even when $p>n$ - ridge
always has a solution. Lasso's $\ell_1$ geometry has corners on the axes, which is
why it sets coefficients exactly to zero. **But with correlated features (always,
in omics) lasso selects one member of a correlated group arbitrarily**, so its
selected feature set is unstable across resamples. Elastic net's grouping effect
mitigates this. Never interpret lasso-selected features as "the important genes"
without a stability-selection analysis.

**Discrimination.** ROC-AUC has the same probabilistic reading as (7.7):

$$\mathrm{AUC} = \Pr\big(\hat{s}(X_{\text{case}}) > \hat{s}(X_{\text{control}})\big). \tag{31.8}$$

AUC is **prevalence-independent**, which is a virtue for comparing models and a
trap for judging usefulness. With 1% prevalence, an AUC of 0.90 can still give a
positive predictive value below 10%. Report **precision-recall** curves, whose
baseline is the prevalence itself, for imbalanced problems.

**Calibration** is separate from discrimination and is usually ignored.

$$\text{Brier} = \frac{1}{n}\sum_i (\hat{p}_i - y_i)^2, \tag{31.9}$$

which decomposes (Murphy) into

$$\text{Brier} = \underbrace{\text{reliability}}_{\text{miscalibration, want }0}
- \underbrace{\text{resolution}}_{\text{discrimination, want large}}
+ \underbrace{\text{uncertainty}}_{\bar{p}(1-\bar{p}),\ \text{fixed}}. \tag{31.10}$$

The **calibration slope** is the coefficient from regressing the outcome on the
predicted log-odds:

$$\operatorname{logit}\Pr(Y=1) = a + b\cdot\operatorname{logit}(\hat{p}). \tag{31.11}$$

$b = 1$ is perfect; $b < 1$ (the usual finding) means predictions are too extreme
- overfitting. $b$ is also the uniform shrinkage factor you should apply.

**Clinical utility.** Net benefit at threshold probability $p_t$:

$$\mathrm{NB}(p_t) = \frac{\mathrm{TP}}{n} - \frac{\mathrm{FP}}{n}\cdot\frac{p_t}{1-p_t}. \tag{31.12}$$

A decision curve plots (31.12) against $p_t$ versus "treat all" and "treat none".
A model can have excellent AUC and still be useless by (31.12).

**Sample size.** Events-per-variable heuristics (EPV $\ge 10$) are crude;
modern approaches target a small expected optimism in the calibration slope and
typically demand far more data than EPV suggests. High-dimensional biomarker
discovery with $n = 60$ and $p = 20{,}000$ will not produce a reliable
predictor, no matter which algorithm is used.

### Decision rules

1. Split by donor/patient, always. Group-aware CV is not optional.
2. Put every learned step inside the fold; build a single pipeline object so
   this is enforced structurally rather than by discipline.
3. Use nested CV (31.4) whenever you tune.
4. Report discrimination **and** calibration **and** (for clinical models)
   net benefit.
5. Distinguish apparent, internally validated, and externally validated
   performance in the text. Only the third predicts the next cohort.


## Topic 32: Causal inference for observational bioinformatics

**Module:** `22_causal_inference`

### The question

Adjusting for more covariates feels safer. It is not. Which covariates to adjust
for is determined by the assumed causal structure, and some adjustments *create*
bias.

### Key equations

**Estimands.**

$$\mathrm{ATE} = \mathbb{E}[Y(1)-Y(0)], \qquad
\mathrm{ATT} = \mathbb{E}\big[Y(1)-Y(0)\mid A=1\big]. \tag{32.1}$$

**Identification.** The ATE is identified from observational data under
conditional exchangeability, positivity and consistency:

$$\mathrm{ATE} = \mathbb{E}_{\mathbf{Z}}\Big[\mathbb{E}[Y\mid A=1,\mathbf{Z}] - \mathbb{E}[Y\mid A=0,\mathbf{Z}]\Big]. \tag{32.2}$$

Equation (32.2) is **standardisation** (g-formula). Note it averages the
conditional contrast over the covariate distribution - which is *not* the same as
reading a regression coefficient off a model with an interaction, and not the
same as the marginal coefficient for non-collapsible measures (Topic 13).

**DAGs and the backdoor criterion.** A set $\mathbf{Z}$ suffices for adjustment
if it (i) blocks every backdoor path from $A$ to $Y$ and (ii) contains no
descendant of $A$. The three elementary structures:

| Structure | Path | Adjusting for $M$... |
|---|---|---|
| Chain (mediator) | $A \to M \to Y$ | **blocks** the indirect effect - removes real signal |
| Fork (confounder) | $A \leftarrow C \to Y$ | **removes** confounding - do this |
| Collider | $A \to K \leftarrow Y$ | **creates** spurious association - never do this |

**Collider bias is the counterintuitive one and it is everywhere in
bioinformatics.** Selecting samples on a variable caused by both exposure and
outcome (hospital admission, sample availability, cell-quality filters that
depend on both condition and a phenotype, "responders only" cohorts) induces an
association where none exists. Adjusting for a collider does the same thing.

**Propensity score.**

$$e(\mathbf{Z}) = \Pr(A=1\mid\mathbf{Z}). \tag{32.3}$$

The balancing property: conditioning on $e(\mathbf{Z})$ alone suffices -
$A \perp \mathbf{Z} \mid e(\mathbf{Z})$. Inverse probability of treatment
weighting:

$$w_i = \frac{A_i}{\hat{e}(\mathbf{Z}_i)} + \frac{1-A_i}{1-\hat{e}(\mathbf{Z}_i)},
\qquad
\widehat{\mathrm{ATE}} = \frac{1}{n}\sum_i w_i(2A_i-1)Y_i. \tag{32.4}$$

Diagnose with **overlap**: plot the propensity distributions by arm. Propensities
near 0 or 1 produce enormous weights and unstable estimates; that is a positivity
violation, and trimming or restricting to the region of common support is more
honest than extrapolating. Check covariate balance after weighting using
standardised mean differences (target $< 0.1$) - **not** by p-values, which
depend on sample size.

**Doubly robust (AIPW).** Combine an outcome model $\hat{m}_a$ and a propensity
model:

$$\hat\tau_{\mathrm{AIPW}} = \frac{1}{n}\sum_i\left[
\hat{m}_1(\mathbf{Z}_i) - \hat{m}_0(\mathbf{Z}_i)
+ \frac{A_i\big(Y_i-\hat{m}_1(\mathbf{Z}_i)\big)}{\hat{e}(\mathbf{Z}_i)}
- \frac{(1-A_i)\big(Y_i-\hat{m}_0(\mathbf{Z}_i)\big)}{1-\hat{e}(\mathbf{Z}_i)}\right]. \tag{32.5}$$

Consistent if **either** model is correct - a genuine robustness gain.

**Mediation.** Decompose a total effect into direct and indirect parts:

$$\mathrm{TE} = \underbrace{\mathrm{NDE}}_{\text{not through } M} + \underbrace{\mathrm{NIE}}_{\text{through } M}, \tag{32.6}$$

$$\mathrm{NIE} = \mathbb{E}\big[Y\big(1, M(1)\big)\big] - \mathbb{E}\big[Y\big(1, M(0)\big)\big]. \tag{32.7}$$

The classic Baron-Kenny product-of-coefficients approach is valid only without
exposure-mediator interaction and requires **no unmeasured mediator-outcome
confounding** - a strong assumption that randomising $A$ does not deliver. This
is the correct framing for "is the methylation effect mediated by cell
composition?" and for high-dimensional mediation in omics.

**Sensitivity analysis: the E-value.** The minimum strength of association (on
the risk-ratio scale) that an unmeasured confounder would need with *both*
exposure and outcome to explain away an observed $\mathrm{RR}$:

$$\text{E-value} = \mathrm{RR} + \sqrt{\mathrm{RR}\,(\mathrm{RR}-1)}. \tag{32.8}$$

For $\mathrm{RR} = 2$, E-value $= 2 + \sqrt{2} \approx 3.41$. Reporting it turns
"residual confounding is possible" into a quantitative claim a reader can judge.

**Target trial emulation.** Write down the randomised trial you would run -
eligibility, treatment strategies, assignment, time zero, outcome, follow-up,
estimand - and then emulate each component with the observational data.
Mismatches between "time zero" and "eligibility assessment" are the source of
immortal time bias (Topic 28).

### Decision rules

1. Draw the DAG first. Adjustment sets come from the graph, not from
   "everything available" or stepwise selection.
2. Never adjust for a mediator when estimating a total effect; never adjust for
   or select on a collider.
3. Check positivity and covariate balance, not just model fit.
4. State the untestable assumptions explicitly and quantify sensitivity (32.8).
5. Batch, selection and technical filtering are causal structures too - draw them
   in the DAG.


## Topic 33: Bayesian reasoning and hierarchical shrinkage

**Module:** `23_bayesian_and_shrinkage`

### The question

With 3 replicates per group and 20,000 genes, each gene alone is uninformative -
but the *ensemble* is not. How do we borrow strength legitimately?

### Key equations

**Bayes' theorem.**

$$p(\theta\mid y) = \frac{p(y\mid\theta)\,p(\theta)}{p(y)}
\ \propto\ \underbrace{L(\theta;y)}_{\text{likelihood}}\ \underbrace{p(\theta)}_{\text{prior}}. \tag{33.1}$$

A **credible interval** is a probability statement about $\theta$ given the data -
which is what most people wrongly believe a confidence interval to be.

**Conjugate cases worth knowing by heart.**

*Beta-binomial* (allele frequencies, methylation proportions, response rates):

$$\theta\sim\mathrm{Beta}(a,b),\ y\sim\mathrm{Bin}(n,\theta)
\ \Rightarrow\ \theta\mid y \sim \mathrm{Beta}(a+y,\ b+n-y). \tag{33.2}$$

The prior acts as $a+b$ pseudo-observations - an intuitive way to calibrate how
informative it is.

*Normal-normal*, which shows shrinkage explicitly. With
$y\mid\theta\sim\mathcal{N}(\theta,\sigma^2)$ and
$\theta\sim\mathcal{N}(\mu_0,\tau^2)$:

$$\theta\mid y \sim \mathcal{N}\big(\lambda y + (1-\lambda)\mu_0,\ \lambda\sigma^2\big),
\qquad \lambda = \frac{\tau^2}{\tau^2+\sigma^2}. \tag{33.3}$$

The posterior mean is a **precision-weighted average** of the data and the prior.
Compare (33.3) with the mixed-model BLUP (14.6) - they are the same formula.
Empirical Bayes, `limma`'s moderated variance (20.6), DESeq2's dispersion
shrinkage (20.3) and partial pooling are all instances of (33.3).

**Hierarchical model.** The structure that makes borrowing legitimate:

$$y_g \mid \theta_g \sim f(\cdot\mid\theta_g), \qquad
\theta_g \mid \boldsymbol\eta \sim \pi(\cdot\mid\boldsymbol\eta), \qquad
\boldsymbol\eta \sim p(\boldsymbol\eta). \tag{33.4}$$

**Empirical Bayes** estimates $\boldsymbol\eta$ from the marginal distribution of
all $G$ features and then plugs it in. This understates uncertainty slightly (it
ignores the error in $\hat{\boldsymbol\eta}$), but with $G$ in the thousands that
error is negligible - which is exactly why empirical Bayes is so effective in
omics and rarely worth replacing with full MCMC for routine differential
expression.

**Why shrinkage wins: the James-Stein result.** For $G \ge 3$ independent normal
means, the estimator

$$\hat\theta^{\mathrm{JS}} = \left(1 - \frac{(G-2)\sigma^2}{\|\mathbf{y}\|^2}\right)\mathbf{y} \tag{33.5}$$

has **uniformly lower total risk** than the obvious $\hat\theta = \mathbf{y}$,
for every true $\boldsymbol\theta$. The intuition is (5.1): pulling extreme
estimates toward the centre removes far more variance than it adds bias. This is
not a Bayesian assumption - it is a frequentist theorem, and it is the
theoretical licence for every shrinkage method in this document.

**Priors are assumptions, and should be visible.** Prefer weakly informative,
regularising priors (e.g. $\mathcal{N}(0, 2.5^2)$ on standardised logistic
coefficients) over "non-informative" flat priors, which are improper, can produce
improper posteriors, and are not actually neutral after transformation. Always
run a **prior predictive check**: simulate data from the prior and confirm it
produces biologically possible values.

**Posterior predictive check.** Simulate replicated data from the posterior and
compare a test statistic $T$ to the observed value:

$$p_{\text{post}} = \Pr\big(T(\mathbf{y}^{\text{rep}}) \ge T(\mathbf{y}) \mid \mathbf{y}\big). \tag{33.6}$$

Values near 0 or 1 indicate the model cannot reproduce that feature of the data
(e.g. the proportion of zeros, the maximum count). This is the Bayesian analogue
of residual diagnostics and is more informative than any single fit statistic.

**Bayes factors and their fragility.**

$$\mathrm{BF}_{10} = \frac{p(y\mid H_1)}{p(y\mid H_0)}. \tag{33.7}$$

The marginal likelihood in (33.7) depends on the prior even when the posterior
does not (Lindley's paradox: a diffuse prior under $H_1$ drives
$\mathrm{BF}_{10}$ toward favouring $H_0$ regardless of the data). If you report
a Bayes factor, report its sensitivity across prior scales.

**MCMC diagnostics.** Never report a posterior without them:
$\hat{R} < 1.01$ (rank-normalised split-$\hat{R}$), bulk and tail effective
sample size $> 400$, zero divergent transitions (for HMC/NUTS), and visual
inspection of trace plots.

### Decision rules

1. Use empirical Bayes shrinkage whenever you have many parallel features and
   few replicates - it is the single biggest practical win in small-$n$ omics.
2. Report credible intervals as probability statements, and say so; do not call
   them confidence intervals.
3. Make priors explicit, weakly informative, and checked (prior + posterior
   predictive).
4. Check convergence before interpreting anything.


# Part VIII: Reliability and reproducibility

## Topic 34: Model diagnostics and sensitivity analysis

**Module:** `24_diagnostics_and_reproducibility`

### The question

Would this conclusion survive a different reasonable analyst making different
reasonable choices?

### Key equations

**Residual diagnostics by model family.**

| Model | Residual to plot | Looks wrong when |
|---|---|---|
| Linear | Standardised $r_i$ (11.10) vs fitted | Curvature, funnel shape |
| GLM | Deviance or randomised quantile residuals | Trend, overdispersion |
| Mixed | Conditional and marginal residuals, per level | Structure within cluster |
| Cox | Scaled Schoenfeld (28.11), martingale, deviance | Time trend in $\beta$ |
| Count omics | Mean-variance (3.6), dispersion plot | Trend not captured |

**Randomised quantile residuals** deserve a special mention because they solve
the awkwardness of discrete-outcome residuals:

$$r_i = \Phi^{-1}\big(u_i\big), \quad
u_i \sim \mathrm{Uniform}\big(F(y_i - 1;\hat\theta),\ F(y_i;\hat\theta)\big). \tag{34.1}$$

Under a correct model these are exactly $\mathcal{N}(0,1)$, so a Q-Q plot is
interpretable even for Poisson/NB/binomial outcomes - where raw residuals form
useless discrete bands.

**Influence.** Cook's distance (11.11); for GLM/Cox use `dfbeta`:

$$\mathrm{dfbeta}_{ik} = \hat\beta_k - \hat\beta_k^{(-i)}. \tag{34.2}$$

For omics, the analogous question is **leave-one-sample-out** and
**leave-one-batch-out**: refit the whole pipeline $n$ times and record how the
result list changes. If dropping one sample removes 40% of your discoveries,
that is the headline, not a footnote.

**Specification curve / multiverse analysis.** Enumerate the defensible analytic
choices - normalisation $\times$ filtering $\times$ covariate set $\times$
transformation $\times$ test - fit all $S$ combinations, and plot the estimate
and interval for each, ordered by magnitude:

$$\big\{\hat\theta_s, \mathrm{CI}_s\big\}_{s=1}^{S}, \qquad
S = \prod_{c} (\text{options in choice } c). \tag{34.3}$$

Then ask: is the sign stable across the curve? What fraction of specifications
give $p<0.05$? Which single choice moves the result most? A conclusion that
holds in 5 of 48 specifications is a specification-dependent conclusion, and
saying so is a scientific finding.

**Negative and positive controls.** Run the exact pipeline on:
- permuted outcome labels - should produce a flat p-value histogram and
  near-zero discoveries. If not, the pipeline is broken (usually leakage or an
  unmodelled dependence);
- a known-true positive (sex genes with sex, a spike-in, a known treatment
  effect) - should be recovered.

These two checks catch more real bugs than any amount of code review.

**Simulation-based calibration** for any complex pipeline: simulate data from the
assumed model with known parameters, run the analysis, and verify that nominal
95% intervals cover in about 95% of replicates and that the false-positive rate
is at its nominal level.

### Decision rules

1. Diagnose every model you report, using residuals appropriate to its family.
2. Run leave-one-sample-out and leave-one-batch-out for any small-$n$ omics
   result.
3. Pre-register or at least pre-specify the primary analysis; report the
   multiverse around it.
4. Always run the permuted-label negative control.


## Topic 35: Reproducible statistical workflows

**Module:** `24_diagnostics_and_reproducibility`

### The question

Can someone else - including you in a year - regenerate every number, figure and
decision from the raw data?

### Principles, with the statistical reason for each

**Separate immutable inputs from generated outputs.**

```
data/raw/        # read-only, never written by code, checksummed
data/derived/    # regenerable; safe to delete and rebuild
results/         # figures, tables; regenerable
```

If `results/` cannot be deleted and rebuilt by one command, the analysis is not
reproducible.

**Seeds.** Every stochastic step - permutation (4.12), bootstrap (5.12), k-means
(18.10), t-SNE/UMAP (18.14), MCMC, CV splits (31.3) - must be seeded, and the
seed recorded. Seeding once at the top of a script is not enough if execution
order can change; seed at the point of use for anything you will report.
Parallel execution requires a parallel-safe RNG (R: `RNGkind("L'Ecuyer-CMRG")`;
Python: spawn child `Generator`s from a `SeedSequence`) - otherwise workers
silently share or duplicate streams.

**Environment capture.** Record R/Python version, every package version, and the
OS. `sessionInfo()` / `pip freeze` in the output, a lockfile in the repo
(`renv.lock`, `requirements.txt`), and a container image for the strong
guarantee. Statistical results *do* change across versions - default
`p.adjust` methods, RNG algorithms, optimiser defaults and package internals all
shift.

**Annotation provenance.** Gene sets, genome builds and annotation databases are
versioned data. "GO enrichment" without the GO release date and the
organism-annotation version is not reproducible, and results change materially
between releases.

**Literate analysis.** Keep narrative, code and output together (`.Rmd`/`.qmd`,
Jupyter). A figure in a slide deck with no code path to the raw data is an
orphan. Parameterised reports let one document serve many datasets.

**Testing statistical code.** Unit tests for analysis code are different from
software tests - test *properties*:

- **Known-answer:** your permutation t-test reproduces `t.test` on a fixed seed;
  your from-scratch OLS matches `lm` to $10^{-10}$.
- **Invariance:** results are unchanged by row permutation of the input; scaling
  the outcome scales coefficients predictably.
- **Alignment:** assert (2.1) - sample identifiers match between assay and
  metadata - at every join.
- **Null calibration:** under a simulated null, the false-positive rate is
  $\approx\alpha$ (34.x).
- **Boundary:** zero counts, single-sample groups, all-missing features, and
  constant features do not crash or silently return nonsense.

**Machine-readable results.** Write result tables as CSV/TSV/Parquet with the
full statistic vector (estimate, SE, statistic, raw p, adjusted p, n), not just
the filtered significant rows. Someone will want to re-threshold, meta-analyse,
or run a competitive gene-set test - all of which need the complete vector
(Topic 29).

### Decision rules

1. One command rebuilds everything from `data/raw/`.
2. Seed everything stochastic, at the point of use; record the seeds.
3. Pin the environment; ship a container.
4. Version-stamp annotations and gene sets.
5. Export full statistic tables, not filtered lists.
6. Test properties, calibration and alignment - not just that the code runs.


# Appendix A: Distribution reference sheet

| Distribution | Support | PMF / PDF | Mean | Variance | Bioinformatics use |
|---|---|---|---|---|---|
| Bernoulli$(\pi)$ | $\{0,1\}$ | $\pi^y(1-\pi)^{1-y}$ | $\pi$ | $\pi(1-\pi)$ | Mutation present/absent |
| Binomial$(N,\pi)$ | $\{0..N\}$ | $\binom{N}{y}\pi^y(1-\pi)^{N-y}$ | $N\pi$ | $N\pi(1-\pi)$ | Allele counts, methylated reads |
| Beta-binomial | $\{0..N\}$ | see (23.2) | $N\pi$ | $N\pi(1-\pi)[1+(N-1)\rho]$ | Overdispersed bisulfite counts |
| Poisson$(\lambda)$ | $\{0,1,..\}$ | $\lambda^ye^{-\lambda}/y!$ | $\lambda$ | $\lambda$ | Technical read counts |
| Neg. binomial$(\mu,\phi)$ | $\{0,1,..\}$ | (4.8) | $\mu$ | $\mu+\phi\mu^2$ | RNA-seq, scRNA-seq, cluster counts |
| Multinomial$(N,\mathbf{p})$ | simplex counts | $N!\prod p_d^{y_d}/y_d!$ | $Np_d$ | $Np_d(1-p_d)$ | Microbiome, cell-type counts |
| Dirichlet-multinomial | simplex counts | - | $Np_d$ | overdispersed | Microbiome with variability |
| Normal$(\mu,\sigma^2)$ | $\mathbb{R}$ | $\frac{1}{\sigma\sqrt{2\pi}}e^{-(y-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | Log intensities, M-values |
| Log-normal | $(0,\infty)$ | - | $e^{\mu+\sigma^2/2}$ | $(e^{\sigma^2}-1)e^{2\mu+\sigma^2}$ | Raw intensities, concentrations |
| Beta$(a,b)$ | $(0,1)$ | $\frac{y^{a-1}(1-y)^{b-1}}{B(a,b)}$ | $\frac{a}{a+b}$ | $\frac{ab}{(a+b)^2(a+b+1)}$ | Methylation beta values |
| Gamma$(k,\vartheta)$ | $(0,\infty)$ | - | $k\vartheta$ | $k\vartheta^2$ | Positive continuous, NB mixing |
| Exponential$(\lambda)$ | $(0,\infty)$ | $\lambda e^{-\lambda y}$ | $1/\lambda$ | $1/\lambda^2$ | Constant-hazard survival |
| Weibull$(k,\lambda)$ | $(0,\infty)$ | - | $\lambda\Gamma(1+1/k)$ | - | Parametric survival |
| $\chi^2_\nu$ | $(0,\infty)$ | - | $\nu$ | $2\nu$ | LRT, variance tests |
| $t_\nu$ | $\mathbb{R}$ | - | $0\ (\nu>1)$ | $\frac{\nu}{\nu-2}\ (\nu>2)$ | Small-sample inference |
| $F_{d_1,d_2}$ | $(0,\infty)$ | - | $\frac{d_2}{d_2-2}$ | - | ANOVA, QL tests |


# Appendix B: Identities worth memorising

$$\operatorname{Var}(X) = \mathbb{E}[X^2] - \big(\mathbb{E}[X]\big)^2 \tag{B.1}$$

$$\operatorname{Var}(Y) = \mathbb{E}\big[\operatorname{Var}(Y\mid Z)\big] + \operatorname{Var}\big(\mathbb{E}[Y\mid Z]\big) \tag{B.2}$$

$$\operatorname{Var}(\bar{Y}) = \frac{\sigma_b^2}{n} + \frac{\sigma_e^2}{nm}
\quad\text{(the pseudoreplication equation)} \tag{B.3}$$

$$\mathrm{DE} = 1+(m-1)\rho \quad\text{- clustering, and inter-gene correlation, inflate variance identically} \tag{B.4}$$

$$\hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y},
\qquad \operatorname{Var}(\hat{\boldsymbol\beta}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1} \tag{B.5}$$

$$t^2 = F \ \text{(2 groups)}, \qquad
\text{t-test} = \text{ANOVA} = \text{linear model} \tag{B.6}$$

$$\mathrm{AUC} = \Pr(X_{\text{case}} > X_{\text{control}}) = \frac{U}{n_1n_2}
\quad\text{(Mann-Whitney = ROC)} \tag{B.7}$$

$$\mathbb{E}[\text{posterior mean}] = \lambda\,(\text{data}) + (1-\lambda)(\text{prior}),
\quad \lambda = \frac{\tau^2}{\tau^2+\sigma^2}
\quad\text{(shrinkage = BLUP = empirical Bayes)} \tag{B.8}$$

$$\text{MSE} = \text{bias}^2 + \text{variance}
\quad\text{(why shrinkage helps)} \tag{B.9}$$

$$S(t) = e^{-H(t)}, \qquad h(t) = -\frac{d}{dt}\log S(t) \tag{B.10}$$

$$\text{Bonferroni} \prec \text{Holm} \prec \text{BH}
\quad\text{(increasing power, decreasing strictness of guarantee)} \tag{B.11}$$

$$n \approx \frac{16}{d^2} \ \text{per group}
\quad(\alpha=0.05,\ \text{power}=0.8) \tag{B.12}$$


# Appendix C: Symbol-to-code dictionary

| Maths | R | Python |
|---|---|---|
| $\bar{y}$, $s^2$ | `mean(y)`, `var(y)` | `y.mean()`, `y.var(ddof=1)` |
| $\Phi(z)$, $\Phi^{-1}(q)$ | `pnorm(z)`, `qnorm(q)` | `scipy.stats.norm.cdf/ppf` |
| $t_{1-\alpha/2,\nu}$ | `qt(1-a/2, nu)` | `scipy.stats.t.ppf(1-a/2, nu)` |
| Eq. (7.3) Welch | `t.test(x, y)` | `scipy.stats.ttest_ind(x, y, equal_var=False)` |
| Eq. (7.6) Mann-Whitney | `wilcox.test(x, y)` | `scipy.stats.mannwhitneyu(x, y)` |
| Eq. (8.5) BH | `p.adjust(p, "BH")` | `statsmodels.stats.multitest.multipletests(p,method="fdr_bh")` |
| Eq. (8.4) Holm | `p.adjust(p, "holm")` | `...method="holm"` |
| Eq. (11.2) OLS | `lm(y ~ x)` | `statsmodels.formula.api.ols("y ~ x", d).fit()` |
| Eq. (11.6) partial F | `anova(fit0, fit1)` | `fit1.compare_f_test(fit0)` |
| Eq. (11.11) Cook's D | `cooks.distance(fit)` | `fit.get_influence().cooks_distance` |
| Eq. (5.14) robust SE | `sandwich::vcovHC(fit,"HC3")` | `fit.get_robustcov_results("HC3")` |
| Eq. (13.6) logistic | `glm(y~x, family=binomial)` | `smf.logit("y ~ x", d).fit()` |
| Eq. (13.8) Poisson + offset | `glm(y~x+offset(log(t)), family=poisson)` | `smf.glm(..., offset=np.log(t), family=Poisson())` |
| NB GLM | `MASS::glm.nb(y ~ x)` | `smf.glm(..., family=NegativeBinomial(alpha=phi))` |
| Eq. (14.1) LMM | `nlme::lme(y~x, random=~1\|id)` / `lme4::lmer(y~x+(1\|id))` | `smf.mixedlm("y ~ x", d, groups=d.id).fit()` |
| Eq. (12.3) contrasts | `emmeans::contrast(...)` | `fit.t_test(contrast_matrix)` |
| Eq. (17.2) PCA | `prcomp(X, center=TRUE)` | `sklearn.decomposition.PCA()` |
| Eq. (16.6) SVD | `svd(X)` | `numpy.linalg.svd(X)` |
| Eq. (18.10) k-means | `kmeans(X, k, nstart=25)` | `sklearn.cluster.KMeans(n_init=25)` |
| Eq. (18.11) silhouette | `cluster::silhouette(cl, d)` | `sklearn.metrics.silhouette_score` |
| Eq. (28.6) Kaplan-Meier | `survival::survfit(Surv(t,e)~g)` | `lifelines.KaplanMeierFitter` |
| Eq. (28.10) Cox | `survival::coxph(Surv(t,e)~x)` | `lifelines.CoxPHFitter` |
| Eq. (28.11) PH check | `cox.zph(fit)` | `fit.check_assumptions(df)` |
| Eq. (31.5) ridge / lasso | `glmnet::glmnet(x,y,alpha=0/1)` | `sklearn.linear_model.Ridge/Lasso` |
| Eq. (31.8) AUC | `pROC::roc(y, s)$auc` | `sklearn.metrics.roc_auc_score` |
| Eq. (29.1) hypergeometric | `phyper(k-1,K,N-K,n,lower=FALSE)` | `scipy.stats.hypergeom.sf(k-1,N,K,n)` |
| Eq. (4.12) permutation | manual / `coin::` | manual with `numpy.random.Generator` |
| Bootstrap (5.12) | `boot::boot`, `boot.ci` | `scipy.stats.bootstrap` |
| Seeding | `set.seed(1); RNGkind("L'Ecuyer-CMRG")` | `rng = np.random.default_rng(1)` |


# References

[^se1]: Bioconductor. *SummarizedExperiment: coordinating experimental assays, samples, and regions of interest.* https://bioconductor.org/packages/SummarizedExperiment
[^se2]: Bioconductor. *MultiAssayExperiment quick-start guide.* https://bioconductor.org/packages/MultiAssayExperiment
[^asa1]: Wasserstein RL, Lazar NA (2016). The ASA statement on p-values: context, process, and purpose. *The American Statistician* 70(2):129-133.
[^asa2]: Wasserstein RL, Schirm AL, Lazar NA (2019). Moving to a world beyond "p < 0.05". *The American Statistician* 73(sup1):1-19.
[^asa3]: Benjamini Y, et al. (2021). ASA President's Task Force statement on statistical significance and replicability. *Annals of Applied Statistics*.
[^asa4]: Greenland S, et al. (2016). Statistical tests, p-values, confidence intervals, and power: a guide to misinterpretations. *European Journal of Epidemiology* 31:337-350.
[^sc1]: Crowell HL, et al. (2020). muscat detects subpopulation-specific state transitions from multi-sample multi-condition scRNA-seq data. *Nature Communications* 11:6077.
[^sc2]: Squair JW, et al. (2021). Confronting false discoveries in single-cell differential expression. *Nature Communications* 12:5692.
[^sc3]: Bioconductor. *Differential state analysis with muscat.* https://bioconductor.org/packages/muscat
[^sc4]: Junttila S, Smolander J, Elo LL (2022). Benchmarking methods for detecting differential states between conditions from multi-subject single-cell RNA-seq data. *Briefings in Bioinformatics* 23(5):bbac286.
[^rna1]: Love MI, Huber W, Anders S (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biology* 15:550.
[^rna2]: Law CW, Chen Y, Shi W, Smyth GK (2014). voom: precision weights unlock linear model analysis tools for RNA-seq read counts. *Genome Biology* 15:R29.
[^rna3]: Chen Y, Chen L, Lun ATL, Baldoni PL, Smyth GK (2025). edgeR v4: powerful differential analysis of sequencing data with expanded functionality. *Nucleic Acids Research* 53:gkaf018.
[^pca1]: Lever J, Krzywinski M, Altman N (2017). Principal component analysis. *Nature Methods* 14:641-642.
[^pca2]: Ma S, Dai Y (2011). Principal component analysis based methods in bioinformatics studies. *Briefings in Bioinformatics* 12(6):714-722.
[^batch1]: Zhou X, et al. (2024). Assessing and mitigating batch effects in large-scale omics studies. *Genome Biology* 25:254.
[^md1]: CRAN Task View: Missing Data. https://cran.r-project.org/view=MissingData
[^cyt1]: Weber LM, et al. (2019). diffcyt: differential discovery in high-dimensional cytometry via high-resolution clustering. *Communications Biology* 2:183.
[^cyt2]: Nowicka M, et al. (2019). CyTOF workflow: differential discovery in high-throughput high-dimensional cytometry datasets. *F1000Research* 6:748.
[^meth1]: Phipson B, Maksimovic J, Oshlack A (2016). missMethyl: an R package for analyzing data from Illumina's HumanMethylation450 platform. *Bioinformatics* 32(2):286-288.
[^meth2]: Maksimovic J, Phipson B, Oshlack A (2017). A cross-package Bioconductor workflow for analysing methylation array data. *F1000Research* 5:1281.
[^mm1]: Du P, Zhang X, Huang CC, Jafari N, Kibbe WA, Hou L, Lin SM (2010). Comparison of Beta-value and M-value methods for quantifying methylation levels by microarray analysis. *BMC Bioinformatics* 11:587.
[^gwas1]: Uffelmann E, et al. (2021). Genome-wide association studies. *Nature Reviews Methods Primers* 1:59.
[^prot1]: Zhang X, et al. (2018). Proteome-wide identification of ubiquitin interactions using UbIA-MS (DEP). *Nature Protocols* 13:530-550.
[^prot2]: Choi M, et al. (2014). MSstats: an R package for statistical analysis of quantitative mass spectrometry-based proteomic experiments. *Bioinformatics* 30(17):2524-2526.
[^mb1]: Lin H, Peddada SD (2020). Analysis of compositions of microbiomes with bias correction. *Nature Communications* 11:3514.
[^mb2]: Lin H, Peddada SD (2024). Multigroup analysis of compositions of microbiomes with covariate adjustments and repeated measures (ANCOM-BC2). *Nature Methods* 21:83-91.
[^spa1]: Bioconductor. *Orchestrating Spatial Transcriptomics Analysis (OSTA)*, differential colocalization. https://bioconductor.org/books/release/OSTA/
[^surv1]: Therneau TM, Grambsch PM (2000). *Modeling Survival Data: Extending the Cox Model.* Springer.
[^gsea1]: Wu D, Smyth GK (2012). Camera: a competitive gene set test accounting for inter-gene correlation. *Nucleic Acids Research* 40(17):e133.
[^gsea2]: Subramanian A, et al. (2005). Gene set enrichment analysis: a knowledge-based approach for interpreting genome-wide expression profiles. *PNAS* 102(43):15545-15550.

### Further standard references

- Efron B, Hastie T (2016). *Computer Age Statistical Inference.* Cambridge.
- Gelman A, et al. (2013). *Bayesian Data Analysis*, 3rd ed. CRC.
- Hastie T, Tibshirani R, Friedman J (2009). *The Elements of Statistical Learning*, 2nd ed. Springer.
- Hernán MA, Robins JM (2020). *Causal Inference: What If.* CRC.
- Harrell FE (2015). *Regression Modeling Strategies*, 2nd ed. Springer.
- Holmes S, Huber W (2018). *Modern Statistics for Modern Biology.* Cambridge.
- Amezquita RA, et al. (2020). Orchestrating single-cell analysis with Bioconductor. *Nature Methods* 17:137-145.
- van Buuren S (2018). *Flexible Imputation of Missing Data*, 2nd ed. CRC.
- Aitchison J (1986). *The Statistical Analysis of Compositional Data.* Chapman & Hall.
