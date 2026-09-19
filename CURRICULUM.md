# Statistics for Bioinformatics in R: A Decision-Centred Learning Curriculum

## Purpose and scope

This curriculum is designed for learning **statistical decision-making through R-based bioinformatics analyses**, without prescribing scripts. The goal is to learn how to move from a biological question and experimental design to an appropriate data representation, statistical model, diagnostic strategy, uncertainty statement, and reproducible conclusion.

The curriculum is current to September 2026. The current Bioconductor release is 3.23, designed for R 4.6, and includes 2,418 software packages, 437 experiment-data packages, 928 annotation packages, 28 workflows, and eight books. Core Bioconductor data structures remain central: `SummarizedExperiment` coordinates matrix-like assays with feature and sample metadata, while `MultiAssayExperiment` coordinates multiple assays measured on overlapping sets of biological units.[^1][^2][^3]

This syllabus is paired with a runnable implementation: `stats.md` for the
mathematics and `statsPy/` / `statsR/` for the code. See
[How this curriculum is implemented](#how-this-curriculum-is-implemented).

**New to statistics?** This document assumes you already know what a standard
error and a p-value are. If you do not, start with **Part 0 of `stats.md`**
and the six `foundations/` modules, which build everything from arithmetic
upwards and assume no mathematical background at all. Then return here.

The recommended sequence is:

1. Learn the unit of analysis and data-generating process.
2. Learn estimation, uncertainty, tests, and multiplicity.
3. Learn linear models as the unifying framework.
4. Extend models to counts, proportions, repeated measures, survival, and high-dimensional data.
5. Apply these ideas separately to each bioinformatics modality.
6. Finish with prediction, causal reasoning, multi-omics, sensitivity analysis, and reproducibility.

## Statistical decision framework

Every analysis should begin with the following questions. These questions matter more than the choice of package.

| Decision | Questions to answer | Consequence of a wrong decision |
|---|---|---|
| Biological unit | Is the independent unit a patient, donor, animal, tissue, library, well, cell, spot, read, or feature? | Treating cells, reads, or technical replicates as independent can create pseudoreplication and underestimated uncertainty. |
| Outcome | Is it continuous, binary, count, proportion, ordinal, compositional, censored, time-to-event, or multivariate? | The wrong outcome distribution produces inappropriate standard errors, confidence intervals, and predictions. |
| Experimental structure | Is the design independent, paired, blocked, longitudinal, nested, crossed, factorial, or multi-centre? | Dependence may be ignored, and treatment effects may be confounded with donor or batch effects. |
| Scientific estimand | Is the target a mean difference, fold change, odds ratio, rate ratio, hazard ratio, correlation, variance component, abundance difference, or prediction error? | A valid model may still answer the wrong scientific question. |
| Scale | Should analysis use raw counts, log counts, M-values, beta values, arcsinh intensities, proportions, ranks, or transformed compositions? | Effect estimates and assumptions change with scale. |
| Covariates | Which variables are confounders, precision variables, mediators, colliders, batches, or nuisance factors? | Unnecessary or inappropriate adjustment can bias rather than improve an estimate. |
| Multiplicity | What constitutes the family of hypotheses: all genes, all cell types, all contrasts, all time points, or all pathways? | Error control may not match the actual scientific claim. |
| Validation | Is the goal explanation, discovery, estimation, classification, prognosis, or causal inference? | Validation and performance criteria differ substantially across goals. |

## Part I: Foundations

### Topic 1: Statistical thinking for biological experiments

**Description:** Learn to translate a biological question into a population, sampling process, experimental unit, outcome, exposure, estimand, and analysis plan. This module prevents the common mistake of choosing a test based only on the appearance of a dataset.

**Subtopics**

- Population, sample, observational unit, experimental unit, and unit of inference.
- Biological versus technical replication.
- Independent, paired, blocked, repeated-measures, nested, crossed, and factorial designs.
- Randomization, blinding, allocation, balance, and counterbalancing.
- Confounding, selection bias, measurement error, batch effects, and pseudoreplication.
- Explanatory, predictive, descriptive, exploratory, and confirmatory analysis.
- Estimands: average treatment effect, conditional association, mean difference, fold change, odds ratio, rate ratio, and variance explained.
- Pre-specified versus data-driven hypotheses.
- Negative controls, positive controls, spike-ins, reference samples, and technical standards.

**Data types to use**

- A two-group cell-culture experiment with biological and technical replicates.
- A paired before-after intervention study.
- A mouse experiment blocked by litter or cage.
- A multi-centre cohort with centre, processing date, and phenotype metadata.
- Single-cell data with cells nested in donors.

**Decision-making goals**

- Identify the sample size that matters for inference.
- Distinguish a treatment effect from a batch or donor effect.
- Decide whether pairing or blocking must appear in the model.
- State the estimand before looking at p-values.

### Topic 2: Bioinformatics data structures and measurement scales

**Description:** Learn how biological measurement technologies create different statistical objects. A feature-by-sample matrix is not sufficient to determine the model; counts, intensities, proportions, and censored observations have different data-generating mechanisms.

**Subtopics**

- Wide matrices, tidy/long data, sparse matrices, delayed/on-disk matrices, and genomic ranges.
- Assay data, row metadata, column metadata, and provenance.
- `SummarizedExperiment`, `RangedSummarizedExperiment`, `SingleCellExperiment`, `TreeSummarizedExperiment`, and `MultiAssayExperiment` concepts.
- Raw versus normalized measurements.
- Integer counts, continuous intensities, ratios, proportions, probabilities, ranks, and categories.
- Feature-level annotation: genes, transcripts, peaks, CpGs, proteins, taxa, cells, and genomic intervals.
- Sample identifiers, donor identifiers, aliquots, repeated specimens, and assay-specific identifiers.
- Missingness, zeros, values below detection, structural zeros, and absent features.
- Longitudinal and survival representations.

`SummarizedExperiment` stores one or more matrix-like assays with rows commonly representing genomic features and columns representing samples; feature and sample annotations travel with the measurements. `MultiAssayExperiment` adds coordination across different assays and partially overlapping sample sets, including an explicit map from assay-specific samples to biological units.[^2][^3][^4][^5]

**Data types to use**

- Bulk RNA-seq gene counts.
- Microarray expression intensities.
- DNA methylation beta and M-values.
- Cytometry event-by-marker matrices and sample-level summaries.
- Proteomics intensity matrices with missing values.
- Microbiome taxon-count tables.
- Genotype matrices coded as allele dosage.
- Multi-omics cohorts with partially overlapping samples.

**Decision-making goals**

- Determine which object is the assay, which dimension contains samples, and which identifier defines independence.
- Decide which scale is suitable for visualization and which for formal inference.
- Detect accidental misalignment between assay columns and metadata rows.

### Topic 3: Exploratory data analysis and visualization

**Description:** Use graphical and numerical summaries to understand distributions, dependence, missingness, outliers, and technical structure before formal modelling. Exploration should diagnose data quality without quietly redefining the hypothesis after seeing results.

**Subtopics**

- Means, medians, quantiles, variance, median absolute deviation, skewness, and robust summaries.
- Histograms, empirical cumulative distributions, density plots, boxplots, violin plots, raincloud plots, and dot plots.
- Scatterplots, smoothers, hexbin plots, correlation heatmaps, and sample-distance heatmaps.
- Library-size, feature-detection, missingness, and quality-control distributions.
- Mean-variance relationships and heteroscedasticity.
- Outlier detection versus outlier deletion.
- Stratification by batch, group, sex, donor, centre, time, and processing variables.
- Graphical checks for pairing and repeated measures.
- Data leakage caused by exploring test data or outcome-aware preprocessing.

**Data types to use**

- Clinical covariates plus expression measurements.
- RNA-seq library metrics and count distributions.
- Proteomics intensities with left-censoring and missingness.
- Flow-cytometry marker distributions before and after transformation.

**Decision-making goals**

- Decide whether an unusual sample is erroneous, biologically extreme, or influential but valid.
- Choose transformations based on measurement behaviour rather than visual preference.
- Separate exploratory findings from confirmatory claims.

### Topic 4: Probability models and sampling distributions

**Description:** Learn the probability distributions underlying common bioinformatics models. Focus on why standard errors and likelihoods change across continuous measurements, counts, binary outcomes, proportions, and survival times.

**Subtopics**

- Random variables, parameters, estimators, statistics, and likelihood.
- Bernoulli, binomial, multinomial, Poisson, negative-binomial, normal, log-normal, beta, gamma, and zero-inflated distributions.
- Expectation, variance, covariance, conditional probability, and independence.
- Sampling distributions, law of large numbers, and central limit theorem.
- Overdispersion and underdispersion.
- Mixtures, latent variables, censoring, and truncation.
- Frequentist probability versus Bayesian probability.
- Monte Carlo reasoning and permutation null distributions.

**Data types to use**

- RNA-seq counts for Poisson versus negative-binomial thinking.
- Mutation presence/absence for Bernoulli models.
- Allele counts for binomial models.
- Protein intensities for normal or log-normal models.
- Cell-type counts across samples for multinomial or count models.

**Decision-making goals**

- Match the outcome to a plausible distribution.
- Recognize when variance exceeds the mean and a Poisson model is inadequate.
- Understand what assumptions produce a reported standard error.

## Part II: Inference

### Topic 5: Estimation, effect sizes, and confidence intervals

**Description:** Treat effect estimation as the primary scientific task and hypothesis tests as one component of uncertainty assessment. Current statistical guidance emphasizes that p-values do not measure effect magnitude or practical importance and should be interpreted alongside estimates and intervals.[^6][^7][^8]

**Subtopics**

- Point estimates, bias, consistency, efficiency, and robustness.
- Standard errors and confidence intervals.
- Absolute versus relative effect measures.
- Mean and median differences.
- Standardized mean differences and their limitations.
- Correlation coefficients and coefficients of determination.
- Odds ratios, risk ratios, risk differences, rate ratios, and hazard ratios.
- Log2 fold changes and shrinkage estimates.
- Minimum biologically meaningful effects and equivalence margins.
- Bootstrap percentile, basic, and bias-corrected intervals.
- Profile-likelihood and robust/sandwich intervals.
- Practical significance versus statistical compatibility.

**Data types to use**

- Continuous biomarker concentrations.
- Binary disease outcomes.
- RNA-seq log2 fold changes.
- Cell-population proportions.
- Time-to-event clinical outcomes.

**Decision-making goals**

- Choose an effect measure interpretable for the biological question.
- Report direction, magnitude, uncertainty, and biological relevance together.
- Distinguish evidence of absence from absence of evidence.

### Topic 6: Hypothesis tests and p-values

**Description:** Learn what a test statistic and p-value do-and do not-say. A p-value measures incompatibility between data and a specified model under the tested hypothesis; it is not the probability that the null hypothesis is true and does not encode effect importance.[^9][^6]

**Subtopics**

- Null and alternative hypotheses.
- Simple versus composite hypotheses.
- One-sided versus two-sided tests.
- Test statistics and reference distributions.
- Exact, asymptotic, permutation, and bootstrap tests.
- Type I error, Type II error, power, and false-positive risk.
- Statistical thresholds as decision rules rather than natural boundaries.
- Exact p-values versus inequalities and significance stars.
- Optional stopping, repeated analyses, and researcher degrees of freedom.
- Selective reporting, p-hacking, and HARKing.
- Sensitivity analyses and multiverse analyses.
- Equivalence and non-inferiority tests.

**Data types to use**

- A controlled two-group assay.
- A permutation-compatible randomized experiment.
- Gene-wise differential-expression results.
- A biomarker study with a pre-specified meaningful effect threshold.

**Decision-making goals**

- State precisely what probability a p-value represents.
- Avoid interpreting `p > 0.05` as proof of no effect.
- Decide whether the question concerns difference, superiority, equivalence, or non-inferiority.
- Interpret the result in the context of model assumptions and study design.

### Topic 7: t-tests, rank tests, and permutation tests

**Description:** Use simple group comparisons to learn the connection between design, estimand, assumptions, and regression. The t-test should become a special case of a linear model rather than a memorized endpoint.

**Subtopics**

- One-sample, independent two-sample, Welch, pooled-variance, and paired t-tests.
- Mean difference, standard error, degrees of freedom, and confidence interval.
- Normality of observations versus normality of residuals or the sampling distribution.
- Variance heterogeneity and why Welch is usually safer than an equal-variance test.
- Mann-Whitney/Wilcoxon rank-sum and Wilcoxon signed-rank tests.
- Why rank tests do not automatically test medians.
- Randomization and permutation tests under exchangeability.
- Robust location estimators and trimmed-mean tests.
- Effect sizes for independent and paired data.
- Multiple-group extensions through ANOVA and linear models.

**Data types to use**

- Normally distributed log-intensity measurements.
- Skewed cytokine concentrations.
- Paired pre/post expression or cytometry summaries.
- Small randomized experiments suitable for exact permutation.

**Decision-making goals**

- Choose paired or independent analysis from the design, not from a normality test.
- Decide whether the scientific estimand is a mean difference, distributional shift, or rank-based probability.
- Use Welch, transformation, robust methods, or permutation based on the inferential target and assumptions.

### Topic 8: Multiple testing and selective inference

**Description:** Learn to define the hypothesis family before choosing an adjustment. R provides Bonferroni, Holm, Hochberg, Hommel, Benjamini-Hochberg, and Benjamini-Yekutieli adjustments; BH and BY target false discovery rate, whereas family-wise procedures answer a stricter error-control question.[^10][^11]

**Subtopics**

- Per-comparison error, family-wise error rate, false discovery rate, and local false discovery rate.
- Bonferroni, Holm, Hochberg, Hommel, BH, and BY procedures.
- Adjusted p-values and q-values.
- Dependence among genes, CpGs, proteins, taxa, or pathways.
- Independent filtering and covariate-assisted weighting.
- Hierarchical testing across genes, transcripts, cell types, and pathways.
- Multiple contrasts and omnibus-before-pairwise testing.
- Stage-wise testing and selective inference.
- Replicability across studies.
- Effect-size thresholds combined with error control.

**Data types to use**

- Genome-wide gene-level p-values.
- CpG-level methylation tests.
- Multiple cell types by multiple contrasts.
- Pathway-level enrichment results.
- GWAS variant-level tests.

**Decision-making goals**

- Define exactly which discoveries the FDR statement covers.
- Distinguish correction across genes from correction across all genes, cell types, and contrasts.
- Avoid filtering features using outcome information unless the inferential method accounts for it.
- Prioritize effect size and uncertainty after multiplicity correction.

### Topic 9: Power, sample size, and design optimization

**Description:** Learn power as a property of a proposed design, model, effect size, variance, and decision rule-not as a retrospective explanation for a disappointing p-value.

**Subtopics**

- Alpha, power, effect size, sample size, and variance.
- Minimum detectable and minimum relevant effects.
- Balanced versus unbalanced allocation.
- Gains from paired and blocked designs.
- Biological replication versus deeper sequencing or more cells.
- Power under overdispersed counts.
- Multiple-testing burden and discovery power.
- Clustered and longitudinal designs.
- Simulation-based power for complex models.
- Pilot-data uncertainty and optimistic variance estimates.
- Sensitivity curves across plausible parameters.

**Data types to use**

- Two-group continuous outcomes.
- Bulk RNA-seq with different sample sizes and dispersions.
- Single-cell studies varying donors and cells per donor.
- Longitudinal cohorts varying visits and attrition.

**Decision-making goals**

- Decide whether to add donors, cells, sequencing depth, technical replicates, or time points.
- Identify the parameter assumptions driving a power result.
- Separate precision goals from binary significance goals.

## Part III: Association and models

### Topic 10: Correlation and dependence

**Description:** Learn correlation as a scale-specific summary of association, not evidence of causation or agreement. High-dimensional bioinformatics adds dependence among features, repeated observations, compositional constraints, and strong sensitivity to preprocessing.

**Subtopics**

- Covariance and correlation matrices.
- Pearson, Spearman, and Kendall correlation.
- Linear versus monotonic association.
- Partial correlation and conditional association.
- Robust and biweight correlations.
- Distance correlation and nonlinear dependence.
- Correlation uncertainty, testing, and multiplicity.
- Outliers, restricted ranges, mixtures, and Simpson's paradox.
- Repeated-measures correlation and donor dependence.
- Agreement versus correlation: Bland-Altman thinking and concordance measures.
- Gene co-expression networks and correlation shrinkage.
- Spurious correlation in compositional data.

**Data types to use**

- Paired expression and protein abundance measurements.
- Technical replicates for agreement analysis.
- Longitudinal biomarkers within individuals.
- Gene-expression matrices for co-expression.
- Microbiome relative abundances to demonstrate compositional artefacts.

**Decision-making goals**

- Choose Pearson for an approximately linear relationship or a rank method for monotonic association and robustness.
- Account for repeated observations from the same donor.
- Avoid calling high correlation proof of assay agreement, mechanism, or causality.
- Decide whether adjustment for covariates answers the intended question.

### Topic 11: Linear regression as the core framework

**Description:** Treat linear regression as the common language connecting t-tests, ANOVA, covariate adjustment, factorial experiments, trends, interactions, and many omics methods.

**Subtopics**

- Response, predictors, coefficients, residuals, fitted values, and design matrices.
- Intercepts, reference levels, dummy variables, and contrasts.
- Simple and multiple regression.
- Continuous and categorical predictors.
- Confounder adjustment and precision covariates.
- Interactions and effect modification.
- Polynomial terms, splines, and nonlinear trends.
- Nested-model comparison and partial F-tests.
- Prediction intervals versus confidence intervals.
- Centering, scaling, and interpretable parameterization.
- Collinearity, rank deficiency, aliasing, and variance inflation.
- Residual normality, linearity, independence, homoscedasticity, and influential observations.
- Robust standard errors and robust regression.

**Data types to use**

- Continuous biomarker outcome with age, sex, treatment, and batch covariates.
- Microarray or transformed proteomics intensities.
- Time-course measurements using splines and treatment-by-time interactions.
- A factorial perturbation experiment.

**Decision-making goals**

- Interpret coefficients conditionally and on the correct scale.
- Distinguish confounding from mediation and effect modification.
- Diagnose a design matrix before fitting thousands of feature-wise models.
- Select contrasts that directly represent the biological hypothesis.

### Topic 12: ANOVA, factorial designs, and contrasts

**Description:** Learn ANOVA as linear modelling of categorical predictors rather than as a separate family of procedures. Emphasize contrasts, interactions, imbalance, and scientifically targeted comparisons.

**Subtopics**

- One-way and multi-way ANOVA.
- Omnibus tests versus targeted contrasts.
- Treatment, sum-to-zero, Helmert, and custom contrast coding.
- Main effects in the presence of interactions.
- Balanced and unbalanced designs.
- Type I, II, and III sums of squares and why model questions matter more than labels.
- Repeated-measures ANOVA versus mixed models.
- Post-hoc comparisons and multiplicity.
- Trend contrasts for ordered doses.
- Factorial perturbation and genotype-by-environment designs.

**Data types to use**

- Three or more treatment groups.
- Dose-response experiments.
- Genotype-by-treatment experiments.
- Tissue-by-condition expression studies.

**Decision-making goals**

- Decide whether an omnibus hypothesis is useful or a pre-specified contrast is preferable.
- Interpret main effects only after checking interactions.
- Avoid automatic all-pairs testing when only a few biological comparisons matter.

### Topic 13: Generalized linear models

**Description:** Extend regression to outcomes whose variance and range make Gaussian models unsuitable. This module provides the conceptual basis for logistic regression, count models, differential expression, differential abundance, and rate modelling.

**Subtopics**

- Exponential-family distributions, link functions, linear predictors, and likelihood.
- Logistic regression for binary outcomes.
- Binomial regression for proportions with known denominators.
- Poisson regression for event counts and rates.
- Negative-binomial regression for overdispersed counts.
- Offsets for exposure time, library size, or sampling effort.
- Multinomial and ordinal regression.
- Zero-inflated and hurdle models.
- Deviance, likelihood-ratio, Wald, and score tests.
- Dispersion, residuals, calibration, separation, and overfitting.
- Effect interpretation on link and response scales.

**Data types to use**

- Disease status or response/no-response outcomes.
- Mutation counts per genomic region with callable-bases offsets.
- RNA-seq and CRISPR-screen counts.
- Cell counts or microbial counts per sample.
- Ordinal pathology scores.

**Decision-making goals**

- Match the distribution and link to the outcome and estimand.
- Distinguish counts from rates and include appropriate offsets.
- Interpret odds ratios, rate ratios, and predicted probabilities correctly.
- Detect overdispersion and choose a negative-binomial or quasi-likelihood approach when needed.

### Topic 14: Mixed, multilevel, and repeated-measures models

**Description:** Model dependence created by donors, batches, centres, families, plates, repeated visits, and nested cells. Current R resources cover mixed models for Gaussian, count, survival, ordinal, zero-inflated, hurdle, and censored outcomes.[^12][^13]

**Subtopics**

- Fixed effects, random effects, variance components, and partial pooling.
- Random intercepts and random slopes.
- Nested and crossed random effects.
- Repeated measurements and covariance structures.
- Subject-specific versus population-average effects.
- Linear and generalized linear mixed models.
- Marginal models and generalized estimating equations.
- Intraclass correlation.
- Singular fits, boundary estimates, and convergence.
- Degrees-of-freedom approximations and likelihood-ratio tests.
- Cluster-robust standard errors.
- Feature-wise mixed models and computational trade-offs.

**Data types to use**

- Repeated biomarker measurements by patient.
- Cells nested within samples and donors.
- Multi-centre omics with centre and batch structure.
- Family-based genetic data.
- Multiple tissues from the same individual.

**Decision-making goals**

- Identify every level of clustering before fitting the model.
- Choose between aggregation, random effects, fixed blocking effects, and marginal models.
- Avoid interpreting thousands of cells as thousands of biological replicates.

### Topic 15: Missing data, censoring, and measurement limits

**Description:** Learn that deleting incomplete cases or automatically imputing a matrix can change the estimand and introduce bias. Current R missing-data guidance distinguishes exploration, likelihood-based methods, single and multiple imputation, weighting, data-specific methods, and application-specific approaches.[^14][^15]

**Subtopics**

- Missing completely at random, missing at random, and missing not at random.
- Structural missingness versus random missingness.
- Complete-case analysis and its assumptions.
- Single imputation versus multiple imputation.
- Predictive mean matching and chained equations.
- Likelihood-based handling of incomplete outcomes.
- Inverse-probability weighting.
- Sensitivity analysis for missing-not-at-random mechanisms.
- Left-censoring and limits of detection.
- Dropout in longitudinal studies.
- Missing proteomic intensities and the danger of assuming all missing values are low abundance.
- Missing assay blocks in multi-omics studies.

**Data types to use**

- Clinical covariates with sporadic missingness.
- Longitudinal cohorts with dropout.
- Proteomics/metabolomics intensities below detection.
- Multi-omics data with assays unavailable for some patients.

**Decision-making goals**

- Explain a plausible missingness mechanism for each variable.
- Decide whether imputation belongs inside each resampling fold for prediction.
- Separate missing-at-random assumptions from left-censoring assumptions.
- Perform sensitivity analyses rather than treating imputed values as observed truth.

## Part IV: Multivariate statistics

### Topic 16: Matrix algebra for high-dimensional biology

**Description:** Build the linear-algebra foundation needed for PCA, regression, distances, batch correction, factor models, and multi-omics integration.

**Subtopics**

- Vectors, matrices, transposition, inner products, norms, and projections.
- Rank, inverse, generalized inverse, and condition number.
- Covariance and correlation matrices.
- Eigenvalues and eigenvectors.
- Singular value decomposition.
- Orthogonality and basis changes.
- Centering and scaling as matrix operations.
- Sparse matrices and computational implications.
- The high-dimensional setting where features greatly exceed samples.

**Data types to use**

- Gene-by-sample expression matrices.
- Cell-by-gene sparse matrices.
- Patient-by-biomarker matrices.
- Genotype dosage matrices.

**Decision-making goals**

- Recognize rank deficiency and collinearity.
- Understand which dimension is being decomposed.
- Predict how centering and scaling change a multivariate analysis.

### Topic 17: Principal component analysis

**Description:** Learn PCA as unsupervised variance decomposition, not a classifier or formal batch-correction method. PCA creates orthogonal linear combinations that summarize variation in high-dimensional measurements, but the largest variation need not be the biology of interest.[^16][^17]

**Subtopics**

- PCA through covariance eigen-decomposition and singular value decomposition.
- Scores, loadings, eigenvalues, and proportion of variance explained.
- Centering, unit-variance scaling, and feature weighting.
- Covariance PCA versus correlation PCA.
- Scree plots, biplots, loading plots, and contribution measures.
- Sign indeterminacy and rotational interpretation.
- Sample PCA versus feature PCA.
- PCA after variance-stabilizing or log transformations.
- Association of PCs with batch, condition, library size, sex, and other covariates.
- Outlier and influence assessment.
- Selecting components for visualization versus downstream modelling.
- Sparse, robust, probabilistic, and supervised alternatives.
- Why testing features selected from the same PCA can create circularity.

**Data types to use**

- Variance-stabilized bulk RNA-seq expression.
- Log-transformed proteomics or metabolomics intensities.
- M-values from methylation arrays.
- Arcsinh-transformed cytometry summaries.
- Standardized continuous clinical biomarkers.

**Decision-making goals**

- Decide whether to scale features based on units and variance structure.
- Interpret a PC through loadings and metadata associations, not through plot appearance alone.
- Avoid presenting separation as proof of a biological effect.
- Distinguish PCA used for quality control from PCs used as covariates.

### Topic 18: Distances, clustering, and embeddings

**Description:** Learn how distance definitions, transformations, feature selection, and algorithm settings determine apparent biological groups. Clustering is exploratory unless stability and external evidence support the inferred structure.

**Subtopics**

- Euclidean, Manhattan, correlation, cosine, Jaccard, Bray-Curtis, and Aitchison distances.
- Hierarchical clustering and linkage criteria.
- k-means, partitioning around medoids, graph clustering, and mixture models.
- Cluster-number selection and stability.
- Silhouette scores and internal versus external validation.
- t-SNE and UMAP as neighbourhood visualizations.
- Stochasticity, parameter sensitivity, and seed dependence.
- Consensus clustering and bootstrap stability.
- Double dipping when clusters are defined and tested on the same features.
- Batch-driven clusters and sample imbalance.

**Data types to use**

- Bulk expression profiles.
- Single-cell expression or cytometry data.
- Microbiome compositions using appropriate distances.
- Binary mutation profiles.

**Decision-making goals**

- Choose a distance compatible with the measurement scale.
- Treat t-SNE/UMAP proximity and cluster boundaries as exploratory.
- Evaluate stability across preprocessing choices, seeds, and resamples.
- Avoid interpreting technical clusters as cell types or disease subtypes.

### Topic 19: Batch effects and unwanted variation

**Description:** Learn to prevent, diagnose, model, and-in carefully justified settings-correct technical variation. Batch effects can create false patterns when ignored, while overcorrection can erase biology; recent reviews recommend design-stage controls, diagnostics before and after correction, and explicit assessment of biological preservation.[^18]

**Subtopics**

- Sources of batch effects across laboratories, machines, operators, plates, runs, and pipelines.
- Randomization and balanced allocation across batches.
- Reference samples, pooled controls, and bridging samples.
- Confounding between batch and phenotype.
- PCA, variance decomposition, quality metrics, and control-feature diagnostics.
- Including batch in the design matrix.
- Location-scale correction, factor models, neighbourhood methods, and latent-variable approaches.
- Surrogate variables and unwanted-factor estimation.
- Correction for visualization versus modelling raw/appropriately transformed data with covariates.
- Overcorrection, biological signal removal, and unverifiable correction under complete confounding.
- Pre/post-correction evaluation and sensitivity analysis.

**Data types to use**

- Multi-batch RNA-seq.
- Multi-centre proteomics.
- Cytometry acquired over several days.
- Integrated single-cell datasets.
- Multi-omics reference-material studies.

**Decision-making goals**

- Prefer prevention and balanced design over post-hoc correction.
- Determine whether batch is observed, latent, crossed with biology, or completely confounded.
- Decide whether correction is for visualization, integration, prediction, or inference.
- Verify that adjustment reduces technical variation without removing protected biological effects.

## Part V: Modality-specific statistics

### Topic 20: Bulk RNA-seq differential expression

**Description:** Learn count-based inference through gene filtering, library normalization, design matrices, dispersion estimation, contrasts, effect-size shrinkage, multiplicity, and diagnostics. Current Bioconductor workflows present DESeq2, edgeR, and limma-voom as principal alternatives; DESeq2 expects raw integer counts and supports designs such as batch plus condition.[^19][^20][^21]

**Subtopics**

- Read and fragment counts, transcript quantification, and gene-level summarization.
- Raw counts versus TPM/FPKM and why normalized abundance is not a substitute for count-model input.
- Library-size and composition normalization.
- Low-expression filtering independent of the tested outcome.
- Negative-binomial mean-variance modelling.
- Dispersion estimation and empirical Bayes shrinkage.
- Quasi-likelihood inference.
- voom mean-variance weights and linear modelling.
- Design matrices for batch, pairing, interactions, and continuous covariates.
- Coefficient tests, contrasts, and omnibus tests.
- Log2 fold-change estimation and shrinkage.
- MA plots, dispersion plots, sample distances, and residual diagnostics.
- Transcript-level uncertainty and differential transcript usage.

The current edgeR guidance recommends filtering lowly expressed features, normalizing effective library sizes, and using quasi-likelihood fitting and testing. DESeq2's current vignette recommends importing transcript quantifications through `tximport` for gene-level analysis and allows explicit adjustment for batch in the design.[^21][^22]

**Data types to use**

- Gene-level raw count matrices with sample metadata.
- Transcript-level estimates with inferential replicates.
- Paired treatment/control RNA-seq.
- Factorial or time-course RNA-seq.

**Decision-making goals**

- Choose between negative-binomial and voom-based frameworks based on design, data, and inferential needs.
- Include known nuisance factors without correcting the count matrix into an uninterpretable form.
- Interpret shrunken fold changes separately from hypothesis-test statistics.
- Validate that the contrast corresponds to the intended biological comparison.

### Topic 21: Single-cell RNA-seq: replicated inference

**Description:** Separate cell-level exploration from sample-level inference. For multi-sample condition comparisons, evidence consistently favours methods that preserve the biological replicate-typically pseudobulk aggregation or appropriate mixed models-over tests that incorrectly treat cells from one donor as independent.[^22][^23][^24][^25]

**Subtopics**

- Cells, libraries, samples, donors, conditions, and nested hierarchy.
- Quality control and filtering without condition-dependent bias.
- Normalization for visualization versus count-based testing.
- Cell-type annotation uncertainty.
- Differential expression versus differential state versus differential abundance.
- Pseudobulk aggregation by cell type and biological sample.
- Sum aggregation versus means of normalized expression.
- Negative-binomial or voom analysis of pseudobulks.
- Mixed models for cell-level or pseudobulk outcomes.
- Donor blocking, paired designs, repeated measures, and incomplete cell-type/sample combinations.
- Minimum cell counts and low-information pseudobulks.
- Multiple testing across genes, clusters, and contrasts.
- Compositional changes in cell populations.
- Integration methods versus inferential models.

`muscat` supports differential-state analysis using either cell-level mixed models or pseudobulk data, with pseudobulk measurements aggregated at the cluster-by-sample level and analysed by bulk RNA-seq frameworks. Recent benchmarking continues to find that naive cell-level analyses can inflate false positives, while pseudobulk and subject-aware mixed models provide better error control.[^26][^23][^27][^24][^25]

**Data types to use**

- Multi-donor scRNA-seq count matrices.
- Cell metadata containing donor, condition, batch, and cell type.
- Paired pre/post single-cell studies.
- Nested multi-centre or repeated-visit single-cell studies.

**Decision-making goals**

- Define donors-not cells-as replicates for condition-level claims.
- Decide between pseudobulk and mixed models based on hierarchy, sample size, and computational cost.
- Distinguish changes within a cell type from changes in cell-type abundance.
- Perform integration for representation cautiously and preserve a valid sample-level inferential path.

### Topic 22: Flow, mass, and imaging cytometry

**Description:** Learn statistical analysis of event-level marker measurements while preserving sample-level replication. Current `diffcyt` methods combine high-resolution clustering with empirical-Bayes moderated tests; differential abundance commonly uses edgeR-style count models and differential state uses limma-style models.[^28][^29][^30]

**Subtopics**

- FCS data, events, samples, markers, panels, and batch metadata.
- Compensation, spillover, transformation, bead normalization, and quality control.
- Arcsinh transformations and modality-specific cofactors.
- Manual gating versus clustering.
- Marker roles: lineage/type versus state/function markers.
- Sample-level cell counts, frequencies, and median marker expression.
- Differential abundance of clusters or populations.
- Differential state within populations.
- Design and contrast matrices for paired, blocked, or factorial studies.
- Rare populations and minimum-event thresholds.
- Batch and acquisition-day effects.
- Multiple testing across clusters, markers, and contrasts.
- Cluster stability and annotation uncertainty.

The current `diffcyt` workflow accepts FCS-derived objects or preprocessed CATALYST objects, constructs design and contrast matrices, and calculates sample-by-cluster counts and marker medians before differential testing.[^29]

**Data types to use**

- Conventional flow cytometry event-by-marker data.
- CyTOF data with sample and marker annotations.
- Imaging mass-cytometry cell-level measurements.
- Sample-level manually gated frequencies.

**Decision-making goals**

- Keep the specimen or donor as the inferential unit.
- Choose count models for abundance and continuous-expression models for state.
- Decide whether a finding is a population-size change, marker-state change, or both.
- Account for panel, batch, acquisition day, and pairing in the design.

### Topic 23: DNA methylation and epigenomics

**Description:** Learn the distinct roles of beta values, M-values, probe design, cell composition, genomic annotation, and region-level inference. Current methylation workflows cover quality control, filtering, normalization, probe-wise differential methylation, regions, differential variability, gene-set analysis, and cell-composition estimation.[^31][^32]

**Subtopics**

- IDAT intensities, detection p-values, beta values, M-values, and probe annotations.
- Background correction, dye bias, and normalization.
- Cross-reactive probes, SNP-affected probes, sex chromosomes, replicate probes, and failed measurements.
- Beta values for interpretation versus M-values for modelling.
- Differentially methylated positions and regions.
- Differential variability.
- Cell-type composition estimation and adjustment.
- Age, smoking, sex, ancestry, and technical confounding.
- Batch effects and unwanted variation.
- Probe-wise linear models and empirical Bayes moderation.
- Multiple testing and spatially correlated CpGs.
- Enrichment bias caused by unequal probe representation per gene.
- Epigenetic clocks: training, calibration, transportability, and leakage.

The `missMethyl` workflow uses linear models and empirical Bayes methods for differential methylation and supports differential variability; its workflow recommends M-values for formal probe-wise modelling after appropriate normalization and filtering.[^33][^31]

**Data types to use**

- Illumina methylation-array IDAT data.
- CpG beta and M-value matrices.
- Bisulfite-sequencing methylated and total read counts.
- Epigenetic-age scores with chronological age and phenotype metadata.

**Decision-making goals**

- Choose a modelling scale separately from a presentation scale.
- Decide whether cell composition is a confounder, mediator, or biological outcome.
- Separate site-level and region-level hypotheses.
- Prevent leakage when evaluating methylation-based predictors.

### Topic 24: Genotypes, GWAS, and statistical genetics

**Description:** Learn association testing under population structure, relatedness, linkage disequilibrium, phenotype heterogeneity, and extreme multiplicity. Current Bioconductor tools support large GWAS data storage, quality control, frequentist analysis, and Bayesian variable-selection approaches.[^34][^35]

**Subtopics**

- Genotype calls, allele dosage, imputation probabilities, and variant annotation.
- Sample call rate, variant call rate, allele frequency, Hardy-Weinberg checks, and heterozygosity.
- Sex checks, duplicates, relatedness, and sample swaps.
- Population structure and ancestry principal components.
- Quantitative and binary trait association models.
- Additive, dominant, recessive, and interaction models.
- Linear and logistic mixed models for relatedness.
- Rare-variant burden and kernel tests.
- Linkage disequilibrium and clumping.
- Genome-wide thresholds and false-discovery control.
- Genomic inflation, QQ plots, and calibration.
- Fine mapping, conditional analysis, colocalization, and credible sets.
- Polygenic scores, cross-validation, ancestry transferability, and overfitting.
- Mendelian randomization assumptions and pleiotropy.

**Data types to use**

- SNP dosage matrices with quantitative traits.
- Case-control genotype data.
- Family or biobank cohorts with related individuals.
- Summary statistics plus LD reference data.

**Decision-making goals**

- Determine whether population structure and relatedness require PCs, kinship models, or stratification.
- Separate association, fine-mapping probability, prediction, and causal claims.
- Avoid data leakage when tuning and testing polygenic scores.
- Assess whether a reference panel and study population are compatible.

### Topic 25: Proteomics and metabolomics

**Description:** Learn statistics for continuous abundance measurements with heteroscedasticity, batch effects, peptide-to-protein summarization, missing values, and censoring. Current Bioconductor tools include integrated filtering, normalization, imputation, and limma-based testing in DEP, and model-based protein significance analysis in MSstats.[^36][^37]

**Subtopics**

- DDA, DIA, targeted assays, label-free, and isobaric designs.
- Peptide-spectrum matches, peptides, proteins, metabolites, and features.
- Log transformation and variance stabilization.
- Normalization and run-order drift.
- Peptide-to-protein summarization.
- Shared peptides and ambiguous identifiers.
- Missing-at-random versus abundance-dependent censoring.
- Batch, plate, run, channel, and plex effects.
- Linear models, mixed models, and empirical Bayes moderation.
- Differential abundance and fold-change thresholds.
- Quality-control pools and reference channels.
- Multiple testing and pathway analysis.

**Data types to use**

- Label-free protein-intensity matrices.
- TMT reporter-ion data with plex metadata.
- Peptide-level DIA measurements.
- Metabolite intensities with pooled QC and run order.

**Decision-making goals**

- Decide whether missing values reflect censoring, stochastic identification, or technical failure.
- Choose the level-peptide, protein, or metabolite-at which the hypothesis is defined.
- Include run, plex, and batch structure in the analysis rather than relying only on global correction.

### Topic 26: Microbiome and compositional data

**Description:** Learn why relative abundances are constrained and ordinary correlations or unconstrained tests can be misleading. Current ANCOM-BC2 methods address sample-specific sampling fractions and taxon-specific sequencing-efficiency bias and support covariates, repeated measurements, and multi-group testing.[^38][^39][^40]

**Subtopics**

- Amplicon sequence variants, operational taxonomic units, taxa, and metagenomic features.
- Library size, prevalence, sparsity, and structural zeros.
- Relative abundance and the simplex.
- Compositional closure and spurious correlations.
- Log-ratio transformations: additive, centred, and isometric log ratios.
- Zero handling and pseudocount sensitivity.
- Alpha diversity and its uncertainty.
- Beta diversity, PERMANOVA, dispersion, and constrained ordination.
- Differential abundance with bias correction.
- Repeated measures and covariate adjustment.
- Taxonomic hierarchy and phylogenetic structure.
- Correlation networks under compositional constraints.

**Data types to use**

- 16S taxon-count tables.
- Shotgun metagenomic species or pathway counts.
- Absolute abundance measurements with spike-ins.
- Longitudinal microbiome samples.

**Decision-making goals**

- Determine whether the claim concerns relative or absolute abundance.
- Avoid ordinary correlation on closed compositions without justification.
- Distinguish location differences from dispersion differences in multivariate distance tests.
- Define prevalence and filtering rules independently of the outcome when possible.

### Topic 27: Spatial transcriptomics and spatial omics

**Description:** Learn that neighbouring spots or cells are not independent and that spatial questions concern gradients, domains, neighbourhoods, colocalization, and sample-level differences. Recent Bioconductor developments include functional-data methods and tests for multiscale colocalization and spatial patterns in replicated studies.[^41]

**Subtopics**

- Spots, cells, fields of view, slides, tissue sections, patients, and spatial coordinates.
- Spatial autocorrelation and neighbourhood graphs.
- Spatially variable genes.
- Tissue domains and boundary detection.
- Cell-type deconvolution and uncertainty.
- Spatial colocalization and interaction.
- Differential spatial patterns between conditions.
- Replication at patient or specimen level.
- Nested slide, region, and patient structure.
- Multiple testing across genes, cell pairs, scales, and regions.
- Edge effects, tissue geometry, and uneven cell density.
- Registration and alignment uncertainty.

**Data types to use**

- Visium spot-by-gene counts.
- Imaging-based cell-by-gene matrices with coordinates.
- Spatial proteomics or imaging cytometry.
- Multi-patient tissue sections.

**Decision-making goals**

- Distinguish within-section spatial evidence from between-patient biological inference.
- Select statistics that account for autocorrelation and tissue geometry.
- Avoid treating spots or cells as independent patient replicates.
- Separate descriptive maps from replicated condition comparisons.

### Topic 28: Time-course, longitudinal, and survival data

**Description:** Learn to model time as more than a set of unrelated group labels. For expression time courses, regression splines and treatment-by-time interactions can test differences between smooth trajectories; survival analyses require censoring-aware estimands and models.[^42][^43]

**Subtopics**

- Cross-sectional age effects versus within-subject change.
- Baseline adjustment and change scores.
- Repeated-measures mixed models.
- Random slopes and nonlinear trajectories.
- Splines, generalized additive models, and functional data analysis.
- Treatment-by-time interactions.
- Irregular observation times and informative dropout.
- Time-to-event outcomes, censoring, and risk sets.
- Kaplan-Meier estimation and log-rank tests.
- Cox proportional-hazards models.
- Proportional-hazards diagnostics.
- Competing risks and multi-state models.
- Time-dependent covariates and landmarking.
- Longitudinal-survival joint models.

**Data types to use**

- Repeated omics from ageing cohorts.
- Perturbation time-course RNA-seq.
- Serial immune profiling.
- Clinical survival data linked to molecular biomarkers.

**Decision-making goals**

- Decide whether the target is cross-sectional difference, individual change, trajectory difference, or event risk.
- Avoid treating repeated visits as independent.
- Use flexible time functions when linearity is implausible.
- Check whether a biomarker is prognostic, predictive, time-varying, or affected by immortal-time bias.

## Part VI: Gene sets and systems-level inference

### Topic 29: Functional enrichment and pathway statistics

**Description:** Learn how pathway results inherit assumptions and biases from feature selection, gene universes, ranking statistics, gene-gene correlation, annotation databases, and multiple testing. Current Bioconductor workflows support over-representation, competitive gene-set tests such as CAMERA, rotation tests, and preranked enrichment.[^44][^45]

**Subtopics**

- Over-representation analysis versus ranked enrichment.
- Competitive versus self-contained null hypotheses.
- Gene universe and detectability bias.
- Gene-set size and overlapping pathways.
- Directional versus non-directional effects.
- Inter-gene correlation.
- Ranking by test statistic, signed effect, or p-value-derived scores.
- Permutation of genes versus samples.
- Multiple testing across pathways.
- Redundancy reduction and network-based interpretation.
- Annotation versioning and identifier mapping.
- Cell-type-specific and multi-contrast enrichment.

**Data types to use**

- Full ranked differential-expression statistics.
- A pre-specified set of discovered genes plus an explicit tested-gene universe.
- CpG, peak, or protein statistics mapped to genes.
- Multiple condition contrasts.

**Decision-making goals**

- Choose a test whose null hypothesis matches the biological claim.
- Prefer complete ranked statistics when thresholded gene lists discard useful information.
- Correct for feature-to-gene mapping and detectability biases.
- Report pathway redundancy and leading features rather than treating pathway names as independent discoveries.

### Topic 30: Networks and multivariate integration

**Description:** Learn how co-expression, graphical models, latent factors, and multi-omics integration summarize coordinated biology while creating substantial tuning, stability, and validation challenges.

**Subtopics**

- Co-expression networks and adjacency construction.
- Modules, eigengenes, hubs, and module-trait association.
- Partial correlation and graphical models.
- Regularized covariance and precision matrices.
- Canonical correlation and partial least squares.
- Matrix factorization and latent-factor models.
- Joint versus data-block-specific variation.
- Supervised versus unsupervised integration.
- Sample matching and incomplete assay blocks.
- Batch and modality effects.
- Feature-selection stability.
- Permutation testing and external validation.

**Data types to use**

- Matched transcriptomics and proteomics.
- Methylation, expression, and phenotype measurements.
- Multi-tissue data from the same donors.
- MultiAssayExperiment-style cohorts with incomplete overlap.

**Decision-making goals**

- Decide whether the aim is visualization, latent biology, prediction, or mechanistic inference.
- Keep cross-validation at the biological-unit level.
- Distinguish shared technical factors from shared biology.
- Validate modules or latent factors in independent data.

## Part VII: Prediction and causal reasoning

### Topic 31: Statistical learning and biomarker prediction

**Description:** Learn prediction as a separate objective from association testing. A model can contain strongly associated variables yet predict poorly, or predict well using variables that should not receive causal interpretations.

**Subtopics**

- Training, validation, and test sets.
- Resampling, cross-validation, nested cross-validation, and bootstrap validation.
- Leakage through normalization, imputation, feature selection, batch correction, or repeated samples.
- Linear and logistic regression baselines.
- Ridge, lasso, elastic net, and stability selection.
- Trees, forests, boosting, support-vector machines, and neural models.
- Hyperparameter tuning.
- Class imbalance and prevalence shift.
- Discrimination: ROC-AUC and precision-recall.
- Calibration, calibration slope, Brier score, and decision curves.
- Continuous-outcome metrics: RMSE, MAE, and explained variance.
- External, temporal, geographical, and cross-platform validation.
- Model interpretability versus causal interpretation.

**Data types to use**

- Gene-expression disease classifiers.
- Methylation age prediction.
- Multi-omics survival-risk models.
- Cytometry-based response prediction.

**Decision-making goals**

- Split by donor or patient, never randomly across their cells or repeated samples.
- Place every learned preprocessing step inside resampling.
- Choose metrics based on intended use, prevalence, and decision costs.
- Distinguish apparent, internally validated, and externally validated performance.

### Topic 32: Causal inference for observational bioinformatics

**Description:** Learn why regression adjustment does not automatically produce causal effects. Causal inference requires a defined intervention or exposure, a causal structure, assumptions about exchangeability and selection, and sensitivity to unmeasured bias.

**Subtopics**

- Association, prediction, and causation.
- Potential outcomes and causal estimands.
- Directed acyclic graphs.
- Confounders, mediators, colliders, and instruments.
- Total, direct, and indirect effects.
- Matching, stratification, weighting, and regression adjustment.
- Propensity scores and overlap.
- Target-trial emulation.
- Time-varying confounding.
- Mediation analysis and high-dimensional mediators.
- Negative controls and sensitivity analyses.
- Mendelian randomization assumptions.
- Batch and selection mechanisms as causal structures.

**Data types to use**

- Observational cohorts linking exposure, omics, and disease.
- Longitudinal ageing data.
- Genetic-instrument and molecular-trait data.
- Case-control molecular profiles with selection concerns.

**Decision-making goals**

- Draw the assumed causal graph before choosing adjustment variables.
- Do not adjust automatically for every available covariate.
- Separate mediation questions from confounding-control questions.
- State causal assumptions that cannot be verified from the data.

### Topic 33: Bayesian reasoning and hierarchical shrinkage

**Description:** Learn Bayesian estimation as a framework for combining likelihoods with prior information and propagating uncertainty. Hierarchical and empirical-Bayes ideas are already embedded in widely used omics methods through dispersion, variance, and effect-size shrinkage.

**Subtopics**

- Prior, likelihood, posterior, and posterior predictive distribution.
- Weakly informative, regularizing, and domain-informed priors.
- Credible intervals versus confidence intervals.
- Posterior probabilities and decision thresholds.
- Hierarchical models and partial pooling.
- Empirical Bayes versus fully Bayesian modelling.
- Shrinkage of variances, dispersions, and effects.
- Bayesian model checking and posterior predictive checks.
- Bayes factors and their prior sensitivity.
- Multilevel omics models.
- Computation: optimization, Markov chain Monte Carlo, and variational inference.

**Data types to use**

- Small-sample multi-group expression studies.
- Multi-level donor/tissue measurements.
- GWAS variable-selection problems.
- Meta-analysis across related studies.

**Decision-making goals**

- Choose priors that regularize without silently determining the answer.
- Interpret posterior probabilities without translating them into p-values.
- Assess convergence and posterior predictive adequacy.
- Understand when empirical-Bayes borrowing across features improves stability.

## Part VIII: Reliability and reproducibility

### Topic 34: Model diagnostics and sensitivity analysis

**Description:** Learn to challenge every important conclusion with diagnostics and alternative reasonable specifications. A statistically sophisticated model is not trustworthy if it depends on one undocumented filtering rule, one outlier, or one batch-correction setting.

**Subtopics**

- Residual, leverage, influence, and calibration diagnostics.
- Distributional and mean-variance checks.
- Alternative transformations and link functions.
- Robust standard errors and robust regression.
- Leave-one-sample and leave-one-batch analyses.
- Alternative covariate sets based on causal reasoning.
- Alternative normalization and filtering rules.
- Sensitivity to cell-type labels and cluster resolutions.
- Negative controls and falsification outcomes.
- Multiverse and specification-curve reasoning.
- Replication in independent datasets.

**Data types to use**

- Any completed differential or regression analysis.
- Multi-batch datasets.
- Small cohorts with influential samples.
- Analyses involving inferred cell types or latent factors.

**Decision-making goals**

- Identify which conclusions are stable and which are specification-dependent.
- Distinguish a diagnostic warning from a cosmetic imperfection.
- Report sensitivity findings as part of the scientific result.

### Topic 35: Reproducible statistical workflows

**Description:** Build analyses so that data provenance, software versions, parameters, random seeds, environments, and outputs can be audited and regenerated. The goal is not merely rerunning code, but reproducing the statistical decisions that generated each claim.

**Subtopics**

- Project structure separating raw data, metadata, derived data, results, and reports.
- Data dictionaries and schema validation.
- Version control and atomic commits.
- Dependency locking and environment capture.
- Workflow orchestration and dependency graphs.
- Parameterized reports and literate analysis.
- Random seeds and deterministic versus nondeterministic algorithms.
- Unit tests for transformations and statistical helper functions.
- Continuous integration for analysis checks.
- Machine-readable result tables.
- Session and package-version recording.
- Provenance for annotation databases and gene sets.
- Containers and portable execution.
- Privacy, controlled-access data, and publishable derived outputs.

CRAN task views provide maintained maps of relevant R packages but explicitly do not endorse a single "best" package, reinforcing the need to select methods by estimand, assumptions, and data structure rather than popularity.[^46]

**Data types to use**

- A complete public Bioconductor experiment dataset.
- An analysis combining raw assay data, metadata, and annotation.
- A multi-stage workflow with quality control, modelling, enrichment, and reporting.

**Decision-making goals**

- Make every table and figure traceable to inputs and parameters.
- Separate immutable raw data from generated outputs.
- Preserve enough metadata to rerun an analysis after package or annotation updates.
- Test analytical assumptions and data alignment automatically.

## Cross-cutting decision map

| If the outcome is... | First model family to understand | Typical bioinformatics example | Main complication |
|---|---|---|---|
| Continuous and approximately symmetric | Linear model | Log protein intensity | Heteroscedasticity, batch, missingness |
| Continuous and strongly skewed | Transformation, robust model, or suitable GLM | Cytokine concentration | Outliers and detection limits |
| Binary | Logistic regression | Case/control status | Separation, imbalance, calibration |
| Count | Poisson or negative-binomial GLM | RNA-seq reads, cluster counts | Overdispersion and exposure/library size |
| Proportion with denominator | Binomial or beta-binomial model | Positive cells out of total cells | Overdispersion and dependence |
| Continuous proportion in 0-1 | Beta or transformed model | Methylation beta value | Boundary values and interpretability |
| Composition | Log-ratio or composition-aware model | Microbiome relative abundance | Closure and zeros |
| Ordinal | Ordinal regression | Pathology grade | Proportional-odds assumption |
| Repeated or clustered | Mixed model, GEE, or aggregation | Multiple cells/visits per donor | Correct unit of inference |
| Time-to-event | Survival model | Time to relapse | Censoring and proportional hazards |
| High-dimensional multivariate | PCA/factor model for exploration | Expression matrix | Scaling, batch, circular interpretation |
| Multiple linked assays | Multi-block/integrative model | RNA, protein, methylation | Sample matching and shared technical effects |

## Method-selection questions

### When comparing two groups

1. What is the independent biological unit?
2. Are observations paired, blocked, nested, or independent?
3. Is the target a mean, median, distributional shift, proportion, count rate, or fold change?
4. Is a Gaussian sampling model plausible after an interpretable transformation?
5. Is variance similar enough for a pooled model, or should Welch/heteroscedastic methods be used?
6. Is the sample size sufficient for asymptotic inference, or is randomization/permutation justified by design?
7. What effect is biologically meaningful?
8. How many outcomes or features are tested?

### When testing thousands of features

1. What filtering is independent of the outcome?
2. What model matches the assay's mean-variance relationship?
3. Which covariates and blocks belong in the design matrix?
4. What is the exact contrast?
5. What defines the multiplicity family?
6. Are effect estimates shrunken, moderated, transformed, or directly interpretable?
7. Do diagnostics support the fitted model?
8. Are pathway analyses based on all tested features and a valid background?

### When analysing single-cell or cytometry data

1. Are donors or specimens replicated across conditions?
2. Which operations are cell-level exploration and which produce sample-level inference?
3. Is the target differential abundance, differential state, or differential expression?
4. Should cells be aggregated by sample and cell type?
5. Are low-cell sample-cluster combinations informative enough?
6. Is cell-type annotation fixed, uncertain, or outcome-dependent?
7. Are multiple genes, markers, populations, and contrasts jointly accounted for?

### When building a predictor

1. What population and future use does the test set represent?
2. Are all samples from the same donor assigned to one resampling partition?
3. Are normalization, imputation, batch handling, and feature selection estimated only from training data?
4. Is the metric aligned with the intended decision and class prevalence?
5. Is performance calibrated as well as discriminative?
6. Has the complete pipeline been externally validated?

## Recommended learning order

### Stage 1: Core inference

1. Statistical units and experimental design.
2. Measurement scales and distributions.
3. Exploratory analysis and data quality.
4. Estimation, effect sizes, and confidence intervals.
5. Hypothesis tests and p-values.
6. t-tests, permutation tests, and power.
7. Multiple testing.

**Milestone:** Given a simple study, identify the estimand, replicate, test/model, effect size, assumptions, multiplicity family, and defensible conclusion.

### Stage 2: Unified modelling

1. Correlation and dependence.
2. Linear regression.
3. ANOVA and contrasts.
4. Generalized linear models.
5. Mixed models.
6. Missing-data mechanisms.
7. Diagnostics and sensitivity analysis.

**Milestone:** Express t-tests, group comparisons, covariate adjustment, interactions, counts, and repeated measures through explicit model formulas and contrasts.

### Stage 3: High-dimensional analysis

1. Matrix algebra.
2. PCA.
3. Distances and clustering.
4. Batch effects.
5. Feature-wise empirical Bayes inference.
6. Gene-set testing.
7. Reproducible workflows.

**Milestone:** Explain which operations are exploratory, which are inferential, where information is borrowed across features, and how preprocessing changes the target of inference.

### Stage 4: Modality tracks

Choose at least three tracks:

- Bulk RNA-seq.
- Single-cell RNA-seq.
- Flow/CyTOF/imaging cytometry.
- DNA methylation.
- GWAS/statistical genetics.
- Proteomics/metabolomics.
- Microbiome/compositional analysis.
- Spatial omics.
- Longitudinal and survival analysis.

**Milestone:** For each chosen modality, map raw measurement, normalized representation, experimental unit, likelihood/model, contrast, multiplicity strategy, diagnostic plots, and interpretation limits.

### Stage 5: Advanced reasoning

1. Prediction and nested validation.
2. Causal inference.
3. Bayesian and hierarchical modelling.
4. Multi-omics integration.
5. Robustness, transportability, and external replication.

**Milestone:** Distinguish discovery, prediction, and causal estimation, and design validation appropriate to each goal.

## Literature-informed priorities for 2026

- **Biological replication before cell count:** Single-cell differential analyses should preserve donor-level replication. Pseudobulk or subject-aware mixed models remain better-supported defaults than naive per-cell tests.[^27][^24][^25][^22]
- **Effect sizes before dichotomies:** Interpret exact p-values with effect estimates, uncertainty intervals, design assumptions, and biological relevance rather than relying on a universal 0.05 boundary.[^7][^8][^6][^9]
- **Batch prevention before correction:** Randomization, balanced processing, common references, and pre/post-correction diagnostics remain more defensible than attempting to recover biology from completely confounded batches.[^47][^18]
- **Assay-aware likelihoods:** Raw RNA-seq counts belong in count-aware methods such as DESeq2 or edgeR; transformed values may be used in methods such as voom when the mean-variance relationship is explicitly modelled.[^19][^21][^22]
- **Compositional awareness:** Microbiome and cell-composition measurements require methods that respect sampling fractions, taxon-specific biases, totals, and compositional constraints.[^40][^38]
- **Replicated spatial inference:** Spatial maps alone do not provide patient-level replication. Newer spatial methods increasingly focus on comparing spatial functions or patterns across replicated samples and nested designs.[^41]
- **Reproducible containers and environments:** `SummarizedExperiment` and `MultiAssayExperiment` remain foundational for keeping measurements aligned with metadata, while Bioconductor 3.23 targets R 4.6.[^3][^1][^2]

## How this curriculum is implemented

This document is the **syllabus**: it states, for each topic, the decision to
be made and the consequence of getting it wrong. Three companion pieces turn
it into something you can run.

| File | What it gives you |
|---|---|
| `stats.md` | the **mathematics**: Part 0 (statistics from zero, no prerequisites) plus 35 topics matching the ones below, 305 numbered equations, a notation table, and appendices |
| `statsPy/`, `statsR/` | the **code**: 46 modules per language (6 foundations, 25 core, 11 applied, 4 integrative exercises), each demonstrating a topic on simulated data with a known truth |
| `README.md` | **how to run everything**, in Docker, with nothing installed locally |

The numbering is shared. Topic *n* below corresponds to equations `(n.x)` in
`stats.md`; the module map in `README.md` names the script for each topic.

Every module is built the same way: it performs the naive analysis first,
measures how wrong it is against the simulated truth, then does it correctly
and measures that too. Most end by running the whole pipeline on data with no
signal, because a method that cannot return "nothing" on noise is not usable.

### Repository layout

```
stats.md                  the mathematical companion (Part 0 + Topics 1-35)
README.md                 how to build, run, learn and test
statsPy/  statsR/         foundations/ (F1-F6), core/ (00-24),
                          bioinformatics/ (30-40), exercises/ (E1-E4),
                          notebooks/ (generated)
env/  docker/             pinned manifests and the two images
tools/                    test runners, solution checker, notebook builder
```

Scripts are the source of truth; `.ipynb` and `.Rmd` notebooks are generated
from them by `tools/build_notebooks.sh` and should not be edited directly.

The organizing question throughout is **"What statistical decision is being
made, under which assumptions, for which biological unit and data type?"**
rather than **"Which function should be called?"**


## References

[^1]: [Bioconductor 3.23 Released](https://www.bioconductor.org/news/bioc_3_23_release/) - Bioconductor 3.23, consisting of 2418 software packages, 437 experiment data packages, 928 annotatio...
[^2]: [SummarizedExperiment](https://bioconductor.org/packages/release/bioc/html/SummarizedExperiment.html) - The SummarizedExperiment container contains one or more assays, each represented by a matrix-like ob...
[^3]: [MultiAssayExperiment](https://www.bioconductor.org/packages/devel/bioc/manuals/MultiAssayExperiment/man/MultiAssayExperiment.pdf)
[^4]: [SummarizedExperiment for Coordinating Experimental Assays, Samples, and Regions of Interest](https://www.bioconductor.org/packages/release/bioc/vignettes/SummarizedExperiment/inst/doc/SummarizedExperiment.html)
[^5]: [MultiAssayExperiment: Quick Start Guide](https://www.bioconductor.org/packages/devel/bioc/vignettes/MultiAssayExperiment/inst/doc/QuickStartMultiAssay.html)
[^6]: [The ASA Statement on p-Values: Context, Process, and Purpose](https://www.tandfonline.com/doi/full/10.1080/00031305.2016.1154108) - Published in The American Statistician (Vol. 70, No. 2, 2016)
[^7]: [ASA President's Task Force Statement on Statistical Significance ...](https://magazine.amstat.org/blog/2021/08/01/task-force-statement-p-value/)
[^8]: [Moving to a World Beyond "p < 0.05" - Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/00031305.2019.1583913) - Published in The American Statistician (Vol. 73, No. sup1, 2019)
[^9]: [Exploring the proper use of p-values and confidence intervals ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12834938/) - Misinterpretation of null-hypothesis tests (p-values) and confidence intervals has been a longstandi...
[^10]: [Adjust P-values for Multiple Comparisons](https://stat.ethz.ch/R-manual/R-patched/library/stats/html/p.adjust.html)
[^11]: [Adjust P-values for Multiple Comparisons - R](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
[^12]: [CRAN Task View: Mixed, Multilevel, and Hierarchical Models ...](https://cran.r-project.org/web/views/MixedModels.html)
[^13]: [CRAN Task View: Mixed, Multilevel, and Hierarchical Models in R](https://cran.r-project.org/view=MixedModels) - Mixed (or mixed-effect) models are a broad class of statistical models used to analyze data where ob...
[^14]: [CRAN Task View: Missing Data - R Project](https://cran.r-project.org/view=MissingData) - Missing data are very frequently found in datasets. Base R provides a few options to handle them usi...
[^15]: [CRAN Task View: Missing Data - R Project](https://cran.r-project.org/web/views/MissingData.html)
[^16]: [Principal component analysis based methods in bioinformatics studies](https://pmc.ncbi.nlm.nih.gov/articles/PMC3220871/) - ...bioinformatics data, a unique challenge arises from the high dimensionality of measurements. With...
[^17]: [Principal component analysis | Nature Methods](https://www.nature.com/articles/nmeth.4346) - PCA helps you interpret your data, but it will not always find the important patterns.
[^18]: [Assessing and mitigating batch effects in large-scale omics studies - Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03401-9) - Batch effects in omics data are notoriously common technical variations unrelated to study objective...
[^19]: [RNA-seq workflow: gene-level exploratory analysis and ...](https://www.bioconductor.org/packages//release/workflows/vignettes/rnaseqGene/inst/doc/rnaseqGene.html)
[^20]: [RNA-seq workflow: gene-level exploratory analysis and ...](https://bioconductor.org/help/course-materials/2022/CSAMA/lab/2-tuesday/lab-03-rnaseq/rnaseqGene_CSAMA2022.html)
[^21]: [Analyzing RNA-seq data with DESeq2](https://bioconductor.org/packages/devel/bioc/vignettes/DESeq2/inst/doc/DESeq2.html)
[^22]: [edgeR: differential analysis of sequence read count data User's Guide](https://bioconductor.org/packages//release/bioc/vignettes/edgeR/inst/doc/edgeRUsersGuide.pdf)
[^23]: [Differential state analysis with muscat](https://bioconductor.org/packages/release/bioc/vignettes/muscat/inst/doc/analysis.html)
[^24]: [Confronting false discoveries in single-cell differential expression](https://www.nature.com/articles/s41467-021-25960-2) - Differential expression analysis of single-cell transcriptomics allows scientists to dissect cell-ty...
[^25]: [Benchmarking methods for detecting differential states ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9487674/) - Single-cell RNA-sequencing (scRNA-seq) enables researchers to quantify transcriptomes of thousands o...
[^26]: [[PDF] muscat: Multi-sample multi-group scRNA-seq data analysis tools](https://www.bioconductor.org/packages/devel/bioc/manuals/muscat/man/muscat.pdf)
[^27]: [Single-cell differential expression analysis between conditions within nested settings](https://www.biorxiv.org/content/10.1101/2024.08.01.606200v1.full.pdf)
[^28]: [CATALYST: Cytometry dATa anALYSis Tools](https://bioconductor.org/packages/devel/bioc/manuals/CATALYST/man/CATALYST.pdf)
[^29]: [diffcyt workflow - Bioconductor](https://www.bioconductor.org/packages/release/bioc/vignettes/diffcyt/inst/doc/diffcyt_workflow.html)
[^30]: [diffcyt.pdf](https://www.bioconductor.org/packages/devel/bioc/manuals/diffcyt/man/diffcyt.pdf)
[^31]: [Contents](https://bioconductor.org/packages/release/bioc/vignettes/missMethyl/inst/doc/missMethyl.html)
[^32]: [A cross-package Bioconductor workflow for analysing ...](https://www.bioconductor.org/packages/release/workflows/vignettes/methylationArrayAnalysis/inst/doc/methylationArrayAnalysis.html)
[^33]: [missMethyl.pdf](https://bioconductor.org/packages/devel/bioc/manuals/missMethyl/man/missMethyl.pdf)
[^34]: [GWAS.BAYES](https://bioconductor.org/packages/release/bioc/html/GWAS.BAYES.html) - This package is built to perform GWAS analysis using Bayesian techniques. Currently, GWAS.BAYES has ...
[^35]: [GWASTools: Tools for Genome Wide Association Studies](https://bioconductor.org/packages/release/bioc/manuals/GWASTools/man/GWASTools.pdf)
[^36]: [DEP: Differential Enrichment analysis of Proteomics data](https://bioconductor.org/packages/release/bioc/manuals/DEP/man/DEP.pdf)
[^37]: [MSstats](https://www.bioconductor.org/packages//release/bioc/manuals/MSstats/man/MSstats.pdf)
[^38]: [ANCOMBC.pdf](https://www.bioconductor.org/packages/devel/bioc/manuals/ANCOMBC/man/ANCOMBC.pdf)
[^39]: [dar: Differential Abundance Analysis by Consensus](https://bioconductor.org/packages//release/bioc/manuals/dar/man/dar.pdf)
[^40]: [ANCOM-BC Tutorial - Bioconductorwww.bioconductor.org > devel > bioc > vignettes > ANCOMBC > inst > doc](https://www.bioconductor.org/packages/devel/bioc/vignettes/ANCOMBC/inst/doc/ANCOMBC.html)
[^41]: [36 Differential colocalization - Orchestrating Spatial Transcriptomics ...](https://bioconductor.org/books/release/OSTA/pages/mult-diff-colocalization.html) - CRAWDAD (Dos Santos Peixoto et al. 2025) is an R package for multi-scale characterization of spatial...
[^42]: [[PDF] limma: Linear Models for Microarray and RNA-Seq Data User's Guide](https://bioconductor.statistik.tu-dortmund.de/packages/3.12/bioc/vignettes/limma/inst/doc/usersguide.pdf)
[^43]: [moanin: An R Package for Time Course RNASeq Data Analysis](https://bioconductor.uib.no/packages/3.19/bioc/manuals/moanin/man/moanin.pdf)
[^44]: [Performing gene set enrichment analyses with sparrow - Bioconductor](https://bioconductor.org/packages//release/bioc/vignettes/sparrow/inst/doc/sparrow.html) - The sparrow package facilitates the use of gene sets in the analysis of high throughput genomics dat...
[^45]: [fgsea: Fast Gene Set Enrichment Analysis - Bioconductor](https://bioconductor.org/packages//release/bioc/manuals/fgsea/man/fgsea.pdf)
[^46]: [Task Views](https://cran.r-project.org/web/views/)
[^47]: [Correcting batch effects in large-scale multiomics studies using a ...](https://link.springer.com/article/10.1186/s13059-023-03047-z?error=cookies_not_supported&code=aaada5e0-43fa-4722-9164-a8bf8be5a0e6)
