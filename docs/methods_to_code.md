# Methods-to-code map

## Cohort generation and analysis

- Homologue retrieval, MSA preparation, MDLM training, and generation entry point: `MDLM_pipeline/main.py`.
- MSA construction and tuning: `MDLM_pipeline/modules/alignment.py`, `alignment_compression.py`, and `msa_context.py`.
- Model training: `MDLM_pipeline/modules/train_mdlm.py`.
- Generation and orchestration: `MDLM_pipeline/modules/wrapper.py`.
- MSA analysis: `MDLM_pipeline/analysis/msa_analysis.py`.
- Latent analysis: `MDLM_pipeline/analysis/latent_geometry_analysis.py`.
- Structural analysis: `MDLM_pipeline/structural_analysis.py`.
- Cohort compilation: `MDLM_pipeline/analysis/build_cohort_dataset.py` and the cohort-builder scripts.

The companion release provides upstream BLAST/MSA inputs and the final consolidated residue-level tables used by Figures 5 and 6. Complete training runs and model checkpoints are not required for figure regeneration.

## Independent folding benchmark

- Tune MSAs: `cofolding_study/core_files/tune_msas_for_folding.py`.
- Run Chai-1: `cofolding_study/core_files/run_chai1.py`.
- Compare predictions with experimental structures: `cofolding_study/core_files/compare_chai1_to_experimental.py`.
- Summary analyses: `cofolding_study/core_files/figure7_meta_plot.py` and `cofolding_study/core_files/supplementary_figure5cd_plot.py`.

## Figure and panel map

| Manuscript output | Primary code | Principal inputs | Environment |
|---|---|---|---|
| Figure 1 and Supplementary Figure 1 | `figure_generation/figure1_plot.py` | `figure_generation/cohort.xlsx`, `data_summary.xlsx` | `oss` |
| Figure 2 and Supplementary Figure 2 | `figure_generation/figure2_plot.py` | `figure_generation/cohort.xlsx`, `data_summary.xlsx` | `oss` |
| Figure 3 | `figure_generation/figure3_plot.py` | repository-resident cohort summaries | `oss` |
| Figure 4 | `figure_generation/figure4_plot.py` | repository-resident cohort summaries | `oss` |
| Figure 5 and Supplementary Figure 3 | `figure_generation/figure5_plot.py` | `figure_generation/compiled/final/per_site/`, cohort metadata | `oss` |
| Figure 6 and Supplementary Figure 4 | `figure_generation/figure6_plot.py`, `figure6_mediation_plot.py` | `figure_generation/compiled/final/per_site/`, cohort metadata | `oss` |
| Figure 7 | `figure_generation/figure7_plot.py`, `figure7g_plot.py` | retained comparison tables, structures, scores, and tuned-MSA metadata | `oss` |
| Supplementary Figure 5c,d | `figure_generation/supplementary_figure5cd_plot.py` | retained residue-level Figure 7 comparison outputs | `oss` |
| Sensitivity analyses | `figure_generation/sensitivity_analysis.py` | repository-resident and companion processed inputs | `oss` |

All figure scripts are executed from `figure_generation/` by `scripts/run_workflow.py`. Source-data workbooks and supplementary data files are retained under `figure_generation/` and covered by `SOURCE_DATA_CHECKSUMS.sha256`.

## Reconstruction versus figure regeneration

Figure regeneration uses repository-resident workbooks plus processed companion data. Reconstructing the full study additionally requires BLASTP, MUSCLE Super5, MDLM training, Chai-1 inference, TM-score, suitable CPU/GPU resources, and external model/database assets described in the manuscript.

## Information-unit normalization

Archived cohort, compiled residue-level, and folding-result files retain the numerical coordinates produced by the frozen upstream workflows. Manuscript-facing scripts under `figure_generation/` normalize affected ProDy-derived information and effective-alphabet fields when those archived data are loaded. Historical p10 targets, paths, and Figure 7 condition identifiers remain unchanged for provenance.

All regime-based analyses use an operational threshold of 0.972093947 bits/site, the exact corrected equivalent of the numerical cutoff used by the frozen upstream workflows. Figures and manuscript text generally describe this threshold as approximately 1 bit/site. The frozen `MDLM_pipeline/` and `cofolding_study/` code and their archived outputs are not modified by this analysis-layer normalization.
