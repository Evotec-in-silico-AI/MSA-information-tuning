# Reproduction

## 1. Create the analysis environment

```bash
conda env create -f environments/analysis_environment.yml
conda activate oss
```

The environment uses Python 3.10. The Chai-1 environment is required only for Chai-1 inference and uses Python 3.12.10 on Linux.

## 2. Install companion data

Overlay the companion record onto the repository root, then extract the starting MSA archive:

```bash
cd cofolding_study/core_files
tar -xJf msas.tar.xz
cd ../..
python scripts/manage_data.py status
```

The P29972 and P07740 BLAST retrievals and parental alignments used by the documented example runs are supplied by the companion data release.

## 3. Repository smoke test

```bash
python scripts/run_workflow.py test
```

The smoke test checks the repository layout, compact data manifest, locked-code checksums, direct analysis packages, and a deterministic FASTA/A3M fixture. The fixture asserts alignment depth, width, gap count, non-gap occupancy, and substitutions relative to the reference. Expected resources are one CPU, less than 1 GB RAM, no GPU, and less than one minute.

## 4. Main figures

Inspect commands and paths for all main figures:

```bash
python scripts/run_workflow.py figures --check-only
```

Run all main figures:

```bash
python scripts/run_workflow.py figures
```

The default figure set is `1,2,3,4,5,6,7`. Figures 5 and 6 require `figure_generation/compiled/final/per_site/` from the companion release. Figure 7 uses retained folding summaries and structural examples.

A repository-resident quick run that does not require the large compiled residue-level tables is:

```bash
python scripts/run_workflow.py figures --figures 1,2,3,4 --check-only
python scripts/run_workflow.py figures --figures 1,2,3,4
```

Optional supplementary and sensitivity workflows are included with:

```bash
python scripts/run_workflow.py figures --include-supplementary --include-sensitivity --check-only
python scripts/run_workflow.py figures --include-supplementary --include-sensitivity
```

After all figure and sensitivity workbooks have been regenerated, rebuild the six publication-facing Supplementary Data workbooks and their inventory:

```bash
cd figure_generation
python build_supplementary_data_files.py
cd ..
```

Run the packaging command only after the upstream figure and sensitivity workbooks are final, because it reads those workbooks and consolidates their publication-facing tables.

Figure generation uses the `oss` analysis environment and does not require a GPU when the processed companion data are installed.

## 5. Processed-data demonstration

Figure 1 provides a lightweight scientific demonstration using repository-resident processed data:

```bash
python scripts/run_workflow.py figures --figures 1 --check-only
python scripts/run_workflow.py figures --figures 1
```

The workflow writes PNG and SVG panels, legends, Supplementary Figure 1a–e outputs, and `figure1_and_supplementary_figure1_source_data.xlsx` under `figure_generation/figure1_and_supplementary_figure1_outputs/`. Record the clean-environment runtime and compare the output set with the tracked reference outputs before release.

## 6. Example manuscript proteins

Two proteins provide compact entry points into the full MDLM workflow:

- P29972, used in Supplementary Figure 1a, with 15 historical pipeline p10 targets from 2.05 to 2.75;
- P07740, used in Supplementary Figure 1b and Figure 6a, with 16 historical pipeline p10 targets from 2.35 to 3.10.

Inspect the exact commands:

```bash
python scripts/run_workflow.py examples --check-only
```

Run the manuscript settings:

```bash
python scripts/run_workflow.py examples
```

These runs train a separate model for each targeted and matched-control MSA and are substantially more expensive than the smoke test or figure regeneration.

## 7. Folding benchmark

Chai-1 inference uses `environments/chai1_environment.yml` and requires a supported Linux CUDA GPU. Exact commands for MSA tuning and Chai-1 inference are documented in `docs/software_and_compute.md`.

The structural comparison step requires TM-score. Obtain it from https://zhanggroup.org/TM-score/ and pass its path explicitly:

```bash
python cofolding_study/core_files/compare_chai1_to_experimental.py   --mode jackhmmer   --tmscore-exec /path/to/TMscore
```

Do not rely on the historical developer-specific default embedded in the locked script.

## 8. Full study stages

1. Retrieve homologues with BLASTP.
2. Build parental MSAs with MUSCLE Super5.
3. Construct information-targeted and matched-depth control MSAs.
4. Train one MDLM denoiser per MSA and generate sequences.
5. Run MSA, latent, and structural analyses.
6. Compile the cohort data.
7. Run the independent JackHMMER and Chai-1 folding benchmark.
8. Generate manuscript source data and figures.

The original scheduler wrappers are not distributed because they were cluster-specific. Python entry points and historical resources are recorded in `docs/software_and_compute.md`; users should adapt scheduler directives to their systems.


Historical p10 target values are retained in identifiers for reproducibility. Figure-generation scripts convert achieved ProDy-derived information coordinates to bits before analysis and reporting. The historical operational cutoff of 2.0 in the archived pipeline coordinate corresponds to 0.972093947 bits/site and is used for all regime-based analyses; figures and manuscript text generally describe it as approximately 1 bit/site.
