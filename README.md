# Tuning evolutionary information in MSAs shapes protein generation and folding
![graphical_abstract](images/graphica_abstract.png)
This repository contains the code and repository-resident source data for our paper at https://www.biorxiv.org/content/10.64898/2026.11.23.768656v1. A separate companion data release supplies the larger inputs and processed products needed for complete reproduction.

## Repository structure

- `MDLM_pipeline/`: homologue retrieval, MSA construction and tuning, MDLM training, generation, and cohort analysis.
- `cofolding_study/`: the independent JackHMMER and Chai-1 folding benchmark.
- `figure_generation/`: manuscript figure scripts and repository-resident source-data workbooks.
- `scripts/`: public workflow and companion-data utilities.
- `docs/`: reproduction, data, software, compute, and methods-to-code documentation.
- `example_data/`: a small deterministic FASTA/A3M fixture used by the repository smoke test.

## Installation

The analysis environment is used for repository validation, MDLM analyses, and figure generation:

```bash
conda env create -f environments/analysis_environment.yml
conda activate oss
```

Chai-1 inference uses a separate Linux GPU environment:

```bash
conda env create -f environments/chai1_environment.yml
conda activate chai1
```

The Chai-1 workflow requires a CUDA-capable GPU with bfloat16 support. See `docs/software_and_compute.md` for platform and resource details.

## Companion data

Obtain the companion data record and overlay its contents onto the repository root while preserving relative paths. The current code release (v1.0.0) is available at 10.5281/zenodo.23674166, and data can be found at 10.5281/zenodo.23671811.

After overlay installation, extract the starting folding-benchmark MSAs:

```bash
cd cofolding_study/core_files
tar -xJf msas.tar.xz
cd ../..
```

Check the compact repository data requirements with:

```bash
python scripts/manage_data.py status
```

The companion record contains its own authoritative file-level `artifact_manifest.tsv`, `checksums.sha256`, and `release_summary.json`. The smaller `data/artifact_manifest.tsv` in this repository tracks only repository-resident files and selected externally installed artifacts.

## Repository smoke test

```bash
python scripts/run_workflow.py test
```

This validates the repository layout, direct dependencies, locked-code checksums, the compact data manifest, and a deterministic alignment fixture. It does not rerun the full study.

## Figures

Check the commands and required paths for all main figures:

```bash
python scripts/run_workflow.py figures --check-only
```

Generate all main figures after installing the companion data:

```bash
python scripts/run_workflow.py figures
```

Include the supplementary and sensitivity workflows with:

```bash
python scripts/run_workflow.py figures --include-supplementary --include-sensitivity --check-only
python scripts/run_workflow.py figures --include-supplementary --include-sensitivity
```

After regenerating all figure and sensitivity workbooks, rebuild the publication-facing Supplementary Data workbooks and inventory:

```bash
cd figure_generation
python build_supplementary_data_files.py
cd ..
```

See `docs/reproduction.md` for scoped runs, example proteins, expected inputs, and computational cost. A lightweight processed-data demonstration is available with:

```bash
python scripts/run_workflow.py figures --figures 1 --check-only
python scripts/run_workflow.py figures --figures 1
```

The expected output files and required inputs are described in `docs/reproduction.md`.

## Folding benchmark and TM-score

The structural comparison stage requires the official TM-score executable. Obtain it from the Zhang Group TM-score distribution, compile or install it for the target platform, and pass its location explicitly:

```bash
python cofolding_study/core_files/compare_chai1_to_experimental.py   --mode jackhmmer   --tmscore-exec /path/to/TMscore
```

The source retains the historical local default used during the study. Public users should always set `--tmscore-exec` or `TMSCORE_EXEC` rather than relying on that default.

Official TM-score distribution: https://zhanggroup.org/TM-score/

## Documentation

- `docs/reproduction.md`: executable workflows and expected resources.
- `docs/data.md`: companion-data layout, installation, and verification.
- `docs/software_and_compute.md`: versions, platform constraints, and historical compute.
- `docs/methods_to_code.md`: manuscript-to-code and figure-to-input map.

## Citation and archival identifiers

Use `CITATION.cff` for citation metadata. 

## License

The repository code is released under the MIT License. External software, model assets, databases, and deposited third-party-derived data remain subject to their respective terms.

## Information units

ProDy reports Shannon entropy using natural logarithms. Manuscript-facing MSA information values are converted to bits before analysis as `log2(20) - H / ln(2)`. Historical p10 targets embedded in identifiers and paths are retained as provenance labels and therefore do not numerically match corrected bits/site values. All regime-based analyses use an operational threshold of 0.972093947 bits/site, the exact corrected equivalent of the numerical cutoff used by the frozen upstream workflows. Figures and manuscript text generally describe this threshold as approximately 1 bit/site. Archived companion tables are normalized by the figure-generation code and should not be pre-converted.
