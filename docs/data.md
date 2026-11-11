# Companion data

## Release model

The companion data release overlays the matching tagged code release. Copy the data-release contents onto the repository root while preserving relative paths. Files already distributed with the code repository, including `figure_generation/cohort.xlsx` and `figure_generation/data_summary.xlsx`, are not duplicated in the companion record.

The release identifiers below must be completed when the public records are assigned:

```text
Code release tag: v1.0.0
Code commit: initial release (1)
Archived code DOI: 10.5281/zenodo.23674166
Data DOI: 10.5281/zenodo.23671811
Manuscript DOI: pending
```

## Retained data

The companion record contains:

- `MDLM_pipeline/input/` and `MDLM_pipeline/data/`: reference sequences, BLAST retrievals, parental alignments, and related reconstruction inputs;
- `figure_generation/compiled/final/`: 45 consolidated per-protein residue-level tables used by Figures 5 and 6;
- `cofolding_study/core_files/fasta/`: standard FASTA representations;
- `cofolding_study/core_files/chai1_fasta/`: Chai-1-formatted inputs;
- `cofolding_study/core_files/msas.tar.xz`: the 1,178 starting JackHMMER MSAs;
- experimental structures, information-tuned MSAs, tuning metadata, Chai-1 predictions, score files, comparison tables, and Figure 7 outputs.

Historical compilation intermediates under `figure_generation/compiled/archive/` are intentionally not distributed. The figure scripts read `figure_generation/compiled/final/per_site/` directly.

## Installation

From the code-repository root, copy or extract the companion data release so that its `MDLM_pipeline/`, `figure_generation/`, and `cofolding_study/` directories merge with the same directories in the code repository.

Restore the starting MSA directory:

```bash
cd cofolding_study/core_files
tar -xJf msas.tar.xz
cd ../..
```

Check selected required artifacts:

```bash
python scripts/manage_data.py status
```

The compact installer can copy or link the selected external files listed in `data/artifact_manifest.tsv`:

```bash
python scripts/manage_data.py install --data-root /path/to/companion_data --dry-run
python scripts/manage_data.py install --data-root /path/to/companion_data --copy
```

Direct overlay installation is recommended for the complete data release because it contains many files and directories beyond the compact install manifest.

## Verification

The data-release root contains the authoritative file-level `artifact_manifest.tsv` and `checksums.sha256`.

On Linux or macOS:

```bash
cd /path/to/companion_data
sha256sum --check checksums.sha256
```

The machine-readable `release_summary.json` records the validated release scope and target counts.

## Repository-local manifest

`data/artifact_manifest.tsv` uses:

```text
artifact_id
source_path
install_path
sha256
size_bytes
required_for
source
version
status
notes
```

`source_path` is relative to the companion-data root for an external artifact. `install_path` is relative to the code-repository root. Repository-resident entries are marked `included`; selected overlay files are marked `required_external`.

## Information-coordinate convention

Archived cohort and compiled tables retain the ProDy-derived pipeline coordinates used during MSA selection. Figure-generation code converts affected information and effective-alphabet fields at load time. Historical p10 targets in identifiers are provenance labels. Publication-facing outputs use corrected bits/site. All regime-based analyses use an operational threshold of 0.972093947 bits/site, generally described in figures and manuscript text as approximately 1 bit/site.
