# Repository data requirements

`artifact_manifest.tsv` is the compact installation manifest used by `scripts/manage_data.py`. It is not the file-level manifest of the companion data deposit.

The companion data release is an overlay of this repository and contains the authoritative:

```text
artifact_manifest.tsv
checksums.sha256
release_summary.json
```

Its retained scientific data are installed under:

```text
MDLM_pipeline/
figure_generation/compiled/final/
cofolding_study/core_files/
```

The repository-local manifest records:

- repository-resident workbooks and compact example inputs;
- the P29972 and P07740 files used by the documented example reruns;
- the compressed starting-MSA archive used by the Figure 7 benchmark.

For the full release layout, overlay procedure, archive extraction, and verification commands, see `docs/data.md`.
