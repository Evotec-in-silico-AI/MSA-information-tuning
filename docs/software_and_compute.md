# Software and compute

## Analysis environment

`environments/analysis_environment.yml` specifies Python 3.10 and the direct versions reported in the manuscript:

- TensorFlow 2.8.0
- PyTorch 2.8.0
- NumPy 1.23.5
- SciPy 1.10.1
- pandas 2.3.2
- scikit-learn 1.2.1
- ProDy 2.4.1
- scikit-dimension 0.3.4
- factor-analyzer 0.5.1
- NetworkX 2.5.1
- Biopython 1.79
- seaborn 0.13.2
- Matplotlib 3.9.4
- PyArrow 10.0.1
- OpenPyXL 3.1.5

Python 3.10 is used because it is compatible with the pinned TensorFlow and PyTorch releases. Figure generation from processed companion data is CPU-compatible. GPU acceleration for reconstruction workflows depends on the installed framework build and local CUDA stack.

## Chai-1 environment

`environments/chai1_environment.yml` specifies Python 3.12.10 and:

- chai_lab 0.6.1
- TensorFlow 2.19.0
- PyTorch 2.6.0
- NumPy 1.26.4
- SciPy 1.15.3
- pandas 2.2.3
- scikit-learn 1.6.1
- NetworkX 3.4.2
- Biopython 1.85
- Biotite 1.4.0
- Gemmi 0.6.7
- Matplotlib 3.10.3
- PyArrow 20.0.0
- OpenPyXL 3.1.5

Chai-1 0.6.1 is a Linux GPU workflow. The public Chai-1 documentation requires Python 3.10 or later and a CUDA-capable GPU with bfloat16 support. Model assets are obtained through Chai-1 and are not redistributed here.

## TM-score

The Figure 7 structural comparison uses the TM-score executable distributed by the Zhang Group:

https://zhanggroup.org/TM-score/

The locked comparison script retains the historical local default used during the study. Public runs must provide the executable explicitly:

```bash
python cofolding_study/core_files/compare_chai1_to_experimental.py   --mode jackhmmer   --tmscore-exec /path/to/TMscore
```

The same path may be supplied through the `TMSCORE_EXEC` environment variable.


## External software provenance

The study utilizes BLASTP 2.13.0+, MUSCLE Super5 5.1, and `chai_lab` 0.6.1. 

## Historical commands

### One MDLM protein

```bash
cd MDLM_pipeline
python main.py --name <accession> --blast_database nr
```

The manuscript settings use 4,000 maximum training steps and seed 42. Exact P29972 and P07740 commands are emitted by:

```bash
python scripts/run_workflow.py examples --check-only
```

### Cohort assembly

```bash
cd MDLM_pipeline
python cohort/cohort_builder_cpu.py --repo_root . --cohort_xlsx cohort/cohort.xlsx --workers 128 --max_in_flight 256
python cohort/cohort_builder_gpu.py --repo_root . --cohort_xlsx cohort/cohort.xlsx --device /GPU:0
python cohort/cohort_builder_finalize.py --repo_root . --cohort_xlsx cohort/cohort.xlsx --archive_intermediates
python analysis/build_cohort_dataset.py --runs_root runs --out_xlsx cohort/cohort.xlsx
```

The companion release distributes `figure_generation/compiled/final/`, which is the processed input consumed by Figures 5 and 6. Historical `compiled/archive/` intermediates are not required for figure regeneration and are not distributed.

### Folding benchmark

```bash
cd cofolding_study/core_files
python tune_msas_for_folding.py --n-msas 10 --mode jackhmmer --strip-dots --max-row-gap-frac 0.25 --cap-jackhmmer 20000 --downsample-method simhash --downsample-seed 42 --simhash-buckets-bits 12 --simhash-k 3 --log-level INFO
python run_chai1.py --mode jackhmmer
python compare_chai1_to_experimental.py --mode jackhmmer --tmscore-exec /path/to/TMscore
```

## Historical resource requests

- Chai-1 inference: one GPU, four CPU cores.
- MSA tuning: 12 CPU cores, no GPU. Wall time for most runs was 3-4 days, larger proteins could take longer if many information targets were available.
- Cohort CPU compilation: up to 128 workers.
- Cohort GPU compilation: one GPU.
