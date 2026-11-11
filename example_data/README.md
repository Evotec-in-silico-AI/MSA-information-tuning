# Deterministic alignment fixture

This directory contains a compact reference FASTA and parental A3M used by:

```bash
python scripts/run_workflow.py test
```

The fixture is synthetic and is not a manuscript result. The smoke test verifies:

- one 33-residue reference sequence;
- alignment depth of six sequences;
- alignment width of 33 columns;
- one gap character;
- three non-gap substitutions relative to the reference across the five homologues;
- non-gap occupancy of 197/198.

These fixed expectations exercise repository parsing and basic alignment bookkeeping without requiring external data, model assets, or a GPU.
