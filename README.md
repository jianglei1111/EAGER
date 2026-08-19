# EAGER Anonymous Artifact

This package contains the EAGER implementation used for the paper's RQ and SC
experiments. It is intentionally EAGER-only: third-party repositories,
baseline adapters, manuscript source, historical prototypes, and private run
checkpoints are not redistributed. Compact comparison tables remain available
under `results/` so the reported analyses can be inspected directly.

## Contents

| Path | Purpose |
| --- | --- |
| `src/eager/` | Evidence activation, hierarchical selection, state transition, controllers, data, evaluation, and runtime code |
| `prompts/` | RQ, SC, intermediate-answer, reader, and optional evaluator prompts |
| `configs/` | Result-bound EAGER configuration and model registry |
| `experiments_sandbox/` | Six main wrappers, 20 historical one-switch wrappers, and one matched MuSiQue-RQ state-anchor intervention wrapper |
| `provenance/` | Byte hashes and the executed main/ablation wrapper snapshots |
| `results/` | Current ConAcc/F1 tables plus compact run-bound diagnostics |
| `tests/` | EAGER unit and contract tests with small offline fixtures |
| `analysis/verify.py` | Offline inventory and result-table verifier |

`src/eager_baselines/__init__.py` is only a small compatibility boundary. The
result-bound EAGER CLI imported optional baseline entry points at module load
time; the compatibility module makes those omitted commands fail explicitly
without distributing baseline implementations.

## Install

Python 3.10--3.12 is supported.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[core,dev]"
python -m spacy download en_core_web_sm
```

Model revisions and platform-specific dependency locks are recorded under
`requirements/`. Model weights and official benchmark files are downloaded
from their original providers and are not included in the archive.

## Offline Checks

The release verifier needs no model, dataset, API key, or network access:

```bash
python analysis/verify.py
```

It checks the complete export inventory, the six result-bound main wrappers,
the 20 historical ablation wrappers, the matched MuSiQue-RQ state-anchor
intervention, formal split ID counts, and all directly readable result tables.

The small fixture suite exercises indexing, retrieval, controllers, state
retention, evaluation, and configuration resolution:

```bash
python -m pytest -q
```

Tests that require external benchmark files or model snapshots skip when those
dependencies are unavailable.

## EAGER Smoke Run

The fixture workflow does not use a hosted model:

```bash
python -m eager.cli data index \
  --corpus tests/fixtures/indexing/corpus.jsonl \
  --output tmp/quickstart-index \
  --dimensions 32

python -m eager.cli retrieve \
  --dataset 2wiki \
  --question "Where is the company founded by Mira Sol headquartered?" \
  --corpus tests/fixtures/indexing/corpus.jsonl \
  --index tmp/quickstart-index \
  --mode retrieval-question
```

## Full EAGER Runs

The package exports the exact EAGER source bytes bound by the reported run
preflights. After downloading the registered datasets and building the MiniLM
and surface indexes, the six main wrappers reproduce EAGER-RQ and EAGER-SC on
the three benchmarks. For example:

```bash
python experiments_sandbox/20260728_027_hotpotqa_rq_v1_full1000/run.py --help
python experiments_sandbox/20260728_032_musique_sc_v1_full1000/run.py --help
```

Five wrapper files were standardized between the main runs and the component
ablations without changing the retrieval kernel, prompts, model registry,
question IDs, indexes, or effective settings. Both byte-exact wrapper sets are
under `provenance/eager_v1_execution_snapshot/`. Select the ablation-bound set
before executing wrappers 033--052:

```bash
python scripts/select_eager_v1_protocol_snapshot.py \
  --mode ablation --materialize
```

Use `--mode main` to restore the main-run wrappers.

The 21st component intervention is a matched MuSiQue-RQ state-anchor toggle.
Its compact comparison is in
`results/eager_v1_component_ablation_holm21_20260818/`; the wrapper and its
isolated source snapshot are under
`experiments_sandbox/20260817_068_musique_rq_state_anchor_on_v1_openlux_matched1000/`
and `provenance/state_anchor_intervention/`. The package intentionally omits
the 1,000-question raw predictions and retrieval traces (which are large and
contain hosted-model outputs).

API-backed generation reads credentials from `OPENAI_API_KEY` or an ignored
`configs/llm_config.local.json` created from
`configs/llm_config.example.json`. The reported manifests preserve the
temperature, requested model alias, prompt hashes, and endpoint metadata. New
runs through another compatible endpoint receive a new manifest.

## Result Tables

Start with `results/p9_conacc_alias_f1_v1/README.md`. It maps the current paper
tables to compact CSV files for complete-system results, the six matched
EAGER-versus-Standalone comparisons, 21 component interventions, dependency
structure, the Hard-200 comparison, and bounded-scaling inputs. ConAcc uses the
primary canonical answer; token F1 takes the maximum over the canonical answer
and benchmark-provided aliases. These are compact analysis products rather
than hosted-model response caches or multi-gigabyte retrieval checkpoints.

The complete 1,000-question scale comparison in Table 4 is under
`results/million_matched_rq_20260819/`. It reports low- and million-scale
Hard-200 EAGER-RQ and Standalone-RQ metrics, paired direct and scale-interaction
tests, and an audit of the shared corpora, questions, prompts, and budgets.

## Release Boundary

The archive has no Git history, credentials, machine-local absolute paths,
model weights, raw hosted traces, third-party code, or manuscript source. Its
`EXPORT_MANIFEST.json` records the byte size and SHA-256 digest of every file.
