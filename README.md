# EAGER: Anchor-Driven Evidence Construction for Multi-Hop RAG

EAGER has been accepted as a **Findings paper at WSDM 2027** (Paper ID
1877). The conference will be held in Hong Kong on February 15--19, 2027;
Findings papers are presented as posters. The camera-ready deadline is
December 1, 2026, and the author-registration deadline is December 17, 2026
(AoE).

This repository is the public implementation and reproducibility package for
the paper, *EAGER: Anchor-Driven Evidence Construction for Multi-Hop RAG*.
It contains the EAGER retrieval kernel, controller prompts, frozen experiment
wrappers, compact result tables, provenance checks, and offline tests. It does
not contain model weights, benchmark downloads, hosted-model response caches,
private checkpoints, or API credentials.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/eager/` | Evidence activation, hierarchical selection, state transitions, controllers, data, evaluation, and runtime code |
| `prompts/` | Retrieval-question, subquestion-chain, intermediate-answer, reader, and evaluator prompts |
| `configs/` | Reproducible model and retrieval settings; local credentials stay outside version control |
| `experiments_sandbox/` | The six main wrappers, component ablations, and the matched MuSiQue-RQ state-anchor intervention |
| `results/` | Compact ConAcc/F1 tables, provenance-bound diagnostics, and million-unit scaling results |
| `provenance/` | Hashes and executed-source snapshots that bind reported results to code |
| `splits/` | Development and formal question-ID lists |
| `tests/` | Offline unit and contract tests with small fixtures |
| `analysis/verify.py` | Offline inventory, split, provenance, and result-table verifier |

The current paper reports six fixed controller--benchmark configurations with
no question-level routing at test time. The main quality metrics are ConAcc
and alias-aware token F1. Some provenance-bound CSV files retain GPTAcc fields
from earlier audit runs; those fields are kept to preserve the executed
artifact history and are not the headline metrics for the accepted paper.

## Installation

Python 3.10--3.12 is supported.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[core,dev]"
python -m spacy download en_core_web_sm
```

Model revisions and platform-specific dependency locks are recorded under
`requirements/`. Model weights and official benchmark files must be obtained
from their original providers.

## Verification and tests

The release verifier is fully offline:

```bash
python analysis/verify.py
```

It checks the export inventory, six main wrappers, component interventions,
formal split counts, current result tables, and terminal-state audits.

Run the fixture-backed test suite with:

```bash
python -m pytest -q
```

Tests that require external benchmark files or local model snapshots skip when
those dependencies are unavailable.

## Smoke run

The fixture workflow does not call a hosted model:

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

## Full runs

After downloading the registered datasets and building the MiniLM and surface
indexes, the six main wrappers reproduce EAGER-RQ and EAGER-SC on HotpotQA,
2WikiMultiHopQA, and MuSiQue. Wrapper help is available with:

```bash
python experiments_sandbox/20260728_027_hotpotqa_rq_v1_full1000/run.py --help
python experiments_sandbox/20260728_032_musique_sc_v1_full1000/run.py --help
```

The component-ablation wrappers use the byte-exact source snapshots under
`provenance/eager_v1_execution_snapshot/`. The matched MuSiQue-RQ state-anchor
comparison is documented under
`results/eager_v1_component_ablation_holm21_20260818/`.

Hosted generation reads credentials from `OPENAI_API_KEY` or an ignored local
configuration copied from `configs/llm_config.example.json`. No credential is
stored in this repository. The checked-in defaults use an OpenAI-compatible
endpoint suitable for new runs; provenance records retain the endpoint
metadata of the recorded experiments.

## Results

The compact result tables under `results/` map to the paper's main results,
component interventions, structural analysis, and million-unit scaling study.
Each result directory contains its own manifest or protocol note where
applicable. The export inventory in `EXPORT_MANIFEST.json` records the byte
size and SHA-256 digest of every tracked release file.

## Citation

Please cite the WSDM 2027 Findings paper once the final bibliographic record
is available. The title and acceptance information above identify the work
unambiguously in the meantime.

## Release boundary

This public repository contains source code, prompts, compact tables, split
IDs, provenance metadata, and offline fixtures. Raw hosted-model traces,
model weights, benchmark downloads, private run checkpoints, review material,
and machine-local paths are excluded.
