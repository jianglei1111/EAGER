# Result-bound EAGER V1 execution snapshot

This directory preserves the code bytes bound by the canonical preflights for
the six reported EAGER main runs (2026-07-28) and the 20 component-ablation
runs (2026-07-29 through 2026-07-30).

The main runs were executed from a dirty checkout whose shared retrieval code
was subsequently committed and then hardened. The three files under
`root_overlay/` are the exact preflight-bound versions and replace the newer
working-tree versions during anonymous export. No prediction is recomputed or
changed by this export operation.

`checkpoint_export_manifest.json` is the complete SHA-256 inventory exported
immediately before the component ablations. The offline verifier uses it to
check all 83 execution-relevant files under `src/eager/`, `configs/wsdm/`, and
the six active prompts. Together with the wrapper-specific hashes below, this
binds the package to the actual July execution tree rather than to a later
development checkout.

Five orchestration wrappers were standardized after the main runs and before
the component ablations. Both byte-exact sets are therefore retained:

- `main_wrappers_executed/` contains the wrappers recorded by the six main-run
  preflights.
- `ablation_base_wrappers/` contains the standardized wrappers recorded by the
  20 ablation preflights.

This wrapper-only transition did not change the shared retrieval kernel,
controllers, prompts, prepared question IDs, indexes, or effective retrieval
configuration. The anonymous package uses the executed main wrappers at its
root and publishes the ablation-bound set under `provenance/`. Run
`scripts/select_eager_v1_protocol_snapshot.py --mode ablation --materialize`
before reproducing the dated component-ablation entries.

`configs/llm_config.local.json` is intentionally excluded because it contains
a private credential-bearing runtime override. Its SHA-256 and the non-secret
endpoint/model metadata are retained in result provenance; users create their
own ignored local configuration from the supplied example.
