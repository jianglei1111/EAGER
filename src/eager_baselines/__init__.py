"""Compatibility boundary for the result-bound EAGER CLI.

The reported EAGER CLI imported optional baseline entry points at module load
time. Baseline implementations are intentionally not distributed in the
EAGER-only anonymous artifact. These stubs preserve the exact EAGER runtime
source while making unsupported baseline commands fail explicitly.
"""


class BaselineRegistryError(RuntimeError):
    """Raised when an omitted baseline command is requested."""


def _not_distributed(*args, **kwargs):  # type: ignore[no-untyped-def]
    del args, kwargs
    raise BaselineRegistryError(
        "Baseline adapters are outside the EAGER-only anonymous artifact."
    )


build_shared_stack_executor = _not_distributed
load_baseline_registry = _not_distributed
