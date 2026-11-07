"""Compatibility boundary for the result-bound EAGER CLI.

The reported EAGER CLI imports optional baseline entry points at module load
time. Baseline implementations are outside this public EAGER package. These
stubs preserve the result-bound EAGER runtime while making unsupported
baseline commands fail explicitly.
"""


class BaselineRegistryError(RuntimeError):
    """Raised when an omitted baseline command is requested."""


def _not_distributed(*args, **kwargs):  # type: ignore[no-untyped-def]
    del args, kwargs
    raise BaselineRegistryError(
        "Baseline adapters are outside this public EAGER package."
    )


build_shared_stack_executor = _not_distributed
load_baseline_registry = _not_distributed
