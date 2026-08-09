"""Register this domain into tau2's global registry.

tau2 has no plugin/entry-point mechanism for third-party domains — the bundled
domains are registered inline in `tau2/registry.py`. Rather than fork the
framework, this module imports the same global `registry` singleton and
registers the domain into it. Any tau2 entrypoint (`run_domain`, the
evaluator, `tau2 view`) then treats `saas_support` exactly like a built-in
domain, as long as this module is imported first.
"""

from tau2.registry import registry

from saas_support.environment import get_environment, get_tasks, get_tasks_split
from saas_support.utils import DOMAIN_NAME


def register_domain() -> None:
    """Idempotently register the saas_support domain and its task set."""
    if DOMAIN_NAME not in registry.get_domains():
        registry.register_domain(get_environment, DOMAIN_NAME)
    if DOMAIN_NAME not in registry.get_task_sets():
        registry.register_tasks(
            get_tasks, DOMAIN_NAME, get_task_splits=get_tasks_split
        )
