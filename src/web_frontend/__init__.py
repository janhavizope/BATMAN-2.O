"""
BATMAN Frontend Modules
-----------------------
Dashboard UI pages and shared components for the BATMAN application.
All pages use placeholder/development data until the backend is integrated.
"""

from src.web_frontend.components import render_metric_card, render_header, render_footer, render_dev_badge

_PAGE_MODULES = {
    "render_overview": "overview",
    "render_alerts": "alerts",
    "render_entity_investigation": "entity_investigation",
    "render_transaction_analysis": "transaction_analysis",
    "render_network_graph": "network_graph",
}

__all__ = [
    "render_metric_card",
    "render_header",
    "render_footer",
    "render_dev_badge",
    *_PAGE_MODULES,
]


def __getattr__(name: str):
    module_name = _PAGE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    render_page = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = render_page
    return render_page
