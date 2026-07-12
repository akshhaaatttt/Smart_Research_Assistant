"""Shared visualization styling so every chart pulls its template and palette from one place."""

from __future__ import annotations

from typing import Optional

from src.utils.config import AppConfig

# Colour-blind friendly, prints well.
PALETTE = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706",
    "#7c3aed", "#0891b2", "#db2777", "#65a30d",
]


def get_template(config: Optional[AppConfig] = None) -> str:
    config = config or AppConfig.load()
    return config.get("visualization.theme", "plotly_white")


def apply_layout(fig, title: str, config: Optional[AppConfig] = None, **kwargs):
    """Apply the shared template, title, and sensible margins to a figure."""
    fig.update_layout(
        template=get_template(config),
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16)),
        margin=dict(l=60, r=30, t=60, b=50),
        colorway=PALETTE,
        **kwargs,
    )
    return fig
