"""Read-only MCP tools for Merchmix forecasting evidence."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from .evidence import DEFAULT_BUNDLE, EvidenceStore


mcp = FastMCP("merchmix-forecast-evidence")
BUNDLE = DEFAULT_BUNDLE


@lru_cache(maxsize=1)
def store() -> EvidenceStore:
    return EvidenceStore(BUNDLE)


@mcp.tool()
def get_winning_styles() -> dict:
    """Get the three selected styles, forecasts, scores, and limitations."""
    return store().winners_packet()


@mcp.tool()
def get_evaluation_metrics() -> dict:
    """Get validation/test metrics and the separate ETS sample metrics."""
    return store().metrics_packet()


@mcp.tool()
def get_reference_product(winner_rank: int) -> dict:
    """Get a selected source article and its actual JPEG, by rank 1–3."""
    return store().reference_packet(winner_rank)


@mcp.tool()
def verify_selected_model_predictions() -> dict:
    """Replay the saved CatBoost model for the three selected styles."""
    return store().replay_model(all_rows=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
    )

    BUNDLE = parser.parse_args().bundle.resolve()
    mcp.run(transport="stdio")
