"""MCP client used by the Merchmix agent workflow."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timedelta
import json
from pathlib import Path
import sys

from .audit import digest_object
from .evidence import require


@asynccontextmanager
async def evidence_session(bundle: Path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "merchmix.mcp_server",
            "--bundle",
            str(bundle.resolve()),
        ],
    )

    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(
            reader,
            writer,
            read_timeout_seconds=timedelta(seconds=180),
        ) as session:
            await session.initialize()
            yield session


async def call_json(
    session,
    name: str,
    arguments: dict | None = None,
    audit=None,
) -> dict:
    arguments = arguments or {}

    if audit:
        audit.event(
            "mcp_tool_started",
            tool=name,
            arguments=arguments,
        )

    response = await session.call_tool(name, arguments)

    require(
        not response.isError,
        f"MCP tool {name} failed: "
        + " ".join(
            getattr(block, "text", "")
            for block in response.content
        ),
    )

    text_blocks = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]

    require(
        bool(text_blocks),
        f"MCP tool {name} returned no JSON text.",
    )

    result = json.loads("\n".join(text_blocks))

    require(
        isinstance(result, dict),
        f"Unexpected MCP result type for {name}.",
    )

    if audit:
        summary = {
            key: value
            for key, value in result.items()
            if key != "image_base64"
        }

        audit.event(
            "mcp_tool_completed",
            tool=name,
            arguments=arguments,
            result_sha256=digest_object(result),
            result_without_image_bytes=summary,
        )

    return result


async def collect_evidence(
    bundle: Path,
    audit=None,
    replay: bool = True,
) -> dict:
    async with evidence_session(bundle) as session:
        packet = await call_json(
            session,
            "get_winning_styles",
            audit=audit,
        )

        packet["evaluation"] = await call_json(
            session,
            "get_evaluation_metrics",
            audit=audit,
        )

        packet["references"] = []

        for winner in packet["winners"]:
            reference = await call_json(
                session,
                "get_reference_product",
                {"winner_rank": winner["winner_rank"]},
                audit=audit,
            )

            require(
                reference["article_id"] == winner["article_id"],
                "MCP source article changed.",
            )
            require(
                reference["image_sha256"]
                == winner["source_image_sha256"],
                "MCP image identity changed.",
            )

            packet["references"].append(reference)

        if replay:
            packet["model_replay"] = await call_json(
                session,
                "verify_selected_model_predictions",
                audit=audit,
            )
        else:
            packet["model_replay"] = {"status": "not_run"}

        return packet
