
"""Check delegated colour selection and approval consistency."""

import pytest

from test_workflow_v2 import fixtures
from merchmix.audit import digest_object
from merchmix.design_v2 import (
    validate_proposal,
    approval_payload,
    save_approval,
    require_approval,
)


def agent_plan():
    packet, source, plan, settings, _ = fixtures()
    settings["colours"] = {
        "1": "agent_choice",
        "2": "agent_choice",
        "3": "agent_choice",
    }

    # Deliberately different colours exercise the delegated-choice path.
    for concept, colour in zip(plan.concepts, ["blue", "pink", "white"]):
        concept.colour = colour

    return packet, source, plan, settings


def test_agent_can_choose_supported_colours():
    packet, source, plan, settings = agent_plan()
    validate_proposal(plan, source, packet["winners"], settings["colours"])


def test_agent_cannot_return_unclear_or_unsupported_colour():
    packet, source, plan, settings = agent_plan()

    for invalid in ["unclear", "magenta"]:
        changed = plan.model_copy(deep=True)
        changed.concepts[0].colour = invalid

        with pytest.raises(ValueError):
            validate_proposal(
                changed, source, packet["winners"], settings["colours"]
            )


def test_agent_colour_cannot_change_after_approval(tmp_path):
    packet, source, plan, settings = agent_plan()

    reviewed_hash = digest_object(
        approval_payload(plan, source, packet, settings)
    )

    save_approval(
        tmp_path, plan, source, packet, settings, reviewed_hash
    )
    require_approval(tmp_path, plan, source, packet, settings)

    plan.concepts[0].colour = "black"

    with pytest.raises(ValueError):
        require_approval(tmp_path, plan, source, packet, settings)
