
"""Validate source observations, proposed designs and image comparisons.

Visual observations are model assessments, not verified material facts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

from .audit import digest_object, now, write_json
from .contracts import validate_identity
from .evidence import require


Attribute = Literal[
    "neckline", "sleeves", "length", "hem",
    "surface", "leg", "waist", "detail",
]

Value = Literal[
    "high_neck", "crew_neck", "v_neck", "square_neck", "boat_neck",
    "straight_sleeves", "full_sleeves", "short_sleeves",
    "cropped", "regular", "long", "ankle_length", "full_length",
    "straight_hem", "curved_hem", "plain", "ribbed", "cable",
    "tapered_leg", "straight_leg", "wide_leg", "flared_leg",
    "high_waist", "mid_waist", "no_added_detail", "front_pintucks",
    "patch_pockets", "cargo_pocket", "unclear",
]

Colour = Literal[
    "black", "light_beige", "beige", "pink",
    "terracotta", "white", "blue", "unclear",
]

OPTIONS = {
    "neckline": [
        "high_neck", "crew_neck", "v_neck", "square_neck", "boat_neck"
    ],
    "sleeves": ["straight_sleeves", "full_sleeves", "short_sleeves"],
    "length": ["cropped", "regular", "long", "ankle_length", "full_length"],
    "hem": ["straight_hem", "curved_hem"],
    "surface": ["plain", "ribbed", "cable"],
    "leg": ["tapered_leg", "straight_leg", "wide_leg", "flared_leg"],
    "waist": ["high_waist", "mid_waist"],
    "detail": [
        "no_added_detail", "front_pintucks", "patch_pockets", "cargo_pocket"
    ],
}

KEYS = {
    "Sweater": ["neckline", "sleeves", "length", "hem", "surface"],
    "Trousers": ["leg", "waist", "length", "detail"],
}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Feature(Strict):
    attribute: Attribute
    value: Value


class GarmentObservation(Strict):
    winner_rank: int
    article_id: str
    category: Literal["Sweater", "Trousers", "unclear"]
    colour: Colour
    full_garment_visible: bool
    features: list[Feature]


class Observations(Strict):
    garments: list[GarmentObservation]
    exactly_three_garments: bool


class Change(Strict):
    attribute: Attribute
    before: Value
    after: Value


class ProposedConcept(Strict):
    winner_rank: int
    article_id: str
    colour: Colour
    changes: list[Change]


class Proposal(Strict):
    concepts: list[ProposedConcept]


def label(value):
    return str(value).replace("_", " ")


def attributes(observation):
    result = {
        feature.attribute: feature.value
        for feature in observation.features
    }

    require(
        len(result) == len(observation.features),
        "Duplicate visual feature.",
    )

    for key, value in result.items():
        require(
            value in OPTIONS[key] or value == "unclear",
            f"Invalid {key}: {value}.",
        )

    return result


def validate_observations(observations, winners, *, source=False):
    validate_identity(observations.garments, winners)

    for observation, winner in zip(observations.garments, winners):
        values = attributes(observation)
        expected_category = winner["product_type"]

        require(
            set(values) == set(KEYS[expected_category]),
            "Missing or unexpected visual attributes.",
        )

        if source:
            require(
                observation.category == expected_category
                and observation.full_garment_visible,
                "Source category or full view is uncertain; "
                "inspect the photographs before continuing.",
            )
            require(
                observations.exactly_three_garments,
                "Expected exactly three source garments.",
            )

        allowed_lengths = (
            ["cropped", "regular", "long", "unclear"]
            if expected_category == "Sweater"
            else ["ankle_length", "full_length", "unclear"]
        )

        require(
            values["length"] in allowed_lengths,
            f"Invalid {expected_category.lower()} length.",
        )


def validate_proposal(plan, source, winners, colours):
    validate_observations(source, winners, source=True)
    validate_identity(plan.concepts, winners)

    require(
        set(colours) == {str(w["winner_rank"]) for w in winners},
        "Specify one colour per source rank.",
    )

    for concept, observation in zip(plan.concepts, source.garments):
        before = attributes(observation)

        require(
            concept.colour in get_args(Colour)
            and concept.colour != "unclear"
            and (
                colours[str(concept.winner_rank)] == "agent_choice"
                or concept.colour == colours[str(concept.winner_rank)]
            ),
            "The concept colour is invalid or conflicts with a fixed colour.",
        )
        require(
            len(concept.changes) == 2,
            "Specify exactly two clear visual changes per garment.",
        )

        changed_attributes = {
            change.attribute for change in concept.changes
        }

        require(
            len(changed_attributes) == 2,
            "Duplicate change dimension.",
        )

        for change in concept.changes:
            require(
                change.attribute in before,
                "Change uses a feature outside the garment category.",
            )
            require(
                change.before == before[change.attribute]
                and change.before != "unclear",
                "Design invents or changes an uncertain source observation.",
            )
            require(
                change.after in OPTIONS[change.attribute]
                and change.after != change.before,
                "The proposed change must be distinct and visible.",
            )

            if change.attribute == "length":
                allowed = (
                    ["cropped", "regular", "long"]
                    if observation.category == "Sweater"
                    else ["ankle_length", "full_length"]
                )
                require(
                    change.after in allowed,
                    "Proposed length belongs to another garment category.",
                )

        known_retained = [
            key
            for key, value in before.items()
            if value != "unclear" and key not in changed_attributes
        ]

        require(
            len(known_retained) >= 2,
            "Keep at least two known visual source attributes.",
        )


def target_attributes(concept, source):
    result = attributes(source).copy()

    for change in concept.changes:
        result[change.attribute] = change.after

    return result


def concept_name(concept, source):
    return (
        f"{label(concept.colour).title()} "
        f"{source.category.lower()} concept {concept.winner_rank}"
    )


def compiled_prompt(plan, source):
    panels = []

    for concept, observation in zip(plan.concepts, source.garments):
        target = target_attributes(concept, observation)

        features = "; ".join(
            f"{label(key)}: {label(value)}"
            for key, value in target.items()
            if value != "unclear"
        )

        changes = "; ".join(
            f"{label(change.attribute)} from "
            f"{label(change.before)} to {label(change.after)}"
            for change in concept.changes
        )

        panels.append(
            f"Panel {concept.winner_rank}: {observation.category}, "
            f"reference photo {concept.winner_rank}, "
            f"source article {concept.article_id}. "
            f"Final colour MUST be {label(concept.colour)}. "
            f"Final visible design: {features}. "
            f"Exactly these two structural/visual changes: {changes}. "
            "Preserve all other visible source details; if a detail is "
            "not specified, follow the source photograph. "
            "The garment is a new concept inspired by that source."
        )

    return (
        "Create ONE product concept board, EXACTLY THREE garments in "
        "equally sized left-to-right panels. References 1,2,3 correspond "
        "to panels 1,2,3. Show each entire garment in front view, all "
        "sleeves and hems inside its panel, clear margins, consistent "
        "neutral lighting and off-white background. No people, "
        "accessories, text, logos or watermarks. Render literal visible "
        "differences, not a mood-board interpretation. Do not add extra "
        "pockets, decorations, recolouring or cropped hems unless "
        "explicitly specified. Physical fiber content, comfort, "
        "sustainability and sales cannot be established by this image."
        "\n\n" + "\n\n".join(panels)
    )


def approval_payload(plan, source, packet, settings):
    return {
        "proposal": plan.model_dump(),
        "source_observations": source.model_dump(),
        "references": [
            winner["source_image_sha256"]
            for winner in packet["winners"]
        ],
        "settings": settings,
        "image_prompt": compiled_prompt(plan, source),
    }


def save_approval(
    output, plan, source, packet, settings, reviewed_hash
):
    payload = approval_payload(plan, source, packet, settings)
    expected = digest_object(payload)

    require(
        reviewed_hash == expected,
        "Plan changed since your preview. "
        "Review the new preview before approving.",
    )

    receipt = {
        "approved_at": now(),
        "approved_payload_sha256": expected,
        "meaning": (
            "User approved these source observations, colours, "
            "exact two changes, and image request settings."
        ),
    }

    write_json(Path(output) / "design_approval.json", receipt)
    return receipt


def require_approval(output, plan, source, packet, settings):
    path = Path(output) / "design_approval.json"

    require(
        path.is_file(),
        "Review and approve the design preview before a paid image call.",
    )

    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = digest_object(
        approval_payload(plan, source, packet, settings)
    )

    require(
        receipt["approved_payload_sha256"] == expected,
        "Design, source evidence, image prompt, "
        "or settings changed after approval.",
    )

    return receipt


def compare_board(plan, source, board, winners):
    validate_observations(board, winners)
    rows = []

    for concept, original, seen, winner in zip(
        plan.concepts, source.garments, board.garments, winners
    ):
        wanted = target_attributes(concept, original)
        observed = attributes(seen)
        changed = {change.attribute for change in concept.changes}

        targets = [
            ("category", winner["product_type"]),
            ("colour", concept.colour),
            *wanted.items(),
        ]

        for key, target in targets:
            if key == "category":
                actual = seen.category
            elif key == "colour":
                actual = seen.colour
            else:
                actual = observed[key]

            if target == "unclear":
                status = "not_assessed"
            elif actual == "unclear":
                status = "unclear"
            elif target == actual:
                status = "match"
            else:
                status = "mismatch"

            rows.append({
                "winner_rank": concept.winner_rank,
                "article_id": concept.article_id,
                "attribute": key,
                "proposed": target,
                "observed": actual,
                "status": status,
                "role": (
                    "change"
                    if key in changed
                    else "retained_or_user_colour"
                ),
            })

    full_views = all(
        observation.full_garment_visible
        for observation in board.garments
    )

    passed = (
        board.exactly_three_garments
        and full_views
        and all(
            row["status"] in ["match", "not_assessed"]
            for row in rows
        )
    )

    return {
        "automated_check": "passed" if passed else "needs_review",
        "human_approval": "pending",
        "exactly_three_garments": board.exactly_three_garments,
        "full_garments_visible": full_views,
        "checks": rows,
        "limitation": (
            "Independent model observations can be wrong. "
            "Inspect the image yourself; no automatic regeneration."
        ),
    }
