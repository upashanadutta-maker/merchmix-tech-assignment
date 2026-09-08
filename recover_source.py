
"""Resume a failed source-observation stage with explicit review provenance."""

import json
import shutil
from pathlib import Path

from merchmix.audit import digest_object, write_json
from merchmix.evidence import EvidenceStore, require, sha256
from merchmix.contracts import ForecastAssessment, ForecastAnalysis
from merchmix.forecast_checks import (
    accept_forecast_assessment,
    validate_forecast_analysis,
)
from merchmix.design_v2 import Observations, validate_observations
from merchmix.workflow_v2 import new_audit


def recover(previous, destination, settings):
    previous = Path(previous).resolve()
    destination = Path(destination).resolve()

    manifest = json.loads(
        (previous / "run_manifest.json").read_text(encoding="utf-8")
    )

    require(
        manifest["input_hash"] == digest_object(manifest["inputs"]),
        "Original manifest is inconsistent.",
    )
    require(
        not any(
            stage in manifest["stages"]
            for stage in ("agent_proposal", "image_receipt")
        ),
        "This recovery applies only to a failure before design.",
    )

    def read_stage(name):
        saved = manifest["stages"][name]
        path = (previous / saved["file"]).resolve()

        require(
            path.is_relative_to(previous)
            and sha256(path) == saved["sha256"],
            f"Original {name} changed.",
        )
        return json.loads(path.read_text(encoding="utf-8"))

    packet = read_stage("evidence")
    store = EvidenceStore()

    for key, value in store.winners_packet().items():
        require(packet[key] == value, "Forecast evidence changed.")

    require(
        packet["evaluation"] == store.metrics_packet(),
        "Evaluation evidence changed.",
    )

    assessment = ForecastAssessment.model_validate(
        read_stage("forecast_agent_assessment")
    )
    analysis = ForecastAnalysis.model_validate(
        read_stage("forecast_analysis")
    )

    accept_forecast_assessment(assessment, packet)
    validate_forecast_analysis(analysis, packet)

    require(
        packet["model_replay"]["status"] == "passed",
        "Original model replay did not pass.",
    )

    raw = read_stage("source_observations")
    reviewed = Observations.model_validate(raw)

    corrections = [
        (
            "0673677002", "sleeves", "full_sleeves", "straight_sleeves",
            "Photo assessment: sleeves appear straight, "
            "without deliberate puff or balloon volume.",
        ),
        (
            "0673677002", "surface", "ribbed", "plain",
            "Photo assessment: the main body appears plain; "
            "ribbed trims are a separate detail.",
        ),
        (
            "0865799006", "leg", "straight_leg", "tapered_leg",
            "Catalogue description: wide, tapered legs.",
        ),
        (
            "0865799006", "waist", "mid_waist", "high_waist",
            "Catalogue description: high waist.",
        ),
        (
            "0865799006", "length", "regular", "ankle_length",
            "Catalogue description: ankle-length trousers.",
        ),
    ]

    description = next(
        winner["detail_desc"]
        for winner in packet["winners"]
        if winner["article_id"] == "0865799006"
    )

    require(
        all(
            term in description.lower()
            for term in ("ankle-length", "high waist", "tapered")
        ),
        "Catalogue does not support the proposed trouser corrections.",
    )

    for article, attribute, before, after, basis in corrections:
        garment = next(
            item for item in reviewed.garments
            if item.article_id == article
        )
        feature = next(
            item for item in garment.features
            if item.attribute == attribute
        )

        require(
            feature.value == before,
            f"Unexpected original value for {article}/{attribute}.",
        )
        feature.value = after

    validate_observations(reviewed, packet["winners"], source=True)

    for reference in packet["references"]:
        path = (previous / reference["local_image"]).resolve()
        require(
            path.is_relative_to(previous)
            and sha256(path) == reference["image_sha256"],
            "Original source photo changed.",
        )

    require(
        not destination.exists(),
        "Recovery folder already exists; inspect it before repeating.",
    )

    audit = new_audit(destination, settings)
    current = audit.state["inputs"]
    original = manifest["inputs"]

    for key in ("settings", "artifacts", "skill_sha256", "versions"):
        require(
            current[key] == original[key],
            f"Cannot reuse evidence after {key} changed.",
        )

    changed_code = {
        name
        for name in set(current["code"]) | set(original["code"])
        if current["code"].get(name) != original["code"].get(name)
    }

    require(
        changed_code <= {
            "workflow_v2.py", "report_v2.py", "observation_schema.py"
        },
        "An unexpected code change needs review before recovery.",
    )

    for reference in packet["references"]:
        target = (destination / reference["local_image"]).resolve()
        require(
            target.is_relative_to(destination),
            "Invalid destination image path.",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous / reference["local_image"], target)

    reusable_stages = [
        ("evidence", packet),
        ("forecast_agent_assessment", assessment.model_dump()),
        ("forecast_analysis", analysis.model_dump()),
    ]

    for stage, value in reusable_stages:
        audit.complete_stage(stage, value)
        audit.event(
            "stage_reused",
            stage=stage,
            original_run=str(previous),
            original_sha256=manifest["stages"][stage]["sha256"],
        )

    shutil.copy2(
        previous / manifest["stages"]["source_observations"]["file"],
        destination / "source_observations_raw.json",
    )

    provenance = {
        "original_run": str(previous),
        "original_manifest_sha256": sha256(
            previous / "run_manifest.json"
        ),
        "original_response_sha256": (
            manifest["stages"]["source_observations"]["sha256"]
        ),
        "recovery_code_sha256": sha256(Path(__file__)),
        "method": (
            "Catalogue-grounded corrections and assistant photo assessment; "
            "these are not a new agent response."
        ),
        "changes": [
            dict(zip(
                ("article_id", "attribute", "before", "after", "basis"),
                row,
            ))
            for row in corrections
        ],
    }

    write_json(destination / "source_review.json", provenance)

    note = (
        "## Source evidence review\n\n"
        + provenance["method"]
        + " Other source descriptors remain model estimates and need "
        "design-preview review. The original agent response is retained "
        "in source_observations_raw.json.\n\n"
    )

    note += "\n".join(
        f"- Article {article}, {attribute}: {before} → {after}. {basis}"
        for article, attribute, before, after, basis in corrections
    ) + "\n"

    (destination / "SOURCE_REVIEW.md").write_text(
        note, encoding="utf-8"
    )

    audit.complete_stage("source_observations", reviewed.model_dump())
    audit.state["stages"]["source_observations"]["provenance"] = (
        "source_review.json"
    )
    audit.state["status"] = "ready_to_resume_after_source_review"
    audit.save()
    audit.event("source_evidence_corrected", **provenance)

    return destination
