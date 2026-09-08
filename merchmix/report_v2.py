
"""Create reports separating source evidence, proposed designs and observations."""

from __future__ import annotations

import base64
import html
from pathlib import Path

from .design_v2 import attributes, concept_name, label
from .evidence import require, sha256
from .forecast_checks import render_forecast_analysis


def image_tag(path, caption):
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")

    return (
        f'<figure><img src="data:{mime};base64,{data}" '
        f'alt="{html.escape(caption, quote=True)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def create_report(
    output,
    packet,
    plan,
    source,
    comparison=None,
    *,
    human_approved=False,
):
    output = Path(output)
    analysis = render_forecast_analysis(packet)

    title = "Merchmix: reviewable garment concepts"

    if comparison is None:
        status = "Design preview: approval required"
    elif human_approved:
        status = "Human-approved concept board"
    else:
        status = "Generated board: human review required"

    period = (
        f"Existing-style forecasts are for "
        f"{packet['config']['forecast_start']} to "
        f"{packet['config']['forecast_end']}. "
        "New concept sales have not been forecast."
    )

    md = [
        f"# {title}", "",
        status, "",
        period, "",
        "## Forecasting evidence", "",
    ]

    fragments = [
        f"<h1>{title}</h1>",
        f"<p><strong>{status}</strong></p>",
        f"<p>{html.escape(period)}</p>",
        "<h2>Forecasting evidence</h2>",
    ]

    test = {
        row["model"]: row
        for row in packet["evaluation"]["metrics"]["test"]
    }

    improvement = 100 * (
        1 - test["catboost"]["MAE"] / test["naive"]["MAE"]
    )

    result = (
        f"Held-out test MAE: CatBoost {test['catboost']['MAE']:.4f}; "
        f"naive {test['naive']['MAE']:.4f}. "
        f"MAE reduction: {improvement:.1f}%. "
        f"RMSE: {test['catboost']['RMSE']:.4f} versus "
        f"{test['naive']['RMSE']:.4f}. "
        "The exported model is a subsequent final refit."
    )

    md.extend([
        result, "",
        analysis.model_selection_explanation, "",
    ])

    fragments.extend([
        f"<p>{html.escape(result)}</p>",
        f"<p>{html.escape(analysis.model_selection_explanation)}</p>",
    ])

    for note in analysis.style_notes:
        text = f"Article {note.article_id}: {note.explanation}"
        md.append("- " + text)
        fragments.append("<p>" + html.escape(text) + "</p>")

    review_path = output / "SOURCE_REVIEW.md"
    if review_path.is_file():
        review_note = review_path.read_text(encoding="utf-8")
        md.extend(["", review_note, ""])
        fragments.append(
            "<h2>Source evidence review</h2>"
            '<pre style="white-space:pre-wrap;font:inherit">'
            + html.escape(review_note)
            + "</pre>"
        )

    score = (
        "Winner score: 40% forecast-sales percentile, "
        "30% adjusted historical-growth percentile, "
        "30% recent-customer percentile. "
        "These are business choices, not validated success probabilities."
    )

    md.extend(["", score])
    fragments.append("<p>" + html.escape(score) + "</p>")

    if comparison:
        image = output / (
            "final_concept_board.png"
            if human_approved
            else "concept_board.png"
        )

        require(image.is_file(), "Generated board is missing.")

        fragments.extend([
            "<h2>Generated concept board</h2>",
            image_tag(
                image,
                "Panels 1, 2, 3 follow the original winner order.",
            ),
        ])

        md.extend([
            "", "## Generated concept board", "",
            f"![Generated concepts]({image.name})",
        ])

    for winner, reference, concept, observation in zip(
        packet["winners"],
        packet["references"],
        plan.concepts,
        source.garments,
    ):
        path = (output / reference["local_image"]).resolve()

        require(
            path.is_relative_to(output.resolve())
            and sha256(path) == reference["image_sha256"],
            "Source image is missing or changed.",
        )

        name = concept_name(concept, observation)
        subtitle = f"{concept.winner_rank}. {name}"

        identity = (
            f"Source article {winner['article_id']} | "
            f"catalogue name: {winner['prod_name']} | "
            f"selected source colour: {winner['colour_group_name']}"
        )

        md.extend([
            "",
            f"## {subtitle}",
            "",
            identity,
            "",
            f"![Original selected product]({reference['local_image']})",
            "",
            "Original catalogue description "
            "(quoted; applies to the existing product):",
            "",
            "> " + winner["detail_desc"],
            "",
            "### Proposed design",
            "",
            f"Proposed colour: {label(concept.colour)}.",
        ])

        fragments.extend([
            f"<section><h2>{html.escape(subtitle)}</h2>",
            f"<p>{html.escape(identity)}</p>",
            image_tag(path, "Original selected product"),
            "<p><strong>Original catalogue description; "
            "existing product only:</strong></p>",
            "<blockquote>"
            + html.escape(winner["detail_desc"])
            + "</blockquote>",
            "<h3>Proposed design</h3>",
            "<p>Proposed colour: "
            + html.escape(label(concept.colour))
            + ".</p><ul>",
        ])

        for change in concept.changes:
            line = (
                f"{label(change.attribute).title()}: "
                f"{label(change.before)} → {label(change.after)}."
            )
            md.append("- " + line)
            fragments.append("<li>" + html.escape(line) + "</li>")

        changed = {
            change.attribute for change in concept.changes
        }

        kept = "; ".join(
            f"{label(key)}: {label(value)}"
            for key, value in attributes(observation).items()
            if key not in changed and value != "unclear"
        )

        md.extend([
            "",
            "Retained visual source features: " + kept + ".",
        ])

        fragments.append(
            "</ul><p>Retained visual source features: "
            + html.escape(kept)
            + ".</p>"
        )

        rationale = (
            "Design hypothesis: explore these two visible variations "
            "while retaining the listed source features. Forecasting "
            "selected the source style; it does not establish that "
            "these variations will increase sales."
        )

        md.extend(["", rationale])
        fragments.append("<p>" + html.escape(rationale) + "</p>")

        if comparison:
            rows = [
                row
                for row in comparison["checks"]
                if row["winner_rank"] == concept.winner_rank
            ]

            md.extend([
                "",
                "### Proposed versus independently observed",
                "",
                "| Feature | Proposed | Observed by reviewer | Check |",
                "| --- | --- | --- | --- |",
            ])

            fragments.append(
                "<h3>Proposed versus independently observed</h3>"
                "<table><tr>"
                "<th>Feature</th>"
                "<th>Proposed</th>"
                "<th>Reviewer observed</th>"
                "<th>Check</th>"
                "</tr>"
            )

            for row in rows:
                values = [
                    label(row[key])
                    for key in [
                        "attribute", "proposed", "observed", "status"
                    ]
                ]

                md.append("| " + " | ".join(values) + " |")

                fragments.append(
                    "<tr>"
                    + "".join(
                        "<td>" + html.escape(value) + "</td>"
                        for value in values
                    )
                    + "</tr>"
                )

            fragments.append("</table>")

        fragments.append("</section>")

    explanation = (
        "A manager agent delegates MCP evidence retrieval and numerical "
        "assessment, source-photo observation, and design proposal to "
        "specialist tools. The design specialist receives the complete "
        "reusable skill plus the constrained visual vocabulary. "
        "A user approves the saved source observations, colours, changes "
        "and compiled image prompt. A second manager phase delegates one "
        "reference-image edit and independent visual observation. "
        "The visual reviewer receives the generated pixels without the "
        "proposed design values; Python compares its observations with "
        "the approved plan. A human reviews the final image. "
        "Raw agent responses, MCP calls, approvals, prompts, receipts "
        "and file hashes are saved. "
        "No automatic image regeneration occurs."
    )

    limitations = [
        *analysis.limitations,
        (
            "Source-photo and generated-image observations are model "
            "estimates and may be wrong; unclear attributes remain explicit."
        ),
        (
            "Catalogue fibre and material statements describe the original "
            "product only. The new concepts have no verified composition, "
            "fit, comfort, function or environmental benefit."
        ),
        (
            "The visual vocabulary is deliberately limited; fine "
            "construction details and manufacturing feasibility "
            "need specialist review."
        ),
        (
            "Only two test forecast dates and a separate "
            "600-row ETS sample were evaluated."
        ),
        (
            "Human approval records a design decision, not proof of "
            "future demand or manufacturing feasibility."
        ),
        (
            "Raw H&M training data and the predictive modelling notebook "
            "are separate inputs to reproduce model training."
        ),
    ]

    md.extend([
        "", "## Workflow", "",
        explanation,
        "", "## Limitations", "",
        *["- " + item for item in limitations],
    ])

    fragments.extend([
        "<h2>Workflow</h2>",
        "<p>" + html.escape(explanation) + "</p>",
        "<h2>Limitations</h2><ul>",
        "".join(
            "<li>" + html.escape(item) + "</li>"
            for item in limitations
        ),
        "</ul>",
    ])

    if comparison:
        note = (
            f"Automated comparison: {comparison['automated_check']}. "
            f"Human review: {'approved' if human_approved else 'pending'}."
        )

        md.extend(["", note])
        fragments.append(
            "<p><strong>" + html.escape(note) + "</strong></p>"
        )

    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Merchmix</title>
<style>
body {
    max-width: 1100px;
    margin: 35px auto;
    padding: 0 24px;
    font: 16px/1.6 system-ui;
    color: #202b30;
    background: #faf9f6;
}
h1, h2, h3 { line-height: 1.25; }
h2 { margin-top: 32px; }
img { max-width: 100%; height: auto; }
section img { max-height: 340px; object-fit: contain; }
figure { margin: 12px 0; }
figcaption { font-size: 13px; }
section { border-top: 1px solid #c8d0ce; margin-top: 32px; }
table { width: 100%; border-collapse: collapse; }
td, th {
    padding: 9px;
    text-align: left;
    border-bottom: 1px solid #ccc;
}
blockquote { border-left: 3px solid #829c96; padding-left: 16px; }
@media print {
    body { background: white; }
    tr, figure { break-inside: avoid; }
}
@media(max-width: 600px) {
    table { font-size: 12px; }
    td, th { padding: 4px; }
}
</style>
</head>
<body>
""" + "".join(fragments) + "</body></html>"

    prefix = (
        "DESIGN_PREVIEW"
        if comparison is None
        else "SUBMISSION_REPORT"
    )

    (output / f"{prefix}.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )

    report_path = output / f"{prefix}.html"
    report_path.write_text(page, encoding="utf-8")

    return report_path
