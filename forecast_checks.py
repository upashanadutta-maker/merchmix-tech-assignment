"""Compute forecast comparisons and reject conflicting agent answers."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from .contracts import (
    ForecastAnalysis,
    ForecastAssessment,
    ForecastNote,
    SalesComparison,
    validate_identity,
)
from .evidence import require


def nonnegative_number(value, field: str) -> Decimal:
    message = f"Invalid {field}: expected a finite nonnegative number."

    require(not isinstance(value, bool), message)

    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise ValueError(message) from error

    require(number.is_finite() and number >= 0, message)

    return number


def sales_comparison(winner: dict) -> SalesComparison:
    forecast = nonnegative_number(
        winner["pred_sales_4w"], "pred_sales_4w"
    )
    recent = nonnegative_number(
        winner["recent_units_4w"], "recent_units_4w"
    )
    difference = forecast - recent

    if difference > 0:
        relation = "above"
    elif difference < 0:
        relation = "below"
    else:
        relation = "equal"

    percentage = 100 * difference / recent if recent else None

    return SalesComparison(
        winner_rank=winner["winner_rank"],
        article_id=winner["article_id"],
        forecast_units=float(forecast),
        recent_units=float(recent),
        difference_units=float(difference),
        percentage_change=(
            float(percentage)
            if percentage is not None
            else None
        ),
        relation=relation,
    )


def best_mae(rows: list[dict], scope: str) -> str:
    require(bool(rows), f"Missing {scope} metrics.")

    require(
        len({row["model"] for row in rows}) == len(rows),
        f"Duplicate {scope} model.",
    )

    counts = {
        nonnegative_number(row["n_forecasts"], "n_forecasts")
        for row in rows
    }

    require(
        len(counts) == 1 and next(iter(counts)) > 0,
        f"Compare {scope} models on the same nonempty evaluation population.",
    )

    scored = [
        (
            nonnegative_number(row["MAE"], "MAE"),
            row["model"],
        )
        for row in rows
    ]

    return min(scored)[1]


def expected_assessment(packet: dict) -> ForecastAssessment:
    metrics = packet["evaluation"]["metrics"]

    return ForecastAssessment(
        style_notes=[
            {
                "winner_rank": winner["winner_rank"],
                "article_id": winner["article_id"],
                "relation": sales_comparison(winner).relation,
            }
            for winner in packet["winners"]
        ],
        selected_model=best_mae(
            metrics["validation"], "validation"
        ),
        selection_basis="validation_MAE",
        ets_sample_best_model=best_mae(
            metrics["ets_sample"], "ETS sample"
        ),
    )


def validate_forecast_assessment(
    assessment: ForecastAssessment,
    packet: dict,
) -> None:
    validate_identity(
        assessment.style_notes,
        packet["winners"],
    )

    expected = expected_assessment(packet)

    for actual, correct in zip(
        assessment.style_notes,
        expected.style_notes,
    ):
        require(
            actual.relation == correct.relation,
            f"Forecast comparison mismatch for article {actual.article_id}: "
            f"agent said {actual.relation}; "
            f"Python computed {correct.relation}.",
        )

    require(
        assessment.selected_model == expected.selected_model,
        "Selected model does not match validation MAE.",
    )
    require(
        assessment.selection_basis == expected.selection_basis,
        "Model selection must use validation MAE.",
    )
    require(
        assessment.ets_sample_best_model
        == expected.ets_sample_best_model,
        "ETS-sample conclusion does not match the same-sample MAE comparison.",
    )


def comparison_sentence(comparison: SalesComparison) -> str:
    lead = (
        f"Forecast sales are {comparison.forecast_units:,.2f} units "
        "for the next four weeks. "
        f"Recent four-week sales were {comparison.recent_units:,.2f} units. "
    )

    if comparison.relation == "equal":
        return lead + "The forecast equals recent sales."

    if comparison.percentage_change is None:
        return lead + (
            f"The forecast is {comparison.relation} recent sales. "
            "Percentage change is undefined because recent sales were zero."
        )

    magnitude = abs(comparison.percentage_change)

    amount = (
        "less than 0.01%"
        if magnitude < 0.01
        else f"{magnitude:.2f}%"
    )

    return lead + (
        f"The forecast is {amount} "
        f"{comparison.relation} recent sales."
    )


def render_forecast_analysis(packet: dict) -> ForecastAnalysis:
    assessment = expected_assessment(packet)

    comparisons = [
        sales_comparison(winner)
        for winner in packet["winners"]
    ]

    notes = []

    for winner, comparison in zip(
        packet["winners"],
        comparisons,
    ):
        explanation = comparison_sentence(comparison) + (
            " This existing style-colour is winner rank "
            f"{winner['winner_rank']} under the configured sales, "
            "growth and recent-customer score. "
            "The score is a business heuristic, not a probability. "
            "The new garment concept has no sales forecast."
        )

        notes.append(
            ForecastNote(
                winner_rank=winner["winner_rank"],
                article_id=winner["article_id"],
                explanation=explanation,
            )
        )

    return ForecastAnalysis(
        schema_version=2,
        explanation_source="python_verified_evidence",
        comparisons=comparisons,
        style_notes=notes,
        model_selection_explanation=(
            f"{assessment.selected_model} has the lowest MAE on the "
            "common validation population and was selected using validation MAE. "
            "Test metrics are a separate held-out evaluation, "
            "not a selection criterion. "
            f"{assessment.ets_sample_best_model} has the lowest MAE "
            "on the separate ETS sample. "
            "ETS sample errors cannot be compared directly with the "
            "full-population tree errors. "
            "No XGBoost test result is supplied. "
            "The exported model is the subsequent final refit."
        ),
        limitations=[
            "These are observed sales forecasts for existing style-colour "
            "groups, not new designs.",
            "Inventory availability and unconstrained demand cannot be "
            "separated with these data.",
            "Recent customer counts are observations, not customer forecasts.",
            "The winner-score weights are a business heuristic and have "
            "not been prospectively validated.",
            "Forecasting evidence does not establish causal effects "
            "of garment design features.",
            "Numerical forecast explanations are checked; broader creative "
            "claims still need review.",
        ],
    )


def accept_forecast_assessment(
    assessment: ForecastAssessment,
    packet: dict,
) -> ForecastAnalysis:
    validate_forecast_assessment(assessment, packet)

    return render_forecast_analysis(packet)


def validate_forecast_analysis(
    analysis: ForecastAnalysis,
    packet: dict,
) -> None:
    require(
        analysis.model_dump()
        == render_forecast_analysis(packet).model_dump(),
        "Forecast analysis differs from the Python-verified evidence; "
        "do not publish or reuse it.",
    )
