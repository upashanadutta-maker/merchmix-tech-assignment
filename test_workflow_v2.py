
"""Offline regression tests for the Merchmix v2 workflow.

Integration tests simulate the Agents SDK, image API and MCP responses.
They verify code behaviour, not live model accuracy or image quality.
"""

import asyncio
import base64
from copy import deepcopy
import os
import sys
import tempfile
import types
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

from merchmix.audit import Audit, digest_object
from merchmix.contracts import ForecastAnalysis, ForecastAssessment
from merchmix.evidence import EvidenceStore, sha256
from merchmix.forecast_checks import (
    accept_forecast_assessment,
    comparison_sentence,
    expected_assessment,
    render_forecast_analysis,
    sales_comparison,
    validate_forecast_analysis,
)
from merchmix.design_v2 import (
    Observations,
    GarmentObservation,
    Feature,
    Change,
    ProposedConcept,
    Proposal,
    KEYS,
    target_attributes,
    validate_proposal,
    compiled_prompt,
    label,
    compare_board,
    approval_payload,
    save_approval,
    require_approval,
)
from merchmix.workflow_v2 import (
    run_phase,
    approve,
    accept_image,
    preview,
)


class ForecastChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        store = EvidenceStore()
        cls.packet = {
            **store.winners_packet(),
            "evaluation": store.metrics_packet(),
        }

    def test_liliana_is_below_and_percentage_is_correct(self):
        result = sales_comparison(self.packet["winners"][2])
        self.assertEqual(result.relation, "below")
        self.assertAlmostEqual(
            result.percentage_change, -9.45496123345, places=8
        )
        self.assertIn("9.45% below", comparison_sentence(result))

    def test_reversed_agent_answer_is_rejected(self):
        answer = expected_assessment(self.packet)
        answer.style_notes[2].relation = "above"

        with self.assertRaisesRegex(
            ValueError,
            "0915529001.*agent said above.*computed below",
        ):
            accept_forecast_assessment(answer, self.packet)

    def test_changed_explanation_is_rejected(self):
        analysis = render_forecast_analysis(self.packet)
        analysis.style_notes[2].explanation = (
            "Forecast units of 2,016 are above recent sales of 2,227."
        )

        with self.assertRaisesRegex(ValueError, "Python-verified evidence"):
            validate_forecast_analysis(analysis, self.packet)

    def test_unchecked_schema_cannot_be_accepted_as_verified(self):
        current = render_forecast_analysis(self.packet).model_dump()
        incomplete = {
            key: current[key]
            for key in [
                "style_notes",
                "model_selection_explanation",
                "limitations",
            ]
        }

        with self.assertRaises(ValidationError):
            ForecastAnalysis.model_validate(incomplete)

    def test_above_equal_zero_and_rounding_boundaries(self):
        cases = [
            (110, 100, "above", 10),
            (90, 100, "below", -10),
            (100, 100, "equal", 0),
            (1, 0, "above", None),
            (0, 0, "equal", None),
            (0, 100, "below", -100),
            (100.00001, 100, "above", 0.00001),
        ]

        for forecast, recent, relation, percent in cases:
            with self.subTest(forecast=forecast, recent=recent):
                winner = {
                    **self.packet["winners"][0],
                    "pred_sales_4w": forecast,
                    "recent_units_4w": recent,
                }

                result = sales_comparison(winner)
                self.assertEqual(result.relation, relation)

                if percent is None:
                    self.assertIsNone(result.percentage_change)
                else:
                    self.assertAlmostEqual(
                        result.percentage_change, percent
                    )

                sentence = comparison_sentence(result)

                if recent == 0 and forecast > 0:
                    self.assertIn("undefined", sentence)

                if forecast == 100.00001:
                    self.assertIn("less than 0.01% above", sentence)

    def test_invalid_sales_cannot_be_rendered(self):
        for field in ["pred_sales_4w", "recent_units_4w"]:
            for value in [float("nan"), float("inf"), -1, None, True]:
                with self.subTest(field=field, value=value):
                    winner = {
                        **self.packet["winners"][0],
                        field: value,
                    }
                    with self.assertRaises(ValueError):
                        sales_comparison(winner)

    def test_identity_and_model_selection_are_checked(self):
        answer = expected_assessment(self.packet)
        answer.style_notes[2].article_id = "0915529002"

        with self.assertRaisesRegex(ValueError, "source identity"):
            accept_forecast_assessment(answer, self.packet)

        answer = expected_assessment(self.packet)
        answer.selected_model = "xgboost"

        with self.assertRaisesRegex(ValueError, "validation MAE"):
            accept_forecast_assessment(answer, self.packet)

        answer = expected_assessment(self.packet)
        answer.ets_sample_best_model = "holt"

        with self.assertRaisesRegex(ValueError, "same-sample"):
            accept_forecast_assessment(answer, self.packet)

    def test_test_set_cannot_select_the_model(self):
        packet = deepcopy(self.packet)

        for row in packet["evaluation"]["metrics"]["test"]:
            row["MAE"] = 0 if row["model"] == "naive" else 99999

        self.assertEqual(
            expected_assessment(packet).selected_model, "catboost"
        )

        answer = expected_assessment(packet).model_dump()
        answer["selection_basis"] = "test_MAE"

        with self.assertRaises(ValidationError):
            ForecastAssessment.model_validate(answer)

    def test_population_mismatch_prevents_comparison(self):
        packet = deepcopy(self.packet)
        packet["evaluation"]["metrics"]["validation"][0]["n_forecasts"] = 1

        with self.assertRaisesRegex(
            ValueError, "same nonempty evaluation population"
        ):
            expected_assessment(packet)


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = EvidenceStore()

    def test_actual_export_is_consistent(self):
        report = self.store.validate()
        self.assertEqual(report["ranked_styles"], 32553)
        self.assertEqual(report["features"], 34)
        self.assertEqual(
            [
                row["article_id"]
                for row in self.store.winners_packet()["winners"]
            ],
            ["0673677002", "0865799006", "0915529001"],
        )

    def test_changed_winner_is_rejected(self):
        changed = deepcopy(self.store)
        changed.winners.loc[0, "product_code"] = 751471

        with self.assertRaisesRegex(ValueError, "Winner style mismatch"):
            changed.validate()

    def test_score_tampering_is_rejected(self):
        changed = deepcopy(self.store)
        changed.ranking.loc[10, "winner_score"] += 0.001

        with self.assertRaisesRegex(ValueError, "Business score"):
            changed.validate()

    def test_reference_cannot_escape_bundle(self):
        with self.assertRaisesRegex(ValueError, "leaves the bundle"):
            self.store.image_path({"image_path": "../outside.jpg"})


class CacheTests(unittest.TestCase):
    def test_changed_inputs_cannot_reuse_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            Audit(Path(folder), {"skill": "first"})

            with self.assertRaisesRegex(ValueError, "changed"):
                Audit(Path(folder), {"skill": "second"})

    def test_changed_output_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as folder:
            audit = Audit(Path(folder), {"input": "fixture"})
            audit.complete_stage("design", {"concept": "original"})

            (Path(folder) / "design.json").write_text(
                '{"concept":"changed"}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing or changed"):
                audit.cached("design")


def fixtures():
    """Fixed examples for testing; these do not configure the live designs."""
    store = EvidenceStore()

    packet = {
        **store.winners_packet(),
        "evaluation": store.metrics_packet(),
        "references": [
            store.reference_packet(rank) for rank in [1, 2, 3]
        ],
        "model_replay": {"status": "passed", "test_stub": True},
    }

    values = [
        [
            "high_neck", "straight_sleeves", "regular",
            "straight_hem", "plain",
        ],
        [
            "tapered_leg", "high_waist",
            "ankle_length", "no_added_detail",
        ],
        [
            "square_neck", "full_sleeves", "regular",
            "straight_hem", "plain",
        ],
    ]

    colours = ["black", "light_beige", "beige"]

    source = Observations(
        exactly_three_garments=True,
        garments=[
            GarmentObservation(
                winner_rank=winner["winner_rank"],
                article_id=winner["article_id"],
                category=winner["product_type"],
                colour=colours[index],
                full_garment_visible=True,
                features=[
                    Feature(attribute=key, value=value)
                    for key, value in zip(
                        KEYS[winner["product_type"]], values[index]
                    )
                ],
            )
            for index, winner in enumerate(packet["winners"])
        ],
    )

    edits = [
        [
            ("neckline", "high_neck", "crew_neck"),
            ("length", "regular", "cropped"),
        ],
        [
            ("length", "ankle_length", "full_length"),
            ("detail", "no_added_detail", "front_pintucks"),
        ],
        [
            ("neckline", "square_neck", "v_neck"),
            ("surface", "plain", "cable"),
        ],
    ]

    plan = Proposal(
        concepts=[
            ProposedConcept(
                winner_rank=winner["winner_rank"],
                article_id=winner["article_id"],
                colour=colours[index],
                changes=[
                    Change(attribute=key, before=before, after=after)
                    for key, before, after in edits[index]
                ],
            )
            for index, winner in enumerate(packet["winners"])
        ]
    )

    settings = {
        "text_model": "gpt-4.1-mini",
        "image_model": "gpt-image-2",
        "quality": "medium",
        "size": "1536x1024",
        "colours": {
            str(index + 1): colour
            for index, colour in enumerate(colours)
        },
    }

    board = source.model_copy(deep=True)

    for observation, concept, original in zip(
        board.garments, plan.concepts, source.garments
    ):
        observation.features = [
            Feature(attribute=key, value=value)
            for key, value in target_attributes(concept, original).items()
        ]

    return packet, source, plan, settings, board


class DesignTests(unittest.TestCase):
    def setUp(self):
        (
            self.packet,
            self.source,
            self.plan,
            self.settings,
            self.board,
        ) = fixtures()

    def validate(self):
        validate_proposal(
            self.plan,
            self.source,
            self.packet["winners"],
            self.settings["colours"],
        )

    def test_valid_plan(self):
        self.validate()

    def test_no_material_claim_field(self):
        data = self.plan.model_dump()
        data["concepts"][0]["material"] = "verified wool"

        with self.assertRaises(ValueError):
            Proposal.model_validate(data)

    def test_no_material_visual_value(self):
        with self.assertRaises(ValueError):
            Feature(attribute="surface", value="wool blend")

    def test_catalogue_name_cannot_recolour(self):
        self.plan.concepts[1].colour = "pink"

        with self.assertRaisesRegex(ValueError, "colour"):
            self.validate()

    def test_invented_source_rejected(self):
        self.plan.concepts[0].changes[0].before = "square_neck"

        with self.assertRaises(ValueError):
            self.validate()

    def test_unclear_source_change_rejected(self):
        self.source.garments[0].features[0].value = "unclear"
        self.plan.concepts[0].changes[0].before = "unclear"

        with self.assertRaises(ValueError):
            self.validate()

    def test_identical_change_rejected(self):
        self.plan.concepts[0].changes[0].after = "high_neck"

        with self.assertRaises(ValueError):
            self.validate()

    def test_wrong_category_length_rejected(self):
        self.plan.concepts[1].changes[0].after = "cropped"

        with self.assertRaises(ValueError):
            self.validate()

    def test_source_identity_rejected(self):
        self.plan.concepts[0].article_id = "123"

        with self.assertRaises(ValueError):
            self.validate()

    def test_names_and_prompts_compiled(self):
        prompt = compiled_prompt(self.plan, self.source)
        self.assertNotIn("Pink HW", prompt)

        for concept in self.plan.concepts:
            for change in concept.changes:
                self.assertIn(label(change.after), prompt)

    def test_each_failed_change_reported(self):
        self.board.garments[2].features[0].value = "square_neck"

        result = compare_board(
            self.plan, self.source, self.board, self.packet["winners"]
        )

        self.assertEqual(result["automated_check"], "needs_review")
        self.assertTrue(any(
            row["attribute"] == "neckline"
            and row["status"] == "mismatch"
            for row in result["checks"]
        ))

    def test_unclear_does_not_pass(self):
        self.board.garments[0].features[0].value = "unclear"

        result = compare_board(
            self.plan, self.source, self.board, self.packet["winners"]
        )
        self.assertEqual(result["automated_check"], "needs_review")

    def test_missing_panel_does_not_pass(self):
        self.board.exactly_three_garments = False

        result = compare_board(
            self.plan, self.source, self.board, self.packet["winners"]
        )
        self.assertEqual(result["automated_check"], "needs_review")

    def test_approval_binds_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            reviewed_hash = digest_object(
                approval_payload(
                    self.plan, self.source, self.packet, self.settings
                )
            )

            save_approval(
                output, self.plan, self.source,
                self.packet, self.settings, reviewed_hash,
            )
            require_approval(
                output, self.plan, self.source,
                self.packet, self.settings,
            )

            self.settings["quality"] = "high"

            with self.assertRaises(ValueError):
                require_approval(
                    output, self.plan, self.source,
                    self.packet, self.settings,
                )

    def test_stale_preview_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                save_approval(
                    Path(folder), self.plan, self.source,
                    self.packet, self.settings, "wrong",
                )


class IntegrationTests(unittest.TestCase):
    def run_scenario(self, scenario):
        packet, source, plan, settings, board = fixtures()
        calls = []
        images = []
        review_inputs = []

        class Agent:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class Runner:
            @staticmethod
            async def run(agent, inputs, **kwargs):
                calls.append(agent.name)

                if agent.name == "Merchmix reviewable orchestrator":
                    for tool in agent.tools:
                        await tool()
                    answer = "stop"

                elif agent.name == "Forecast specialist":
                    answer = expected_assessment(packet)

                elif agent.name == "Source photo specialist":
                    answer = source

                elif agent.name == "Design specialist":
                    answer = plan

                elif agent.name == "Independent pixel reviewer":
                    review_inputs.append(inputs)
                    answer = board.model_copy(deep=True)

                    if scenario == "unclear":
                        answer.garments[0].features[0].value = "unclear"

                else:
                    raise AssertionError(agent.name)

                return types.SimpleNamespace(
                    final_output=answer,
                    raw_responses=[],
                    new_items=[],
                )

        class Client:
            def __init__(self, **kwargs):
                self.images = self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def edit(self, **kwargs):
                images.append(kwargs["prompt"])

                if scenario == "image_error":
                    raise RuntimeError(
                        "Simulated ambiguous image request"
                    )

                buffer = BytesIO()
                Image.new(
                    "RGB", (1536, 1024), "white"
                ).save(buffer, format="PNG")

                return types.SimpleNamespace(
                    data=[
                        types.SimpleNamespace(
                            b64_json=base64.b64encode(
                                buffer.getvalue()
                            ).decode("ascii")
                        )
                    ],
                    usage=None,
                    _request_id="offline-fixture",
                )

        async def collect(*args, **kwargs):
            return deepcopy(packet)

        agents_module = types.ModuleType("agents")
        agents_module.Agent = Agent
        agents_module.Runner = Runner
        agents_module.ModelSettings = lambda **kwargs: kwargs
        agents_module.function_tool = (
            lambda **kwargs: lambda function: function
        )
        agents_module.set_tracing_disabled = lambda *args: None

        api_module = types.ModuleType("openai")
        api_module.AsyncOpenAI = Client

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(
                sys.modules,
                {"agents": agents_module, "openai": api_module},
            ),
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "offline-fixture-only"},
            ),
            patch("merchmix.workflow_v2.collect_evidence", collect),
        ):
            output = Path(folder) / "run"

            asyncio.run(run_phase(output, settings, "plan"))

            self.assertEqual(len(images), 0)
            self.assertTrue((output / "DESIGN_PREVIEW.html").exists())

            previous_calls = len(calls)

            with self.assertRaisesRegex(ValueError, "approve"):
                asyncio.run(run_phase(output, settings, "render"))

            self.assertEqual(len(calls), previous_calls)
            approve(output, preview(output))

            if scenario == "image_error":
                for _ in range(2):
                    with self.assertRaises((ValueError, RuntimeError)):
                        asyncio.run(
                            run_phase(output, settings, "render")
                        )

                self.assertEqual(len(images), 1)
                return

            asyncio.run(run_phase(output, settings, "render"))
            self.assertEqual(len(images), 1)
            self.assertFalse(
                (output / "final_concept_board.png").exists()
            )

            review_text = review_inputs[0][0]["content"][0]["text"]
            self.assertNotIn('"changes"', review_text)
            self.assertNotIn("front_pintucks", review_text)

            report = (output / "SUBMISSION_REPORT.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("9.45% below", report)
            self.assertNotIn("Pink HW barrel inspired", report)
            self.assertIn(
                "Original catalogue description (quoted", report
            )

            image_hash = sha256(output / "concept_board.png")

            if scenario == "unclear":
                with self.assertRaisesRegex(ValueError, "flagged"):
                    accept_image(output, image_hash)

                accept_image(output, image_hash, accept_flagged=True)
            else:
                accept_image(output, image_hash)

            self.assertTrue(
                (output / "final_concept_board.png").exists()
            )

            asyncio.run(run_phase(output, settings, "render"))
            self.assertEqual(len(images), 1)

    def test_actual_stages_and_approvals(self):
        self.run_scenario("ok")

    def test_unclear_review_needs_explicit_decision(self):
        self.run_scenario("unclear")

    def test_no_retry_after_ambiguous_image_error(self):
        self.run_scenario("image_error")
