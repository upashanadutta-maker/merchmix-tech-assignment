"""Load, validate, and replay exported forecasting evidence."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class EvidenceStore:
    def __init__(self, bundle: Path | str = DEFAULT_BUNDLE):
        self.bundle = Path(bundle).resolve()

        self.config = json.loads(
            (self.bundle / "config.json").read_text(encoding="utf-8")
        )
        self.ranking = pd.read_csv(self.bundle / "final_ranking.csv")
        self.winners = pd.read_csv(self.bundle / "winning_styles.csv")
        self.references = json.loads(
            (self.bundle / "reference_products.json").read_text(
                encoding="utf-8"
            )
        )

        self.validate()

    def validate(self) -> dict:
        c = self.config
        r = self.ranking
        w = self.winners
        keys = c["style_keys"]

        require(
            c["features"] == c["numeric_features"] + c["categorical_features"],
            "Feature order differs from the exported training schema.",
        )
        require(not r[keys].isna().any().any(), "Missing style identity.")
        require(not r.duplicated(keys).any(), "Duplicate style identities.")
        require(
            set(c["features"]).issubset(r.columns),
            "Missing model features.",
        )
        require(len(w) == 3, "Exactly three selected styles are required.")
        require(
            w["winner_rank"].tolist() == [1, 2, 3],
            "Winner ordering changed.",
        )
        require(
            r["winner_rank"].tolist() == list(range(1, len(r) + 1)),
            "Ranking order is inconsistent.",
        )
        require(
            np.isfinite(
                r[c["numeric_features"]].to_numpy(dtype=float)
            ).all(),
            "Numeric features contain missing or infinite values.",
        )
        require(
            np.isfinite(r["pred_sales_4w"]).all()
            and (r["pred_sales_4w"] >= 0).all(),
            "Invalid exported forecasts.",
        )

        for _, row in w.iterrows():
            saved = r.iloc[int(row["winner_rank"]) - 1]

            require(
                all(row[key] == saved[key] for key in keys),
                "Winner style mismatch.",
            )

            for field in ["pred_sales_4w", "winner_score", "customers_4w"]:
                require(
                    np.isclose(
                        row[field], saved[field], rtol=0, atol=1e-8
                    ),
                    f"Winner {field} differs from full ranking.",
                )

        # Reproduce the notebook's historical-growth adjustment.
        valid = (
            (r["style_age_weeks"] >= 8)
            & (r["prior_units_4w"] > 0)
        )

        k = (
            float(r.loc[valid, "prior_units_4w"].median())
            if valid.any()
            else 1.0
        )

        require(
            np.isclose(k, c["growth_reliability_k"]),
            "Growth shrinkage changed.",
        )

        reliability = r["prior_units_4w"] / (r["prior_units_4w"] + k)
        adjusted = (r["raw_momentum"] * reliability).where(valid)

        require(
            np.allclose(
                adjusted,
                r["adjusted_momentum"],
                atol=1e-10,
                rtol=0,
                equal_nan=True,
            ),
            "Adjusted momentum changed.",
        )

        computed = {
            "sales_rank": r["pred_sales_4w"].rank(
                pct=True, method="average"
            ),
            "growth_rank": r["adjusted_momentum"].rank(
                pct=True, method="average"
            ).fillna(0.5),
            "customer_rank": r["customers_4w"].rank(
                pct=True, method="average"
            ),
        }

        for name, values in computed.items():
            require(
                np.allclose(values, r[name], rtol=0, atol=1e-10),
                f"{name} changed.",
            )

        score = sum(
            r[name] * weight
            for name, weight in c["winner_score_weights"].items()
        )

        require(
            np.allclose(
                score, r["winner_score"], rtol=0, atol=1e-10
            ),
            "Business score differs from configured weights.",
        )

        ordered = r.sort_values(
            ["winner_score", "pred_sales_4w", *keys],
            ascending=[False, False] + [True] * len(keys),
        )

        require(
            ordered.index.tolist() == r.index.tolist(),
            "Ranking tie-break order changed.",
        )
        require(
            pd.to_datetime(r["snapshot_week"]).eq(
                pd.Timestamp(c["snapshot_week"])
            ).all(),
            "Mixed forecasting snapshots.",
        )
        require(
            pd.Timestamp(c["forecast_start"])
            > pd.Timestamp(c["history_available_through"]),
            "Forecast overlaps the available history.",
        )
        require(
            (
                pd.Timestamp(c["forecast_end"])
                - pd.Timestamp(c["forecast_start"])
            ).days == 27,
            "This export must cover four weeks.",
        )

        require(
            len(self.references) == 3,
            "Exactly three source images are required.",
        )
        require(
            sorted(ref["winner_rank"] for ref in self.references)
            == [1, 2, 3],
            "Reference ranks are missing or duplicated.",
        )

        for ref in self.references:
            saved = w.loc[
                w["winner_rank"].eq(ref["winner_rank"])
            ].iloc[0]

            require(
                all(saved[key] == ref[key] for key in keys),
                "Reference style mismatch.",
            )
            require(
                np.isclose(
                    saved["pred_sales_4w"],
                    ref["pred_sales_4w"],
                    atol=1e-7,
                    rtol=0,
                ),
                "Reference forecast differs from its style forecast.",
            )

            article = ref["article_id"]

            require(
                isinstance(article, str)
                and len(article) == 10
                and article.isdigit(),
                "Article IDs must retain their leading zero.",
            )
            require(
                int(article[:-3]) == int(ref["product_code"]),
                "Article/product mismatch.",
            )

            path = self.image_path(ref)

            require(
                path.is_file(),
                f"Missing source image: {path.name}",
            )

        return {
            "status": "passed",
            "ranked_styles": len(r),
            "selected_styles": len(w),
            "features": len(c["features"]),
            "forecast_start": c["forecast_start"],
            "forecast_end": c["forecast_end"],
            "model_replay": "not_run",
        }

    def image_path(self, ref: dict) -> Path:
        path = (self.bundle / ref["image_path"]).resolve()

        require(
            path.is_relative_to(self.bundle),
            "Reference path leaves the bundle.",
        )

        return path

    def winners_packet(self) -> dict:
        fields = [
            "winner_rank",
            *self.config["style_keys"],
            "product_type",
            "pred_sales_4w",
            "recent_units_4w",
            "prior_units_4w",
            "customers_4w",
            "adjusted_momentum",
            "sales_rank",
            "growth_rank",
            "customer_rank",
            "winner_score",
        ]

        rows = json.loads(
            self.winners[fields].to_json(
                orient="records",
                double_precision=15,
            )
        )

        references = {
            ref["winner_rank"]: ref
            for ref in self.references
        }

        for row in rows:
            ref = references[row["winner_rank"]]

            row.update({
                key: ref[key]
                for key in ["article_id", "prod_name", "detail_desc"]
            })

            row["source_image_sha256"] = sha256(
                self.image_path(ref)
            )

        return {
            "config": self.config,
            "ranked_styles": len(self.ranking),
            "winners": rows,
            "limits": [
                "Forecasts describe existing style-colour observed sales, "
                "not new concepts.",
                "The 40/30/30 winner score is a business heuristic, "
                "not a success probability.",
                "Customers are observed recent distinct customers, "
                "not forecast customers.",
                "No inventory history: demand and stock availability "
                "are not separated.",
                "Sales associations do not identify which visual "
                "details caused sales.",
            ],
        }

    def metrics_packet(self) -> dict:
        files = {
            "validation": "validation_metrics.csv",
            "test": "test_metrics.csv",
            "ets_sample": "ets_sample_metrics.csv",
        }

        return {
            "metrics": {
                key: json.loads(
                    pd.read_csv(self.bundle / filename).to_json(
                        orient="records"
                    )
                )
                for key, filename in files.items()
            },
            "scope_note": (
                "ETS uses 600 sampled rows; tree tables use full populations. "
                "These errors cannot be compared across populations. "
                "Test scores are from the evaluation model, "
                "not the final all-label refit."
            ),
        }

    def reference_packet(self, winner_rank: int) -> dict:
        ref = next(
            (
                item
                for item in self.references
                if item["winner_rank"] == winner_rank
            ),
            None,
        )

        require(ref is not None, "Unknown winner rank.")
        path = self.image_path(ref)

        return {
            "winner_rank": winner_rank,
            "article_id": ref["article_id"],
            "product_code": ref["product_code"],
            "colour_group_code": ref["colour_group_code"],
            "colour_group_name": ref["colour_group_name"],
            "prod_name": ref["prod_name"],
            "detail_desc": ref["detail_desc"],
            "mime_type": "image/jpeg",
            "image_sha256": sha256(path),
            "image_base64": base64.b64encode(
                path.read_bytes()
            ).decode("ascii"),
        }

    def replay_model(self, all_rows: bool = False) -> dict:
        """Load the saved model and predict without fitting."""

        from catboost import CatBoostRegressor

        c = self.config
        source = self.ranking if all_rows else self.winners
        x = source[c["features"]].copy()

        for column in c["numeric_features"]:
            x[column] = pd.to_numeric(
                x[column], errors="raise"
            )

        for column in c["categorical_features"]:
            x[column] = (
                x[column]
                .astype("string")
                .fillna(c["missing_category_value"])
                .astype(str)
            )

        model = CatBoostRegressor()
        model.load_model(str(self.bundle / "sales_model.cbm"))

        require(
            model.tree_count_ == c["number_of_trees"],
            "Wrong model tree count.",
        )

        if model.feature_names_:
            require(
                model.feature_names_ == c["features"],
                "Saved model feature order differs.",
            )

        predictions = np.maximum(
            model.predict(x),
            c["prediction_lower_bound"],
        )

        error = np.abs(
            predictions - source["pred_sales_4w"].to_numpy()
        )

        require(
            np.isfinite(predictions).all(),
            "Model replay returned invalid values.",
        )
        require(
            np.allclose(
                predictions,
                source["pred_sales_4w"],
                rtol=1e-6,
                atol=1e-4,
            ),
            "Model replay differs from exported forecasts; "
            f"maximum error {error.max():.6g}.",
        )

        return {
            "status": "passed",
            "rows": len(source),
            "trees": model.tree_count_,
            "maximum_absolute_difference": float(error.max()),
            "refitted": False,
            "model_sha256": sha256(
                self.bundle / "sales_model.cbm"
            ),
        }
