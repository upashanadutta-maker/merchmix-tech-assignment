
"""Category-specific observation fields for this export's three source slots."""

from typing import Literal

from .design_v2 import (
    Strict, Colour, Observations, GarmentObservation, Feature, KEYS,
)
from .evidence import require


class SweaterView(Strict):
    category: Literal["Sweater", "Trousers", "unclear"]
    colour: Colour
    full_garment_visible: bool
    neckline: Literal[
        "high_neck", "crew_neck", "v_neck",
        "square_neck", "boat_neck", "unclear",
    ]
    sleeves: Literal[
        "straight_sleeves", "full_sleeves", "short_sleeves", "unclear",
    ]
    length: Literal["cropped", "regular", "long", "unclear"]
    hem: Literal["straight_hem", "curved_hem", "unclear"]
    surface: Literal["plain", "ribbed", "cable", "unclear"]


class TrouserView(Strict):
    category: Literal["Sweater", "Trousers", "unclear"]
    colour: Colour
    full_garment_visible: bool
    leg: Literal[
        "tapered_leg", "straight_leg", "wide_leg", "flared_leg", "unclear",
    ]
    waist: Literal["high_waist", "mid_waist", "unclear"]
    length: Literal["ankle_length", "full_length", "unclear"]
    detail: Literal[
        "no_added_detail", "front_pintucks",
        "patch_pockets", "cargo_pocket", "unclear",
    ]


class ObservationResponse(Strict):
    exactly_three_garments: bool
    left: SweaterView
    centre: TrouserView
    right: SweaterView

    def to_observations(self, winners):
        require(
            [winner["product_type"] for winner in winners]
            == ["Sweater", "Trousers", "Sweater"],
            "Observation slots do not match this export.",
        )

        garments = []

        for view, winner in zip(
            (self.left, self.centre, self.right), winners
        ):
            garments.append(
                GarmentObservation(
                    winner_rank=winner["winner_rank"],
                    article_id=winner["article_id"],
                    category=view.category,
                    colour=view.colour,
                    full_garment_visible=view.full_garment_visible,
                    features=[
                        Feature(attribute=key, value=getattr(view, key))
                        for key in KEYS[winner["product_type"]]
                    ],
                )
            )

        return Observations(
            garments=garments,
            exactly_three_garments=self.exactly_three_garments,
        )


def adapt_observation_response(value, audit, stage):
    if not isinstance(value, ObservationResponse):
        return value

    packet = audit.cached("evidence")
    require(packet is not None, "Source identities are unavailable.")

    audit.complete_stage(stage + "_raw", value.model_dump())
    return value.to_observations(packet["winners"])
