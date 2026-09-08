
"""Regression tests for the observed trouser-length failure."""

from copy import deepcopy
import pytest

from test_workflow_v2 import fixtures
from merchmix.observation_schema import ObservationResponse


def example_response():
    packet, source, _, _, _ = fixtures()
    data = {"exactly_three_garments": True}

    for name, garment in zip(
        ("left", "centre", "right"), source.garments
    ):
        data[name] = {
            "category": garment.category,
            "colour": garment.colour,
            "full_garment_visible": garment.full_garment_visible,
            **{
                feature.attribute: feature.value
                for feature in garment.features
            },
        }

    return packet, source, data


def test_trousers_reject_sweater_length_labels():
    _, _, data = example_response()

    for invalid in ["regular", "cropped", "long"]:
        changed = deepcopy(data)
        changed["centre"]["length"] = invalid

        with pytest.raises(ValueError):
            ObservationResponse.model_validate(changed)


def test_observation_conversion_preserves_identity_and_features():
    packet, source, data = example_response()
    response = ObservationResponse.model_validate(data)

    assert response.to_observations(packet["winners"]) == source
