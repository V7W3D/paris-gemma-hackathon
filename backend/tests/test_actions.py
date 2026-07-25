from __future__ import annotations

import pytest

from backend.models.schemas.actions import (
    AssessOutput,
    DecomposeOutput,
    GatherOutput,
    VerdictOutput,
    extract_json_object,
    salvage_output,
)
from backend.models.schemas.context import ClaimStatus, VerdictLabel


def test_extract_json_object_ignores_surrounding_prose():
    text = 'Sure! Here is the result:\n{"claims": ["a"], "reply": ""}\nHope that helps.'
    assert extract_json_object(text) == {"claims": ["a"], "reply": ""}


def test_extract_json_object_tolerates_trailing_commas():
    assert extract_json_object('{"claims": ["a",],}') == {"claims": ["a"]}


def test_extract_json_object_skips_a_leading_brace_that_is_not_json():
    text = 'note {not json} then {"thought": "ok", "claims": []}'
    assert extract_json_object(text)["thought"] == "ok"


def test_extract_json_object_raises_without_an_object():
    with pytest.raises(ValueError):
        extract_json_object("no json at all")


def test_salvage_output_builds_the_stage_model():
    output = salvage_output('```json\n{"claims": ["the sky is blue"]}\n```'.strip("`"), DecomposeOutput)
    assert output.claims == ["the sky is blue"]


def test_decompose_accepts_objects_instead_of_strings():
    output = DecomposeOutput.model_validate({"claims": [{"text": "a claim"}, "another"]})
    assert output.claims == ["a claim", "another"]


def test_gather_defaults_to_no_tool_calls():
    output = GatherOutput.model_validate({"thought": "done"})
    assert output.tool_calls == []
    assert output.done is False


def test_assessment_falls_back_to_insufficient_on_an_unknown_status():
    output = AssessOutput.model_validate(
        {"assessments": [{"claim_id": "c1", "status": "PROBABLY", "rationale": "unsure"}]}
    )
    assert output.assessments[0].status is ClaimStatus.INSUFFICIENT


def test_assessment_normalises_case():
    output = AssessOutput.model_validate(
        {"assessments": [{"claim_id": "c1", "status": "Supported"}]}
    )
    assert output.assessments[0].status is ClaimStatus.SUPPORTED


def test_verdict_clamps_confidence_and_unknown_labels():
    output = VerdictOutput.model_validate({"label": "very true", "confidence": 4.2, "summary": "x"})
    assert output.label is VerdictLabel.UNVERIFIED
    assert output.confidence == 1.0
