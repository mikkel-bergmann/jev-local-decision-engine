import pydantic
import pytest
import torch

from jev_local_engine import (
    SCHEMA,
    IntentDecision,
    build_field_prompts,
    candidate_probabilities,
    resolve_candidates,
    score_fields,
    score_fields_full_sequence,
)

CALM_POSITIVE_TEXT = (
    "Thanks so much for the quick help yesterday — everything is working "
    "great now and I really appreciate the support!"
)


@pytest.mark.requires_model
def test_resolve_candidates_returns_distinct_ids(qwen_tokenizer):
    choices = ["low", "medium", "high", "critical"]
    candidates = resolve_candidates(qwen_tokenizer, choices)
    assert set(candidates.keys()) == set(choices)
    assert len(set(candidates.values())) == len(choices)


@pytest.mark.requires_model
def test_resolve_candidates_bare_form_distinct_for_category(qwen_tokenizer):
    choices = SCHEMA["category"]
    candidates = resolve_candidates(qwen_tokenizer, choices)
    assert set(candidates.keys()) == set(choices)
    assert len(set(candidates.values())) == len(choices)


@pytest.mark.requires_model
def test_resolve_candidates_raises_on_collision(qwen_tokenizer):
    # In bare form (the default resolve_candidates uses, matching a prompt that
    # ends in a newline), "technical_support" and "technicality" share the
    # first token id 72137 ("technical") under the real Qwen2.5 vocabulary.
    choices = ["technical_support", "technicality"]
    with pytest.raises(ValueError) as excinfo:
        resolve_candidates(qwen_tokenizer, choices)
    message = str(excinfo.value)
    assert "technical_support" in message
    assert "technicality" in message


def test_candidate_probabilities_normalizes_and_discriminates():
    row_logits = torch.zeros(1000, dtype=torch.float16)
    row_logits[5] = 1.0
    row_logits[10] = 4.0
    row_logits[15] = 0.5
    candidate_ids = torch.tensor([5, 10, 15])

    probs = candidate_probabilities(row_logits, candidate_ids)

    assert abs(float(probs.sum()) - 1.0) < 0.001
    assert not torch.allclose(probs, probs[0].expand_as(probs))


@pytest.mark.requires_model
def test_build_field_prompts_batch_has_one_row_per_field(qwen_tokenizer):
    text = "My account was double charged for last month's subscription, fix this immediately!"
    fields, prompts = build_field_prompts(qwen_tokenizer, text, SCHEMA)

    assert len(fields) == 3
    assert len(prompts) == 3

    batch = qwen_tokenizer(
        prompts, padding=True, return_tensors="pt", add_special_tokens=False
    )
    assert batch["input_ids"].shape[0] == 3


def test_intent_decision_rejects_off_schema_category():
    with pytest.raises(pydantic.ValidationError):
        IntentDecision(
            category="not_a_category",
            urgency="low",
            sentiment="neutral",
        )


@pytest.mark.requires_model
def test_batch_scoring_matches_unbatched_scoring(qwen_model_and_tokenizer):
    model, tokenizer = qwen_model_and_tokenizer
    text = "My account was double charged for last month's subscription, fix this immediately!"

    batched = score_fields(model, tokenizer, text, SCHEMA)["sentiment"]

    single_field_schema = {"sentiment": SCHEMA["sentiment"]}
    unbatched = score_fields(model, tokenizer, text, single_field_schema)["sentiment"]

    assert set(batched.keys()) == set(unbatched.keys())
    for choice in batched:
        assert abs(batched[choice] - unbatched[choice]) < 0.01


@pytest.mark.requires_model
def test_full_sequence_scoring_reduces_technical_support_bias(qwen_model_and_tokenizer):
    # Under first-token scoring, "technical_support" wins the calm positive
    # thank-you note at 0.999993 — its first token ("technical") is a high-prior
    # completion regardless of input. Full-sequence scoring, which weighs both
    # of the candidate's tokens, must knock that down below the 0.99 it gets
    # under first-token scoring.
    model, tokenizer = qwen_model_and_tokenizer
    category_schema = {"category": SCHEMA["category"]}

    result = score_fields_full_sequence(model, tokenizer, CALM_POSITIVE_TEXT, category_schema)

    assert result["category"]["technical_support"] < 0.99


@pytest.mark.requires_model
def test_full_sequence_scoring_runs_one_forward_pass(qwen_model_and_tokenizer):
    model, tokenizer = qwen_model_and_tokenizer
    text = "My account was double charged for last month's subscription, fix this immediately!"

    call_count = {"n": 0}
    original_forward = model.forward

    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_forward(*args, **kwargs)

    model.forward = counting_forward
    try:
        # Full three-field schema: 4 + 4 + 3 = 11 (field, candidate) sequences.
        score_fields_full_sequence(model, tokenizer, text, SCHEMA)
    finally:
        model.forward = original_forward

    assert call_count["n"] == 1


@pytest.mark.requires_model
def test_full_sequence_scoring_probabilities_normalize(qwen_model_and_tokenizer):
    model, tokenizer = qwen_model_and_tokenizer
    text = "My account was double charged for last month's subscription, fix this immediately!"

    result = score_fields_full_sequence(model, tokenizer, text, SCHEMA)

    for field, probabilities in result.items():
        total = sum(probabilities.values())
        assert abs(total - 1.0) < 0.001, f"{field} probabilities sum to {total}, not 1.0"
