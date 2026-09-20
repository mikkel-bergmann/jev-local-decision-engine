import os

import pytest
import torch

# Ten prompts varying in length and subject, used across this file's batch
# tests. Deliberately mixes a short fragment with a long, multi-clause
# sentence so padding cost and any padding-related drift can show.
TEN_TEXTS = [
    "Thanks!",
    "My account was double charged for last month's subscription, fix this immediately!",
    "What are your hours on weekends?",
    "I need to cancel my subscription right away, this is urgent.",
    "Can you help me reset my password? I've tried three times and it keeps failing.",
    "Great job on the new dashboard, it's much faster now.",
    "The app crashed again while I was checking out, I'm really frustrated.",
    "Do you offer discounts for annual plans?",
    "I love the new feature you shipped last week, works perfectly.",
    (
        "I've been a loyal customer for over five years, and this is the third "
        "time this month my payment has failed to process even though my card "
        "is valid and has sufficient funds, so I need someone to look into this "
        "immediately and give me an update on when it will be resolved."
    ),
]


def load_trained_heads():
    from decision_heads import HEADS_SCHEMA, load_heads

    heads_path = os.path.join("data", "heads.pt")
    if not os.path.exists(heads_path):
        pytest.skip(f"trained heads not found at {heads_path}; run train_heads.py first")
    return load_heads(heads_path, HEADS_SCHEMA)


GATE_HEAD_PATH = os.path.join("data", "gate_head.pt")
TOOL_HEAD_PATH = os.path.join("data", "tool_head.pt")

SAMPLE_TOOL_REQUEST = "Please refund this customer's most recent payment."
SAMPLE_REJECTED_TEXT = "hello"


def load_trained_gate_heads():
    from decision_heads import GATE_SCHEMA, load_heads

    if not os.path.exists(GATE_HEAD_PATH):
        pytest.skip(f"trained gate head not found at {GATE_HEAD_PATH}; run train_heads.py first")
    return load_heads(GATE_HEAD_PATH, GATE_SCHEMA)


def load_trained_tool_heads():
    from decision_heads import TOOL_SCHEMA, load_heads

    if not os.path.exists(TOOL_HEAD_PATH):
        pytest.skip(f"trained tool head not found at {TOOL_HEAD_PATH}; run train_heads.py first")
    return load_heads(TOOL_HEAD_PATH, TOOL_SCHEMA)


@pytest.mark.requires_model
def test_run_heads_decision_batch_returns_ten_results_in_order():
    from decision_heads import run_heads_decision, run_heads_decision_batch

    heads = load_trained_heads()

    batch_results = run_heads_decision_batch(TEN_TEXTS, heads)
    assert len(batch_results) == 10

    for text, batch_result in zip(TEN_TEXTS, batch_results):
        single_result = run_heads_decision(text, heads)
        for field in heads.schema:
            assert (
                batch_result["decisions"][field]["decision"]
                == single_result["decisions"][field]["decision"]
            ), f"order mismatch for {text!r} on field {field!r}"


@pytest.mark.requires_model
def test_run_heads_decision_batch_runs_exactly_one_encoder_forward_pass():
    from decision_heads import run_heads_decision_batch
    from jev_local_engine import _get_model_and_tokenizer

    heads = load_trained_heads()
    model, _ = _get_model_and_tokenizer()

    call_count = {"n": 0}
    original_forward = model.model.forward

    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_forward(*args, **kwargs)

    model.model.forward = counting_forward
    try:
        run_heads_decision_batch(TEN_TEXTS, heads)
    finally:
        model.model.forward = original_forward

    assert call_count["n"] == 1


@pytest.mark.requires_model
def test_run_heads_decision_batch_matches_single_within_tolerance():
    from decision_heads import run_heads_decision, run_heads_decision_batch

    heads = load_trained_heads()
    text = TEN_TEXTS[4]

    single_result = run_heads_decision(text, heads)
    batch_results = run_heads_decision_batch(TEN_TEXTS, heads)
    batch_result = batch_results[4]

    for field in heads.schema:
        assert (
            batch_result["decisions"][field]["decision"]
            == single_result["decisions"][field]["decision"]
        )
        single_probs = single_result["decisions"][field]["probabilities"]
        batch_probs = batch_result["decisions"][field]["probabilities"]
        for choice in single_probs:
            assert abs(single_probs[choice] - batch_probs[choice]) < 0.01, (
                f"{field}={choice!r} diverged: single={single_probs[choice]}, "
                f"batch={batch_probs[choice]}"
            )


@pytest.mark.requires_model
def test_run_heads_decision_batch_with_one_text_matches_single_input():
    from decision_heads import run_heads_decision, run_heads_decision_batch

    heads = load_trained_heads()
    text = TEN_TEXTS[1]

    single_result = run_heads_decision(text, heads)
    batch_results = run_heads_decision_batch([text], heads)

    assert len(batch_results) == 1
    batch_result = batch_results[0]

    for field in heads.schema:
        assert (
            batch_result["decisions"][field]["decision"]
            == single_result["decisions"][field]["decision"]
        )
        assert (
            batch_result["decisions"][field]["probabilities"]
            == single_result["decisions"][field]["probabilities"]
        )


@pytest.mark.requires_model
def test_route_tool_batch_runs_exactly_one_encoder_forward_pass():
    from decision_heads import route_tool_batch
    from jev_local_engine import _get_model_and_tokenizer

    gate = load_trained_gate_heads()
    tool_heads = load_trained_tool_heads()
    model, _ = _get_model_and_tokenizer()

    call_count = {"n": 0}
    original_forward = model.model.forward

    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_forward(*args, **kwargs)

    model.model.forward = counting_forward
    try:
        route_tool_batch(TEN_TEXTS, tool_heads, gate)
    finally:
        model.model.forward = original_forward

    assert call_count["n"] == 1


@pytest.mark.requires_model
def test_route_tool_batch_rejected_text_abstains_alongside_a_genuine_request():
    from decision_heads import route_tool_batch

    gate = load_trained_gate_heads()
    tool_heads = load_trained_tool_heads()

    results = route_tool_batch(
        [SAMPLE_REJECTED_TEXT, SAMPLE_TOOL_REQUEST], tool_heads, gate
    )

    rejected_result, genuine_result = results

    assert rejected_result["is_tool_request"] is False
    assert rejected_result["tool"] is None
    assert rejected_result["top_k"] == []

    assert genuine_result["is_tool_request"] is True
    assert genuine_result["tool"] is not None


@pytest.mark.requires_model
def test_route_tool_batch_matches_single_routing():
    from decision_heads import route_tool, route_tool_batch

    gate = load_trained_gate_heads()
    tool_heads = load_trained_tool_heads()

    single_result = route_tool(SAMPLE_TOOL_REQUEST, tool_heads, gate)
    batch_results = route_tool_batch(TEN_TEXTS + [SAMPLE_TOOL_REQUEST], tool_heads, gate)
    batch_result = batch_results[-1]

    assert batch_result["is_tool_request"] == single_result["is_tool_request"]
    assert batch_result["tool"] == single_result["tool"]


@pytest.mark.requires_model
def test_run_jev_decision_batch_returns_ten_results_in_order():
    from jev_local_engine import SCHEMA, run_jev_decision, run_jev_decision_batch

    batch_results = run_jev_decision_batch(TEN_TEXTS, SCHEMA)
    assert len(batch_results) == 10

    for text, batch_result in zip(TEN_TEXTS, batch_results):
        single_result = run_jev_decision(text, SCHEMA)
        for field in SCHEMA:
            assert (
                batch_result["decisions"][field]["decision"]
                == single_result["decisions"][field]["decision"]
            ), f"order mismatch for {text!r} on field {field!r}"


@pytest.mark.requires_model
def test_run_jev_decision_batch_matches_single_within_tolerance():
    from jev_local_engine import SCHEMA, run_jev_decision, run_jev_decision_batch

    text = TEN_TEXTS[4]

    single_result = run_jev_decision(text, SCHEMA)
    batch_results = run_jev_decision_batch(TEN_TEXTS, SCHEMA)
    batch_result = batch_results[4]

    for field in SCHEMA:
        assert (
            batch_result["decisions"][field]["decision"]
            == single_result["decisions"][field]["decision"]
        )
        single_probs = single_result["decisions"][field]["probabilities"]
        batch_probs = batch_result["decisions"][field]["probabilities"]
        for choice in single_probs:
            assert abs(single_probs[choice] - batch_probs[choice]) < 0.01, (
                f"{field}={choice!r} diverged: single={single_probs[choice]}, "
                f"batch={batch_probs[choice]}"
            )


def test_no_batch_entry_point_exists_for_full_sequence_path():
    # score_fields_full_sequence needs 1.2 GB of log-probabilities at ten
    # texts and grows with the batch, so this change deliberately leaves it
    # unbatched (batch-constrained-scoring's non-goal). Guard against a
    # batch entry point being added for it, and against the single-input
    # function's signature drifting.
    import inspect

    import jev_local_engine

    assert not hasattr(jev_local_engine, "score_fields_full_sequence_batch")
    for name in dir(jev_local_engine):
        assert not ("full_sequence" in name and "batch" in name), (
            f"unexpected batch entry point found for the full-sequence path: {name}"
        )

    signature = inspect.signature(jev_local_engine.score_fields_full_sequence)
    assert list(signature.parameters) == ["model", "tokenizer", "text", "schema"]


def test_benchmark_holds_ten_varied_prompts():
    # No timing assertion here by design: a threshold would fail on a
    # loaded machine rather than on a real regression (see plan.md's
    # non-goals). This only checks the prompt set's shape and variety.
    from benchmark import PROMPTS

    assert len(PROMPTS) == 10
    assert len(set(PROMPTS)) == 10, "prompts must not repeat one string ten times"
