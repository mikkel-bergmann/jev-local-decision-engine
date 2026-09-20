import json
import os
import re
from collections import Counter

import pytest
import torch

GATE_NEGATIVES_PATHS = [
    os.path.join("data", "gate_negatives_a.json"),
    os.path.join("data", "gate_negatives_b.json"),
]

GATE_HOLDOUT_PATH = os.path.join("data", "gate_holdout.json")

GATE_ADVERSARIAL_PATH = os.path.join("data", "gate_adversarial.json")

GATE_HEAD_PATH = os.path.join("data", "gate_head.pt")

SAMPLE_TOOL_REQUEST = "Please refund this customer's most recent payment."

TOOL_TRAIN_PATHS = [
    os.path.join("data", "tool_train_a.json"),
    os.path.join("data", "tool_train_b.json"),
    os.path.join("data", "tool_train_c.json"),
    os.path.join("data", "tool_train_d.json"),
]


def load_gate_negatives():
    items = []
    for path in GATE_NEGATIVES_PATHS:
        with open(path) as f:
            items.extend(json.load(f))
    return items


def load_gate_holdout():
    with open(GATE_HOLDOUT_PATH) as f:
        return json.load(f)


def load_tool_training_items():
    items = []
    for path in TOOL_TRAIN_PATHS:
        with open(path) as f:
            items.extend(json.load(f))
    return items


def four_word_phrases(text):
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i : i + 4]) for i in range(len(words) - 3)}


def test_gate_negatives_are_numerous():
    negatives = load_gate_negatives()
    assert len(negatives) >= 150


def test_no_fixed_four_word_phrase_dominates_the_negatives():
    negatives = load_gate_negatives()
    phrase_counts = Counter()
    for item in negatives:
        for phrase in four_word_phrases(item["text"]):
            phrase_counts[phrase] += 1

    threshold = len(negatives) * 0.1
    offenders = {
        phrase: count for phrase, count in phrase_counts.items() if count > threshold
    }
    assert not offenders, f"phrase(s) reused too often: {offenders}"


def test_gate_holdout_is_disjoint_from_all_gate_training_data():
    negatives = load_gate_negatives()
    tool_items = load_tool_training_items()

    training_texts = {item["text"] for item in negatives} | {
        item["text"] for item in tool_items
    }
    holdout_texts = {item["text"] for item in load_gate_holdout()}

    overlap = training_texts & holdout_texts
    assert not overlap, f"holdout text(s) also appear in gate training data: {overlap}"


def load_trained_gate_heads():
    import decision_heads

    if not os.path.exists(GATE_HEAD_PATH):
        pytest.skip(f"trained gate head not found at {GATE_HEAD_PATH}; run train_heads.py first")
    return decision_heads.load_heads(GATE_HEAD_PATH, decision_heads.GATE_SCHEMA)


@pytest.mark.requires_model
def test_shipped_gate_head_accepts_a_tool_request_and_rejects_clear_negatives():
    from decision_heads import run_heads_decision

    heads = load_trained_gate_heads()

    def verdict(text):
        result = run_heads_decision(text, heads)
        return result["decisions"]["is_tool_request"]["decision"]

    assert verdict(SAMPLE_TOOL_REQUEST) == "yes"
    assert verdict("hello") == "no"
    assert verdict("what is 2 + 2") == "no"
    assert verdict("can you write me a poem about the sea") == "no"


@pytest.mark.requires_model
def test_gate_and_tool_heads_share_one_encoder_forward_pass():
    # A gated routing decision scores both the gate head and the tool head
    # off the same encode() call, so it should cost exactly one encoder
    # forward pass — mirrors
    # test_tool_routing_decision_runs_exactly_one_encoder_forward_pass, but
    # over both heads together rather than one.
    from decision_heads import TOOL_SCHEMA, DecisionHeads, encode
    from jev_local_engine import _get_model_and_tokenizer

    model, _ = _get_model_and_tokenizer()
    gate_heads = load_trained_gate_heads()
    torch.manual_seed(0)
    tool_heads = DecisionHeads(TOOL_SCHEMA)

    call_count = {"n": 0}
    original_forward = model.model.forward

    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_forward(*args, **kwargs)

    model.model.forward = counting_forward
    try:
        features = encode([SAMPLE_TOOL_REQUEST])
        with torch.no_grad():
            gate_heads(features)
            tool_heads(features)
    finally:
        model.model.forward = original_forward

    assert call_count["n"] == 1


@pytest.mark.requires_model
def test_threshold_sweep_reports_both_directions_for_every_candidate():
    from train_heads import GATE_THRESHOLD_CANDIDATES, choose_gate_threshold

    gate = load_trained_gate_heads()
    result = choose_gate_threshold(gate)

    assert len(result["sweep"]) == len(GATE_THRESHOLD_CANDIDATES)
    swept_thresholds = [record["threshold"] for record in result["sweep"]]
    assert swept_thresholds == GATE_THRESHOLD_CANDIDATES

    for record in result["sweep"]:
        assert "negative_rejection_rate" in record
        assert "genuine_retention_rate" in record
        assert 0.0 <= record["negative_rejection_rate"] <= 1.0
        assert 0.0 <= record["genuine_retention_rate"] <= 1.0


@pytest.mark.requires_model
def test_gate_threshold_lies_within_swept_range():
    from decision_heads import GATE_THRESHOLD
    from train_heads import GATE_THRESHOLD_CANDIDATES

    assert min(GATE_THRESHOLD_CANDIDATES) <= GATE_THRESHOLD <= max(GATE_THRESHOLD_CANDIDATES)


TOOL_HEAD_PATH = os.path.join("data", "tool_head.pt")


def load_trained_tool_heads():
    import decision_heads

    if not os.path.exists(TOOL_HEAD_PATH):
        pytest.skip(f"trained tool head not found at {TOOL_HEAD_PATH}; run train_heads.py first")
    return decision_heads.load_heads(TOOL_HEAD_PATH, decision_heads.TOOL_SCHEMA)


@pytest.mark.requires_model
def test_route_tool_on_rejected_query_reports_no_tool():
    from decision_heads import route_tool

    gate = load_trained_gate_heads()
    tool_heads = load_trained_tool_heads()

    result = route_tool("hello", tool_heads, gate)

    assert result["is_tool_request"] is False
    assert result["tool"] is None
    assert result["top_k"] == []
    assert "gate_probability" in result


@pytest.mark.requires_model
def test_route_tool_on_accepted_query_matches_tool_head_alone():
    from decision_heads import route_tool, run_heads_decision

    gate = load_trained_gate_heads()
    tool_heads = load_trained_tool_heads()

    result = route_tool(SAMPLE_TOOL_REQUEST, tool_heads, gate)
    tool_only = run_heads_decision(SAMPLE_TOOL_REQUEST, tool_heads)["decisions"]["tool"]

    assert result["is_tool_request"] is True
    assert result["tool"] == tool_only["decision"]
    assert result["top_k"] == tool_only["top_k"]
    assert "gate_probability" in result


def load_gate_adversarial_items():
    with open(GATE_ADVERSARIAL_PATH) as f:
        return json.load(f)


@pytest.mark.requires_model
def test_shipped_gate_meets_the_measured_adversarial_bar():
    # Guards the measured bar recorded in decision_heads.GATE_THRESHOLD's
    # comment (43/45 adversarial negatives rejected, 24/25 genuine
    # incident-logging requests retained at threshold 0.55) so a future
    # retrain that regresses it fails the suite instead of passing
    # unnoticed — the scenario was previously enforced only by prose and
    # manual reporting. Loads the shipped data/gate_head.pt at the shipped
    # GATE_THRESHOLD and scores data/gate_adversarial.json directly; skips
    # cleanly (via load_trained_gate_heads) when the checkpoint is absent.
    from decision_heads import GATE_THRESHOLD, run_heads_decision

    gate = load_trained_gate_heads()
    adversarial = load_gate_adversarial_items()

    neg_total = neg_rejected = 0
    pos_total = pos_retained = 0
    for item in adversarial:
        result = run_heads_decision(item["text"], gate)
        probability = result["decisions"]["is_tool_request"]["probabilities"]["yes"]
        accepted = probability >= GATE_THRESHOLD

        if item["is_tool_request"]:
            pos_total += 1
            if accepted:
                pos_retained += 1
        else:
            neg_total += 1
            if not accepted:
                neg_rejected += 1

    negative_rejection_rate = neg_rejected / neg_total
    genuine_retention_rate = pos_retained / pos_total

    assert negative_rejection_rate >= 0.90, (
        f"adversarial negative-rejection rate fell to {neg_rejected}/{neg_total} "
        f"({negative_rejection_rate:.4f}), below the required 0.90"
    )
    assert genuine_retention_rate >= 0.90, (
        f"adversarial genuine-retention rate fell to {pos_retained}/{pos_total} "
        f"({genuine_retention_rate:.4f}), below the required 0.90"
    )
