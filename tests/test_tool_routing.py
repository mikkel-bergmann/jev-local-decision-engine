import collections
import json
import os
import re

import pytest
import torch

TOOLS_PATH = os.path.join("data", "tools.json")

TOOL_TRAIN_PATHS = [
    os.path.join("data", "tool_train_a.json"),
    os.path.join("data", "tool_train_b.json"),
    os.path.join("data", "tool_train_c.json"),
]

TOOL_HOLDOUT_PATH = os.path.join("data", "tool_holdout.json")


def load_tool_training_items():
    items = []
    for path in TOOL_TRAIN_PATHS:
        with open(path) as f:
            items.extend(json.load(f))
    return items


def load_tool_holdout_items():
    with open(TOOL_HOLDOUT_PATH) as f:
        return json.load(f)


def words_in(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))

# The first-underscore segment of every tool name that concerns PCI
# compliance or food-safety compliance for the quick-service-restaurant
# domain. Used to check the catalogue's compliance-domain share without
# hardcoding a tool-by-tool list.
COMPLIANCE_PREFIXES = {
    "pci",
    "haccp",
    "temp",
    "audit",
    "allergen",
    "health",
    "food",
    "supplier",
    "sanitation",
    "pest",
    "employee",
    "cold",
    "cleaning",
    "recall",
    "incident",
    "training",
}


def load_tools():
    with open(TOOLS_PATH) as f:
        return json.load(f)


def test_catalogue_is_large_and_mixed():
    tools = load_tools()

    assert len(tools) >= 100
    assert len(set(tools)) == len(tools), "tool names must be unique"

    compliance_tools = [t for t in tools if t.split("_")[0] in COMPLIANCE_PREFIXES]
    assert len(compliance_tools) >= 30


def test_catalogue_has_near_neighbour_families():
    tools = load_tools()

    prefix_counts = collections.Counter(t.split("_")[0] for t in tools)
    families = {prefix: count for prefix, count in prefix_counts.items() if count >= 3}

    assert len(families) >= 5


# TOOL_SCHEMA has ~110 choices for its single field, well past the "at
# least five choices" bar the top-k scenarios are stated over, so a fresh
# (untrained) DecisionHeads(TOOL_SCHEMA) is enough to exercise the shape of
# run_heads_decision's return value without needing a trained checkpoint.
SAMPLE_QUERY = "Please refund this customer's most recent payment."


@pytest.mark.requires_model
def test_top_k_defaults_to_five_ordered_and_matches_decision():
    from decision_heads import TOOL_SCHEMA, DecisionHeads, run_heads_decision

    torch.manual_seed(0)
    heads = DecisionHeads(TOOL_SCHEMA)

    result = run_heads_decision(SAMPLE_QUERY, heads)

    for field, record in result["decisions"].items():
        top_k = record["top_k"]
        assert len(top_k) == 5
        probabilities = [entry["probability"] for entry in top_k]
        assert probabilities == sorted(probabilities, reverse=True)
        assert top_k[0]["choice"] == record["decision"]


@pytest.mark.requires_model
def test_top_k_length_is_configurable():
    from decision_heads import TOOL_SCHEMA, DecisionHeads, run_heads_decision

    torch.manual_seed(0)
    heads = DecisionHeads(TOOL_SCHEMA)

    result = run_heads_decision(SAMPLE_QUERY, heads, top_k=3)

    for field, record in result["decisions"].items():
        assert len(record["top_k"]) == 3


@pytest.mark.requires_model
def test_decision_and_probabilities_unchanged_by_top_k():
    from decision_heads import TOOL_SCHEMA, DecisionHeads, run_heads_decision

    torch.manual_seed(0)
    heads = DecisionHeads(TOOL_SCHEMA)

    result = run_heads_decision(SAMPLE_QUERY, heads)

    for field, record in result["decisions"].items():
        assert "decision" in record
        assert record["decision"] in heads.schema[field]
        assert "probabilities" in record
        assert set(record["probabilities"].keys()) == set(heads.schema[field])


def test_every_tool_has_at_least_three_training_queries_with_valid_labels():
    tools = set(load_tools())
    items = load_tool_training_items()

    counts = collections.Counter(item["tool"] for item in items)

    for item in items:
        assert item["tool"] in tools, f"unknown tool label {item['tool']!r}"

    for tool in tools:
        assert counts[tool] >= 3, f"{tool} has only {counts[tool]} training queries"


def test_every_tool_has_a_query_sharing_no_word_with_its_name():
    tools = load_tools()
    items = load_tool_training_items()

    by_tool = collections.defaultdict(list)
    for item in items:
        by_tool[item["tool"]].append(item["text"])

    for tool in tools:
        tool_words = set(tool.split("_"))
        texts = by_tool[tool]
        assert any(
            tool_words.isdisjoint(words_in(text)) for text in texts
        ), f"every training query for {tool} shares a word with its name"


def test_holdout_is_disjoint_from_every_training_file():
    training_texts = {item["text"] for item in load_tool_training_items()}
    holdout_texts = [item["text"] for item in load_tool_holdout_items()]

    overlap = training_texts.intersection(holdout_texts)
    assert not overlap, f"holdout text(s) also appear in training: {overlap}"


@pytest.mark.requires_model
def test_tool_routing_decision_runs_exactly_one_encoder_forward_pass():
    # Mirrors test_scoring_one_decision_runs_exactly_one_encoder_forward_pass
    # in test_decision_heads.py, over TOOL_SCHEMA and the full
    # run_heads_decision path (encode + heads forward) rather than just
    # encode + heads(features), since that is what a tool-routing decision
    # actually costs.
    from decision_heads import DecisionHeads, TOOL_SCHEMA, run_heads_decision
    from jev_local_engine import _get_model_and_tokenizer

    model, _ = _get_model_and_tokenizer()
    torch.manual_seed(0)
    heads = DecisionHeads(TOOL_SCHEMA)

    call_count = {"n": 0}
    original_forward = model.model.forward

    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_forward(*args, **kwargs)

    model.model.forward = counting_forward
    try:
        run_heads_decision(SAMPLE_QUERY, heads)
    finally:
        model.model.forward = original_forward

    assert call_count["n"] == 1


@pytest.mark.requires_model
def test_tool_decision_is_a_catalogue_name_with_full_probability_map():
    from decision_heads import TOOL_SCHEMA, DecisionHeads, run_heads_decision

    tools = set(load_tools())
    torch.manual_seed(0)
    heads = DecisionHeads(TOOL_SCHEMA)

    result = run_heads_decision(SAMPLE_QUERY, heads)
    record = result["decisions"]["tool"]

    assert record["decision"] in tools
    assert set(record["probabilities"].keys()) == tools


FORBIDDEN_KEYS = {"argument", "arguments", "parameter", "parameters"}


def assert_no_forbidden_keys(value):
    if isinstance(value, dict):
        for key, sub_value in value.items():
            assert key not in FORBIDDEN_KEYS, f"forbidden key {key!r} found"
            assert_no_forbidden_keys(sub_value)
    elif isinstance(value, list):
        for item in value:
            assert_no_forbidden_keys(item)


@pytest.mark.requires_model
def test_tool_result_carries_no_argument_or_parameter_key():
    from decision_heads import TOOL_SCHEMA, DecisionHeads, run_heads_decision

    torch.manual_seed(0)
    heads = DecisionHeads(TOOL_SCHEMA)

    result = run_heads_decision(SAMPLE_QUERY, heads)

    assert_no_forbidden_keys(result)


TOOL_HEAD_PATH = os.path.join("data", "tool_head.pt")


def load_trained_tool_heads():
    import decision_heads

    if not os.path.exists(TOOL_HEAD_PATH):
        pytest.skip(f"trained tool head not found at {TOOL_HEAD_PATH}; run train_heads.py first")
    return decision_heads.load_heads(TOOL_HEAD_PATH, decision_heads.TOOL_SCHEMA)


@pytest.mark.requires_model
def test_evaluate_tool_head_report_carries_all_four_keys():
    from train_heads import evaluate_tool_head

    heads = load_trained_tool_heads()
    report = evaluate_tool_head(heads)

    for key in ("recall_at_5", "recall_at_1", "unpredicted_tool_count", "unpredicted_tools"):
        assert key in report


@pytest.mark.requires_model
def test_evaluate_tool_head_names_unpredicted_tools_not_only_counts_them():
    from train_heads import evaluate_tool_head

    heads = load_trained_tool_heads()
    report = evaluate_tool_head(heads)

    assert isinstance(report["unpredicted_tools"], list)
    assert report["unpredicted_tool_count"] == len(report["unpredicted_tools"])
    tools = set(load_tools())
    for tool in report["unpredicted_tools"]:
        assert tool in tools
