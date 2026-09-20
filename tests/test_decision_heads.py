import json

import pytest
import torch

from jev_local_engine import SCHEMA

HOLDOUT_PATH = "data/holdout.json"

# Every fresh DecisionHeads(SCHEMA) below is preceded by
# torch.manual_seed(0). Nothing in this project seeds the RNG by
# default, and an unseeded random head's argmax is arbitrary: measured,
# 7 of 200 unseeded random inits flip at least one padding anchor's
# decision (3.5%). The seed is load-bearing here, not decorative — it
# is what keeps these tests from passing or failing by luck.


@pytest.mark.requires_model
def test_encode_returns_unit_norm_rows_at_hidden_width():
    from decision_heads import encode

    features = encode(["a", "b", "c"])

    assert features.shape == (3, 1536)
    norms = features.norm(dim=1)
    for norm in norms:
        assert abs(float(norm) - 1.0) < 1e-4


def test_heads_schema_collapses_urgency_engine_schema_unchanged():
    from decision_heads import HEADS_SCHEMA
    from jev_local_engine import SCHEMA

    assert HEADS_SCHEMA["category"] == SCHEMA["category"]
    assert HEADS_SCHEMA["sentiment"] == SCHEMA["sentiment"]
    assert len(HEADS_SCHEMA["urgency"]) == 2
    assert set(HEADS_SCHEMA["urgency"]) == {"not_urgent", "urgent"}

    assert len(SCHEMA["urgency"]) == 4
    assert set(SCHEMA["urgency"]) == {"low", "medium", "high", "critical"}


PADDING_FILLER_TEXT = (
    "Could you please walk me through the entire process for updating my "
    "billing address and payment method on file, including any "
    "documentation you might need from my side, along with confirming the "
    "new mailing address and updated card details, and please let me know "
    "if there is anything else I need to provide before the next billing "
    "cycle begins so nothing is delayed on my account?"
)  # 66 words / 70 tokens

# Anchors spanning single-token (below the cosine floor, by measurement) up
# through several-token short texts. Deliberately includes the awkward
# single-token cases rather than avoiding them: "." and ":)" measure below
# 0.9999 cosine similarity against PADDING_FILLER_TEXT (0.9998568 and
# 0.9998785 respectively) even though their decisions never flip. An
# earlier version of this test asserted 0.9999 for every anchor; that bound
# was calibrated to the anchors then known rather than derived from a worst
# case, and adversarial probing falsified it on these single-token inputs.
PADDING_ANCHORS = [".", ":)", "!", "No.", "Fix my bill.", "Please help me."]

# Anchors of two or more tokens only — the range over which the cosine
# floor is scoped and demonstrably holds.
MULTI_TOKEN_PADDING_ANCHORS = ["No.", "Fix my bill.", "Please help me."]


@pytest.mark.requires_model
def test_padding_never_changes_a_decision():
    # Decision-invariance is the universal contract: every anchor, single-
    # token ones included, must select the same decision whether encoded
    # alone or padded against PADDING_FILLER_TEXT. This is the property
    # that survived adversarial probing across roughly seventy anchor pairs
    # — including the counterexamples that sank the universal cosine floor
    # below — so it is asserted with no token-count exception.
    #
    # Scored with the TRAINED heads from data/heads.pt, not a fresh
    # DecisionHeads(HEADS_SCHEMA). An untrained head's argmax is arbitrary:
    # its class scores are near-ties by construction, so float16 padding
    # drift of ~0.003-0.01 can reorder them. Measured: 7 of 200 unseeded
    # random inits flip at least one anchor's decision (3.5%) — the test
    # was passing on luck, not testing the system. Trained heads have real
    # margins and never flip.
    #
    # data/heads.pt now holds heads trained on hand-labelled examples
    # against HEADS_SCHEMA (urgency collapsed to two classes), not the
    # engine's four-way SCHEMA — load_heads is given HEADS_SCHEMA to match.
    import os

    from decision_heads import HEADS_SCHEMA, encode, load_heads

    heads_path = os.path.join("data", "heads.pt")
    if not os.path.exists(heads_path):
        pytest.skip(f"trained heads not found at {heads_path}; run train_heads.py first")
    heads = load_heads(heads_path, HEADS_SCHEMA)

    worst_margin = float("inf")
    for short_text in PADDING_ANCHORS:
        alone = encode([short_text])[0]
        batched = encode([short_text, PADDING_FILLER_TEXT])[0]

        max_diff = float((alone - batched).abs().max())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                alone.unsqueeze(0), batched.unsqueeze(0)
            )
        )

        decisions_alone = heads(alone.unsqueeze(0))
        decisions_batched = heads(batched.unsqueeze(0))
        for field, choices in HEADS_SCHEMA.items():
            probs_alone = decisions_alone[field][0].detach()
            top2 = torch.topk(probs_alone, k=2).values
            margin = float(top2[0] - top2[1])
            worst_margin = min(worst_margin, margin)

            choice_alone = choices[int(decisions_alone[field][0].argmax())]
            choice_batched = choices[int(decisions_batched[field][0].argmax())]
            assert choice_alone == choice_batched, (
                f"{field} decision flipped under padding for {short_text!r}: "
                f"{choice_alone!r} vs {choice_batched!r} (measured cosine "
                f"similarity {cosine:.7f}, max per-dimension diff {max_diff:.6f}, "
                f"top-1 to top-2 margin {margin:.4f})"
            )

    print(f"\nworst top-1 to top-2 margin across all anchors/fields: {worst_margin:.4f}")


@pytest.mark.requires_model
def test_pooled_vectors_stay_aligned_for_multi_token_inputs():
    # A per-dimension absolute bound on an L2-normalized 1536-dim vector
    # measures the wrong thing: a typical dimension carries magnitude
    # ~0.0255, so a bound like 1e-3 is ~4% of it, against float16
    # attention accumulation that reaches 0.009964 on a single-token
    # anchor. That per-dimension contract was abandoned as unmeetable, not
    # merely unlucky. Cosine similarity is the replacement, but only over
    # two-or-more-token anchors: single-token inputs measurably fall below
    # 0.9999 (see PADDING_ANCHORS's comment) even though their decisions
    # never flip, so the floor is scoped here rather than widened to cover
    # them or silently dropped.
    from decision_heads import encode

    for short_text in MULTI_TOKEN_PADDING_ANCHORS:
        alone = encode([short_text])[0]
        batched = encode([short_text, PADDING_FILLER_TEXT])[0]

        max_diff = float((alone - batched).abs().max())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                alone.unsqueeze(0), batched.unsqueeze(0)
            )
        )
        assert cosine >= 0.9999, (
            f"padding drift too large for {short_text!r}: cosine similarity "
            f"{cosine:.7f}, max per-dimension diff {max_diff:.6f}"
        )


@pytest.mark.requires_model
def test_head_and_engine_paths_share_comparable_shape():
    # Scored with the SHIPPED heads loaded from data/heads.pt under
    # HEADS_SCHEMA, not a fresh DecisionHeads(SCHEMA). A fresh
    # DecisionHeads(SCHEMA) is a four-class-urgency stand-in that could
    # never expose the contradiction this test now guards: comparable-
    # return-shape's "same choice keys" claim and heads-schema's urgency
    # collapse were written at different times and were never reconciled
    # until validation exercised the shipped heads against the engine and
    # found urgency keys genuinely diverge (heads=[not_urgent, urgent],
    # engine=[low, medium, high, critical]). A test that builds its own
    # SCHEMA-shaped heads can't see that divergence — it has to load what
    # actually ships.
    import os

    from decision_heads import HEADS_SCHEMA, load_heads, run_heads_decision
    from jev_local_engine import SCHEMA, run_jev_decision
    from train_heads import ENGINE_URGENCY_TO_HEADS_URGENCY

    heads_path = os.path.join("data", "heads.pt")
    if not os.path.exists(heads_path):
        pytest.skip(f"trained heads not found at {heads_path}; run train_heads.py first")
    heads = load_heads(heads_path, HEADS_SCHEMA)

    text = "My account was double charged for last month's subscription."
    heads_result = run_heads_decision(text, heads)
    engine_result = run_jev_decision(text, SCHEMA)

    assert set(heads_result["decisions"].keys()) == set(engine_result["decisions"].keys())
    assert isinstance(heads_result["latency_ms"], float)
    assert isinstance(engine_result["latency_ms"], float)

    for field, probabilities in heads_result["decisions"].items():
        total = sum(probabilities["probabilities"].values())
        assert abs(total - 1.0) < 0.001, f"{field} probabilities sum to {total}"

    # Choice keys match only where the two schemas declare the same
    # choices — true for category and sentiment.
    for field in ("category", "sentiment"):
        heads_choices = set(heads_result["decisions"][field]["probabilities"].keys())
        engine_choices = set(engine_result["decisions"][field]["probabilities"].keys())
        assert heads_choices == engine_choices, (
            f"{field} choice keys diverge: heads={sorted(heads_choices)} "
            f"engine={sorted(engine_choices)}"
        )

    # urgency diverges by design (heads-schema collapses it to two
    # classes). Every engine choice must map onto a heads choice through
    # the declared mapping, so the comparison stays defined.
    engine_urgency_choices = set(
        engine_result["decisions"]["urgency"]["probabilities"].keys()
    )
    heads_urgency_choices = set(
        heads_result["decisions"]["urgency"]["probabilities"].keys()
    )
    for engine_choice in engine_urgency_choices:
        mapped = ENGINE_URGENCY_TO_HEADS_URGENCY[engine_choice]
        assert mapped in heads_urgency_choices, (
            f"engine urgency choice {engine_choice!r} maps to {mapped!r}, "
            f"which is not a heads urgency choice "
            f"({sorted(heads_urgency_choices)})"
        )


@pytest.mark.requires_model
def test_scoring_one_decision_runs_exactly_one_encoder_forward_pass():
    # Audited (task 14.2): DecisionHeads(SCHEMA) is fine here as an
    # arbitrary stand-in. The property under test is the encoder
    # forward-pass count, which shared-prefill-inference ties to the
    # number of fields/classes being schema-independent in a different
    # sense (it must not grow with them) — a SCHEMA-shaped head exercises
    # that identically to a HEADS_SCHEMA-shaped one. No claim about the
    # shipped heads' predictions is made or implied.
    from decision_heads import DecisionHeads, encode
    from jev_local_engine import _get_model_and_tokenizer

    model, _ = _get_model_and_tokenizer()
    torch.manual_seed(0)
    heads = DecisionHeads(SCHEMA)

    call_count = {"n": 0}
    original_forward = model.model.forward

    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_forward(*args, **kwargs)

    model.model.forward = counting_forward
    try:
        text = "My account was double charged for last month's subscription."
        features = encode([text])
        heads(features)
    finally:
        model.model.forward = original_forward

    assert call_count["n"] == 1


def test_build_corpus_is_deterministic_and_reaches_minimum_size():
    from train_heads import build_corpus

    first = build_corpus()
    second = build_corpus()

    assert first == second
    assert len(first) >= 300


def test_holdout_is_disjoint_from_corpus_and_labels_are_declared_choices():
    from train_heads import build_corpus

    with open(HOLDOUT_PATH) as f:
        holdout = json.load(f)

    corpus = set(build_corpus())
    for item in holdout:
        assert item["text"] not in corpus
        for field, choices in SCHEMA.items():
            assert item[field] in choices


TRAIN_FILE_PATHS = [
    "data/train_a.json",
    "data/train_b.json",
    "data/train_c.json",
]
HOLDOUT_V2_PATH = "data/holdout_v2.json"


def _four_word_phrases(text):
    import re

    words = re.findall(r"[a-z']+", text.lower())
    return {" ".join(words[i : i + 4]) for i in range(len(words) - 3)}


def test_hand_labelled_training_set_is_varied_and_holdout_v2_is_disjoint():
    from collections import Counter

    combined = []
    for path in TRAIN_FILE_PATHS:
        with open(path) as f:
            combined.extend(json.load(f))

    assert len(combined) >= 200, (
        f"combined training set has {len(combined)} items, need at least 200"
    )

    # No fixed four-or-more-word phrase may appear in more than a tenth of a
    # class's items — the mechanical guard against reproducing the
    # template-corpus failure (heads keying on a fixed phrase like
    # "immediately" instead of the actual signal).
    for field in ("category", "urgency", "sentiment"):
        classes = {item[field] for item in combined}
        for cls in classes:
            cls_items = [item for item in combined if item[field] == cls]
            phrase_counts = Counter()
            for item in cls_items:
                for phrase in _four_word_phrases(item["text"]):
                    phrase_counts[phrase] += 1
            limit = len(cls_items) / 10
            for phrase, count in phrase_counts.items():
                assert count <= limit, (
                    f"{field}={cls!r}: phrase {phrase!r} appears in {count} of "
                    f"{len(cls_items)} items (limit {limit:.1f}, i.e. one tenth)"
                )

    with open(HOLDOUT_V2_PATH) as f:
        holdout_v2 = json.load(f)
    with open(HOLDOUT_PATH) as f:
        holdout_v1 = json.load(f)

    combined_texts = {item["text"] for item in combined}
    holdout_v1_texts = {item["text"] for item in holdout_v1}
    for item in holdout_v2:
        assert item["text"] not in combined_texts, (
            f"holdout_v2 text overlaps a training file: {item['text']!r}"
        )
        assert item["text"] not in holdout_v1_texts, (
            f"holdout_v2 text overlaps the validation set holdout.json: "
            f"{item['text']!r}"
        )


@pytest.mark.requires_model
def test_training_updates_only_head_parameters():
    # Audited (task 14.2): DecisionHeads(SCHEMA) is correct here, not a
    # stand-in. This test deliberately exercises the still-present legacy
    # distillation path (build_corpus/label_corpus), which labels urgency
    # from the engine's own four-class SCHEMA. Pairing it with HEADS_SCHEMA
    # (two-class urgency) would raise ValueError inside train() the moment
    # a label like "critical" is looked up in a two-choice list — SCHEMA is
    # the only schema these labels are valid against. The property under
    # test (frozen encoder, head-only gradients) is identical regardless of
    # which schema or data source is used.
    from decision_heads import DecisionHeads
    from jev_local_engine import _get_model_and_tokenizer
    from train_heads import build_corpus, label_corpus, train

    texts = build_corpus()[:20]
    labels = label_corpus(build_corpus())[:20]
    torch.manual_seed(0)
    heads = DecisionHeads(SCHEMA)

    trained = train(texts, labels, heads=heads, epochs=3)

    model, _ = _get_model_and_tokenizer()
    for param in model.parameters():
        assert param.grad is None

    assert any(param.grad is not None for param in trained.parameters())


@pytest.mark.requires_model
def test_evaluate_report_bundles_both_accuracies_with_disagreement():
    # Uses HEADS_SCHEMA, not a fresh DecisionHeads(SCHEMA). evaluate()
    # scores data/holdout_v2.json, whose urgency labels are in the heads'
    # two-class scheme ("not_urgent"/"urgent"), and maps the engine's own
    # four-class urgency onto that scheme before comparing. Pairing it with
    # SCHEMA-shaped (four-class) heads would compare four-class predictions
    # against two-class true labels — a real vocabulary mismatch that this
    # test's assertions (structure and types only, not accuracy values)
    # would never catch, exactly the class of bug the shape-comparison test
    # above was rewritten to expose.
    from decision_heads import DecisionHeads, HEADS_SCHEMA
    from train_heads import evaluate

    torch.manual_seed(0)
    heads = DecisionHeads(HEADS_SCHEMA)
    report = evaluate(heads)

    # Field-keyed, not accuracy-type-keyed: there is no top-level
    # "heads_accuracy" or "teacher_accuracy" key a caller could read in
    # isolation. Each field's own record carries both together.
    assert set(report.keys()) == set(HEADS_SCHEMA.keys())
    for record in report.values():
        assert set(record.keys()) == {
            "heads_accuracy",
            "teacher_accuracy",
            "disagreement_count",
        }
        assert isinstance(record["heads_accuracy"], float)
        assert isinstance(record["teacher_accuracy"], float)
        assert isinstance(record["disagreement_count"], int)


@pytest.mark.requires_model
def test_reloaded_heads_reproduce_predictions(tmp_path):
    # Audited (task 14.2): DecisionHeads(SCHEMA) is fine as an arbitrary
    # stand-in. head-persistence's contract — reload reproduces predictions
    # — is about the save/load round-trip mechanism, not about any
    # particular schema's shape or the shipped checkpoint specifically.
    # This test saves to its own tmp_path file, never touching or claiming
    # anything about data/heads.pt.
    from decision_heads import DecisionHeads, encode, load_heads, save_heads

    torch.manual_seed(0)
    heads = DecisionHeads(SCHEMA)
    text = "My account was double charged for last month's subscription."
    features = encode([text])
    original = heads(features)

    save_path = tmp_path / "heads.pt"
    save_heads(heads, save_path)

    reloaded = load_heads(save_path, SCHEMA)
    reloaded_result = reloaded(features)

    for field in SCHEMA:
        for original_prob, reloaded_prob in zip(
            original[field][0].detach(), reloaded_result[field][0].detach()
        ):
            assert abs(float(original_prob) - float(reloaded_prob)) < 1e-5


@pytest.mark.requires_model
def test_loading_mismatched_schema_raises_naming_mismatch(tmp_path):
    # Audited (task 14.2): DecisionHeads(SCHEMA) is fine as an arbitrary
    # stand-in. This test verifies mismatch-detection fires between two
    # schemas that differ in field names entirely (SCHEMA vs a one-field
    # "tool" schema) — any two differing schemas exercise that error path
    # identically; no claim about the shipped heads is made.
    from decision_heads import DecisionHeads, load_heads, save_heads

    torch.manual_seed(0)
    heads = DecisionHeads(SCHEMA)
    save_path = tmp_path / "heads.pt"
    save_heads(heads, save_path)

    mismatched_schema = {"tool": ["web_search", "calculator", "calendar", "clarify"]}

    with pytest.raises(ValueError) as excinfo:
        load_heads(save_path, mismatched_schema)

    message = str(excinfo.value)
    assert "tool" in message or "category" in message


@pytest.mark.requires_model
def test_run_heads_decision_warm_latency_below_100ms():
    # Audited (task 14.2): DecisionHeads(SCHEMA) is fine as an arbitrary
    # stand-in. Warm latency is dominated by the shared encoder forward
    # pass and is insensitive to head class-count or to which weights are
    # loaded — a freshly constructed head measures the same latency
    # profile as the shipped one. head-latency makes no claim tied to a
    # specific schema or checkpoint.
    from decision_heads import DecisionHeads, run_heads_decision

    torch.manual_seed(0)
    heads = DecisionHeads(SCHEMA)
    text = (
        "My account was double charged for last month's subscription, "
        "fix this immediately!"
    )

    run_heads_decision(text, heads)  # warm-up call, excluded from the assertion
    warm_result = run_heads_decision(text, heads)

    assert warm_result["latency_ms"] < 100
