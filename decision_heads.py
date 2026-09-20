"""Distilled decision heads: one linear classification head per schema field,
trained on frozen-encoder features so every field resolves from a single
shared forward pass instead of one prompt per field.
"""

import json
import os
import time

import torch

import jev_local_engine

HIDDEN_SIZE = 1536

TOOLS_PATH = os.path.join("data", "tools.json")

# The heads' own schema: category and sentiment copied unchanged from the
# constrained engine's SCHEMA, but urgency collapsed to two classes. Four-way
# urgency was measured unlearnable from these features — it fits 60 hand
# labels at 1.000 train accuracy yet cross-validates at 0.200 against a
# 0.250 chance floor (perfect memorization, worse-than-guessing
# generalization). Collapsed to two classes the same features reach
# 0.667-0.700 against a 0.500 floor. jev_local_engine.SCHEMA is left
# untouched — it belongs to a separately merged capability and keeps its
# four urgency choices for the constrained-decoding path.
HEADS_SCHEMA = {
    "category": list(jev_local_engine.SCHEMA["category"]),
    "urgency": ["not_urgent", "urgent"],
    "sentiment": list(jev_local_engine.SCHEMA["sentiment"]),
}


def load_tool_catalogue(path=TOOLS_PATH):
    """Return the ordered tool names from `data/tools.json`.

    The catalogue is a flat JSON list of unique, namespaced tool names (see
    `tool-catalogue`). Order is preserved from the file so a schema built
    from it is deterministic.
    """
    with open(path) as f:
        return json.load(f)


# The tool-routing head's own schema: one field, "tool", whose choices are
# the full catalogue. Built the same way HEADS_SCHEMA is — a plain
# {field: choices} mapping — so the existing DecisionHeads class accepts it
# unchanged; it already builds one Linear(HIDDEN_SIZE, n) per field.
TOOL_SCHEMA = {"tool": load_tool_catalogue()}

# The gate head's own schema: one binary field answering whether a query is
# a tool request at all. Built the same way TOOL_SCHEMA and HEADS_SCHEMA
# are — a plain {field: choices} mapping the existing DecisionHeads class
# accepts unchanged.
GATE_SCHEMA = {"is_tool_request": ["no", "yes"]}

# The gate's decision threshold on its "yes" probability, chosen by
# train_heads.choose_gate_threshold sweeping 0.10 to 0.90 in steps of 0.05
# against data/gate_holdout.json (40 positives, 40 negatives) AND the
# evaluation-only data/gate_adversarial.json (45 narrative-incident
# negatives, 25 genuine incident-logging positives), per `gate-threshold`.
#
# The holdout alone ties at harmonic mean 1.0 across a 0.20-wide band
# (0.35 to 0.55) — it cannot discriminate inside that range at all. An
# earlier version of this sweep broke that tie by taking the lowest tying
# value (0.35) and shipped it; the validator then found that 0.35 sat
# inside the exact band where narrative past-tense incident negatives
# (e.g. "yeah that email thread got out of hand fast") were re-admitted,
# even though the retraining that produced this checkpoint had genuinely
# pushed their scores down. The tie-break, not the model, was the bug: it
# was choosing arbitrarily inside a range the holdout could not measure,
# and happened to land on the worst end for that failure class. The sweep
# now breaks any holdout tie on the adversarial negative-rejection rate
# instead of on threshold order, so the choice is measured all the way
# down rather than hand-picked at the last step.
#
# Re-measured 2026-09-20 against the shipped data/gate_head.pt: threshold
# 0.55 — precision 1.0, recall 1.0, and negative-rejection rate 1.0 on the
# holdout (tied with every other candidate from 0.35 to 0.55), with the
# highest adversarial negative-rejection rate among the tied candidates at
# 0.9556 (43/45), and adversarial genuine-retention rate 0.96 (24/25) at
# that threshold. Reproduced identically across independent retrains.
GATE_THRESHOLD = 0.55


class DecisionHeads(torch.nn.Module):
    """One linear classification head per schema field, sharing one feature
    tensor.

    Holds a `torch.nn.ModuleDict` mapping each field name to a
    `Linear(HIDDEN_SIZE, len(choices))`. `forward(features)` applies every
    head to the same `features` tensor and returns `{field: probabilities}`,
    each a softmax over that field's choices — so every field resolves from
    one shared encode, with no per-field re-encoding.
    """

    def __init__(self, schema):
        super().__init__()
        self.schema = schema
        self.heads = torch.nn.ModuleDict(
            {
                field: torch.nn.Linear(HIDDEN_SIZE, len(choices))
                for field, choices in schema.items()
            }
        )

    def forward(self, features):
        return {
            field: torch.softmax(head(features), dim=-1)
            for field, head in self.heads.items()
        }


def encode(texts, batch_size=16):
    """Encode `texts` into L2-normalized, mean-pooled feature vectors.

    Runs `model.model(**batch)` (the base transformer, not the LM head)
    under `torch.no_grad()`, takes the attention-mask-weighted mean of
    `last_hidden_state` so padding never contributes to the pooled vector,
    casts to float32, and L2-normalizes each row. Batches of `batch_size`
    keep memory bounded for larger corpora; the result is the same as
    encoding everything in one batch.

    Reuses the engine's cached model/tokenizer via
    `jev_local_engine._get_model_and_tokenizer()` rather than loading a
    second copy.

    Returns a tensor of shape (len(texts), hidden_size).
    """
    model, tokenizer = jev_local_engine._get_model_and_tokenizer()

    all_features = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        batch = tokenizer(
            chunk, padding=True, return_tensors="pt", add_special_tokens=False
        ).to(model.device)

        with torch.no_grad():
            outputs = model.model(**batch)
        last_hidden_state = outputs.last_hidden_state.float()

        attention_mask = batch["attention_mask"].unsqueeze(-1).float()
        summed = (last_hidden_state * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_features.append(normalized.cpu())

    return torch.cat(all_features, dim=0)


def run_heads_decision(text, heads, top_k=5):
    """Score `text` against `heads` and return the same shape `run_jev_decision`
    returns, so the two paths are directly comparable.

    Each field's record additionally carries a `top_k` list of the
    `top_k` highest-probability `{"choice", "probability"}` entries, ordered
    by descending probability, whose first entry equals that field's
    `decision`. `decision` and `probabilities` keep their previous meaning
    and values — `top_k` is purely additive.

    `latency_ms` times the encode plus the head forward only — the shared
    single pass this path exists to measure — excluding model load. Uses
    `torch.mps.synchronize()` around the timed region so MPS's asynchronous
    queue submission isn't mistaken for compute time.
    """
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    start = time.perf_counter()

    features = encode([text])
    with torch.no_grad():
        field_probabilities = heads(features)

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    decisions = {}
    for field, probabilities in field_probabilities.items():
        choices = heads.schema[field]
        probs_by_choice = {
            choice: float(prob) for choice, prob in zip(choices, probabilities[0])
        }
        best_choice = max(probs_by_choice, key=probs_by_choice.get)
        ranked = sorted(probs_by_choice.items(), key=lambda kv: kv[1], reverse=True)
        top_k_list = [
            {"choice": choice, "probability": prob} for choice, prob in ranked[:top_k]
        ]
        decisions[field] = {
            "decision": best_choice,
            "probabilities": probs_by_choice,
            "top_k": top_k_list,
        }

    return {"decisions": decisions, "latency_ms": latency_ms}


def run_heads_decision_batch(texts, heads, top_k=5):
    """Score every text in `texts` against `heads` in one encoder forward
    pass, returning one result per text, in input order.

    Each result has the same shape `run_heads_decision` returns for a single
    text — `{"decisions": {...}, "latency_ms": ...}` with `top_k` on every
    field. `latency_ms` is the batch's total encode-plus-heads time divided
    by `len(texts)`, so it reads as a comparable per-prompt figure rather
    than the batch total.

    Uses `torch.mps.synchronize()` around the timed region, same as
    `run_heads_decision`, so MPS's asynchronous queue submission isn't
    mistaken for compute time. `run_heads_decision` is untouched — this is a
    separate entry point.
    """
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    start = time.perf_counter()

    features = encode(texts)
    with torch.no_grad():
        field_probabilities = heads(features)

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000 / len(texts)

    results = []
    for row in range(len(texts)):
        decisions = {}
        for field, probabilities in field_probabilities.items():
            choices = heads.schema[field]
            probs_by_choice = {
                choice: float(prob)
                for choice, prob in zip(choices, probabilities[row])
            }
            best_choice = max(probs_by_choice, key=probs_by_choice.get)
            ranked = sorted(
                probs_by_choice.items(), key=lambda kv: kv[1], reverse=True
            )
            top_k_list = [
                {"choice": choice, "probability": prob}
                for choice, prob in ranked[:top_k]
            ]
            decisions[field] = {
                "decision": best_choice,
                "probabilities": probs_by_choice,
                "top_k": top_k_list,
            }
        results.append({"decisions": decisions, "latency_ms": latency_ms})

    return results


def route_tool(text, tool_heads, gate_head, threshold=GATE_THRESHOLD):
    """Route `text` to a tool, consulting `gate_head` before `tool_heads` so
    the router can abstain instead of always naming a catalogue tool.

    Encodes `text` exactly once and reuses those features for both heads —
    a gated routing decision costs the same single encoder forward pass any
    decision does (`tool-request-gate`); it does not call
    `run_heads_decision`, which would encode a second time.

    Compares the gate's "yes" probability against `threshold`. Below
    threshold, the query is reported as not a tool request: no tool name,
    an empty shortlist. At or above threshold, the query is routed by
    `tool_heads` exactly as `run_heads_decision(text, tool_heads)` alone
    would route it — same `decision` and `top_k` (defaulting to five, as
    `run_heads_decision` does), since both read the same features through
    the same trained head.

    Returns `{"is_tool_request": bool, "gate_probability": float,
    "tool": str | None, "top_k": list}`. `gate_probability` is always
    present, whichever way the gate decided (`gated-routing`).

    `run_heads_decision` and its return shape are untouched — this is a
    separate entry point, per `comparable-return-shape`.
    """
    features = encode([text])

    with torch.no_grad():
        gate_probabilities = gate_head(features)
    gate_choices = gate_head.schema["is_tool_request"]
    gate_probs_by_choice = {
        choice: float(prob)
        for choice, prob in zip(
            gate_choices, gate_probabilities["is_tool_request"][0]
        )
    }
    gate_probability = gate_probs_by_choice["yes"]
    is_tool_request = gate_probability >= threshold

    if not is_tool_request:
        return {
            "is_tool_request": False,
            "gate_probability": gate_probability,
            "tool": None,
            "top_k": [],
        }

    with torch.no_grad():
        tool_probabilities = tool_heads(features)
    tool_choices = tool_heads.schema["tool"]
    tool_probs_by_choice = {
        choice: float(prob)
        for choice, prob in zip(tool_choices, tool_probabilities["tool"][0])
    }
    best_tool = max(tool_probs_by_choice, key=tool_probs_by_choice.get)
    ranked = sorted(tool_probs_by_choice.items(), key=lambda kv: kv[1], reverse=True)
    top_k_list = [
        {"choice": choice, "probability": prob} for choice, prob in ranked[:5]
    ]

    return {
        "is_tool_request": True,
        "gate_probability": gate_probability,
        "tool": best_tool,
        "top_k": top_k_list,
    }


def route_tool_batch(texts, tool_heads, gate_head, threshold=GATE_THRESHOLD):
    """Route every text in `texts` to a tool, encoding all of them in one
    shared encoder forward pass and reusing those features for both the
    gate head and the tool head.

    Returns one result per text, in input order, each carrying the same
    keys `route_tool` returns for a single text: `{"is_tool_request": bool,
    "gate_probability": float, "tool": str | None, "top_k": list}`. A text
    the gate head rejects (its "yes" probability below `threshold`) carries
    no tool name and an empty shortlist, exactly as `route_tool` reports it.

    `route_tool` is untouched — this is a separate entry point.
    """
    features = encode(texts)

    with torch.no_grad():
        gate_probabilities = gate_head(features)
    gate_choices = gate_head.schema["is_tool_request"]

    with torch.no_grad():
        tool_probabilities = tool_heads(features)
    tool_choices = tool_heads.schema["tool"]

    results = []
    for row in range(len(texts)):
        gate_probs_by_choice = {
            choice: float(prob)
            for choice, prob in zip(
                gate_choices, gate_probabilities["is_tool_request"][row]
            )
        }
        gate_probability = gate_probs_by_choice["yes"]
        is_tool_request = gate_probability >= threshold

        if not is_tool_request:
            results.append(
                {
                    "is_tool_request": False,
                    "gate_probability": gate_probability,
                    "tool": None,
                    "top_k": [],
                }
            )
            continue

        tool_probs_by_choice = {
            choice: float(prob)
            for choice, prob in zip(tool_choices, tool_probabilities["tool"][row])
        }
        best_tool = max(tool_probs_by_choice, key=tool_probs_by_choice.get)
        ranked = sorted(
            tool_probs_by_choice.items(), key=lambda kv: kv[1], reverse=True
        )
        top_k_list = [
            {"choice": choice, "probability": prob} for choice, prob in ranked[:5]
        ]

        results.append(
            {
                "is_tool_request": True,
                "gate_probability": gate_probability,
                "tool": best_tool,
                "top_k": top_k_list,
            }
        )

    return results


def save_heads(heads, path):
    """Persist trained head weights together with the schema they were
    trained for, so `load_heads` can reload them without retraining.
    """
    torch.save({"schema": heads.schema, "state_dict": heads.state_dict()}, path)


def load_heads(path, schema):
    """Reload head weights from `path` into a fresh `DecisionHeads` instance
    built for `schema`.

    Raises `ValueError` naming the mismatch if the saved schema's fields or
    any field's declared choices differ from `schema`, rather than loading
    mismatched weights.
    """
    checkpoint = torch.load(path, weights_only=False)
    saved_schema = checkpoint["schema"]

    if set(saved_schema.keys()) != set(schema.keys()):
        raise ValueError(
            f"Schema mismatch: saved fields {sorted(saved_schema.keys())} do "
            f"not match requested fields {sorted(schema.keys())}"
        )
    for field, choices in schema.items():
        if list(saved_schema[field]) != list(choices):
            raise ValueError(
                f"Schema mismatch on field {field!r}: saved choices "
                f"{saved_schema[field]!r} do not match requested choices "
                f"{choices!r}"
            )

    heads = DecisionHeads(schema)
    heads.load_state_dict(checkpoint["state_dict"])
    heads.eval()
    return heads
