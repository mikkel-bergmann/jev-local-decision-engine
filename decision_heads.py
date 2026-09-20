"""Distilled decision heads: one linear classification head per schema field,
trained on frozen-encoder features so every field resolves from a single
shared forward pass instead of one prompt per field.
"""

import time

import torch

import jev_local_engine

HIDDEN_SIZE = 1536

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


def run_heads_decision(text, heads):
    """Score `text` against `heads` and return the same shape `run_jev_decision`
    returns, so the two paths are directly comparable.

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
        decisions[field] = {"decision": best_choice, "probabilities": probs_by_choice}

    return {"decisions": decisions, "latency_ms": latency_ms}


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
