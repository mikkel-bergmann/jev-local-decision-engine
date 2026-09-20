"""Local Jev-style decision engine: batched schema-field classification over
Qwen/Qwen2.5-1.5B-Instruct running on Apple Silicon (MPS).
"""

import sys
import time
from typing import Literal

import torch
from pydantic import BaseModel, create_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SCHEMA = {
    "category": ["billing", "technical_support", "sales", "general_inquiry"],
    "urgency": ["low", "medium", "high", "critical"],
    "sentiment": ["negative", "neutral", "positive"],
}


class IntentDecision(BaseModel):
    category: Literal["billing", "technical_support", "sales", "general_inquiry"]
    urgency: Literal["low", "medium", "high", "critical"]
    sentiment: Literal["negative", "neutral", "positive"]


_model_cache = {}
_decision_model_cache = {}


def build_decision_model(schema):
    """Build (or reuse) a Pydantic model validating decisions for `schema`.

    The model's fields are the schema's keys, each typed as a `Literal` over
    that field's own choices, built with `pydantic.create_model`. Cached on a
    key derived from the schema's items so a repeated schema reuses one
    class rather than rebuilding it on every call.
    """
    cache_key = tuple((field, tuple(choices)) for field, choices in schema.items())
    if cache_key not in _decision_model_cache:
        field_definitions = {
            field: (Literal.__getitem__(tuple(choices)), ...)
            for field, choices in schema.items()
        }
        _decision_model_cache[cache_key] = create_model(
            "DynamicDecision", **field_definitions
        )
    return _decision_model_cache[cache_key]


def _get_model_and_tokenizer():
    """Load the model and tokenizer once and cache them for reuse."""
    if "model" not in _model_cache:
        model, tokenizer = load_model()
        _model_cache["model"] = model
        _model_cache["tokenizer"] = tokenizer
    return _model_cache["model"], _model_cache["tokenizer"]


def run_jev_decision(text, schema):
    """Score `text` against `schema` and return schema-validated decisions.

    `latency_ms` times scoring only (the batched forward pass and the
    probability gather), excluding model load.
    """
    model, tokenizer = _get_model_and_tokenizer()

    start = time.perf_counter()
    field_probabilities = score_fields(model, tokenizer, text, schema)
    latency_ms = (time.perf_counter() - start) * 1000

    decisions = {}
    for field, probabilities in field_probabilities.items():
        best_choice = max(probabilities, key=probabilities.get)
        decisions[field] = {"decision": best_choice, "probabilities": probabilities}

    # Validate the selected decisions through a Pydantic model built for this
    # schema's own fields and choices; raises on an off-schema value rather
    # than accepting it.
    decision_model = build_decision_model(schema)
    decision_model(**{field: value["decision"] for field, value in decisions.items()})

    return {"decisions": decisions, "latency_ms": latency_ms}


def run_jev_decision_batch(texts, schema):
    """Score every text in `texts` against `schema` through the first-token
    path, in one shared batched forward pass, and return one
    schema-validated result per text, in input order.

    Builds every text's field prompts (via `build_field_prompts`) and pads
    all of them — every field, every text — into a single batch, so the
    number of forward passes stays at one regardless of how many texts or
    fields are involved. Each row's final position (position -1, since
    padding is on the left) is read through `candidate_probabilities`,
    exactly as `score_fields` reads a single text's rows.

    Never calls `score_fields_full_sequence`: that path's log-probabilities
    over the whole vocabulary need 1.2 GB at ten texts and grow with the
    batch, so it stays unbatched (`batch-constrained-scoring`).

    Each result has the same shape `run_jev_decision` returns for a single
    text — `{"decisions": {...}, "latency_ms": ...}`. `latency_ms` is the
    batch's total scoring time divided by `len(texts)`, so it reads as a
    comparable per-prompt figure rather than the batch total.

    `run_jev_decision` is untouched — this is a separate entry point.
    """
    model, tokenizer = _get_model_and_tokenizer()

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    start = time.perf_counter()

    fields = list(schema.keys())
    all_prompts = []
    row_text_index = []
    for text_index, text in enumerate(texts):
        _, prompts = build_field_prompts(tokenizer, text, schema)
        all_prompts.extend(prompts)
        row_text_index.extend([text_index] * len(prompts))

    batch = tokenizer(
        all_prompts, padding=True, return_tensors="pt", add_special_tokens=False
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**batch)
    last_token_logits = outputs.logits[:, -1, :]

    candidates_by_field = {
        field: resolve_candidates(tokenizer, schema[field]) for field in fields
    }

    per_text_probabilities = [dict() for _ in texts]
    for row, text_index in enumerate(row_text_index):
        field = fields[row % len(fields)]
        candidates = candidates_by_field[field]
        candidate_ids = torch.tensor(
            list(candidates.values()), dtype=torch.long, device=last_token_logits.device
        )
        probs = candidate_probabilities(last_token_logits[row], candidate_ids)
        per_text_probabilities[text_index][field] = {
            choice: float(prob) for choice, prob in zip(candidates.keys(), probs)
        }

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000 / len(texts)

    decision_model = build_decision_model(schema)
    results = []
    for field_probabilities in per_text_probabilities:
        decisions = {}
        for field, probabilities in field_probabilities.items():
            best_choice = max(probabilities, key=probabilities.get)
            decisions[field] = {"decision": best_choice, "probabilities": probabilities}
        decision_model(**{field: value["decision"] for field, value in decisions.items()})
        results.append({"decisions": decisions, "latency_ms": latency_ms})

    return results


def load_model():
    """Load the Qwen model onto MPS in float16, plus its tokenizer.

    Exits non-zero naming MPS as unavailable rather than silently falling
    back to CPU. Loads with no `device_map` argument (which hangs on this
    toolchain), then moves the model with `.to("mps")`.
    """
    if not torch.backends.mps.is_available():
        sys.exit("MPS is not available on this machine; refusing to fall back to CPU.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    model = model.to("mps")
    model.eval()

    return model, tokenizer


def build_field_prompts(tokenizer, text, schema):
    """Render one chat-templated prompt per schema field.

    Keeps the generation prompt intact, terminating newline included. The
    chat template's generation prompt ends in a newline, so the model's next
    token is the bare form of the candidate, not the space-prefixed one.
    Never rstrip this output — doing so leaves the prompt ending on the
    literal token `assistant`, where the model's next-token mass goes to
    `\n` rather than any class word.

    Returns (fields, prompts) as parallel lists, one entry per schema field.
    """
    fields = list(schema.keys())
    prompts = []
    for field, choices in schema.items():
        choices_str = ", ".join(choices)
        user_content = (
            f'Classify the intent of this message.\n\nMessage: "{text}"\n\n'
            f"What is the message's {field}? Respond with exactly one word "
            f"from: {choices_str}."
        )
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return fields, prompts


def candidate_probabilities(row_logits, candidate_ids):
    """Gather a field's candidate logits into one vector and softmax across it.

    Casts to float32 before the softmax so float16 logits do not distort
    small probabilities. Never softmaxes an individual scalar logit.
    """
    if not torch.is_tensor(candidate_ids):
        candidate_ids = torch.tensor(candidate_ids, dtype=torch.long)
    gathered = row_logits.float().index_select(0, candidate_ids)
    return torch.softmax(gathered, dim=0)


def score_fields(model, tokenizer, text, schema):
    """Score every schema field for `text` in a single batched forward pass.

    Builds one prompt per field, tokenizes them together with left padding,
    and reads each field's distribution from the final position of that
    field's own batch row (position -1, since padding is on the left).

    Returns {field: {choice: probability}}.
    """
    fields, prompts = build_field_prompts(tokenizer, text, schema)
    batch = tokenizer(
        prompts, padding=True, return_tensors="pt", add_special_tokens=False
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**batch)
    last_token_logits = outputs.logits[:, -1, :]

    results = {}
    for row, field in enumerate(fields):
        choices = schema[field]
        candidates = resolve_candidates(tokenizer, choices)
        candidate_ids = torch.tensor(
            list(candidates.values()), dtype=torch.long, device=last_token_logits.device
        )
        probs = candidate_probabilities(last_token_logits[row], candidate_ids)
        results[field] = {
            choice: float(prob) for choice, prob in zip(candidates.keys(), probs)
        }
    return results


def resolve_candidates(tokenizer, choices, bare=True):
    """Map each schema choice to the first token id of the form matching the
    prompt's final token boundary.

    `bare=True` (the default) scores the bare form (`tokenizer.encode(choice,
    ...)`), which is correct when the prompt ends in a newline — the case for
    every prompt this engine builds, since the chat template's generation
    prompt is never stripped. `bare=False` scores the space-prefixed form
    (`" " + choice`), for a prompt that ends mid-line instead.

    Raises ValueError naming both choices and the shared id if two choices
    within one field resolve to the same first token id.
    """
    candidates = {}
    for choice in choices:
        surface = choice if bare else " " + choice
        token_id = tokenizer.encode(surface, add_special_tokens=False)[0]
        for other_choice, other_id in candidates.items():
            if other_id == token_id:
                raise ValueError(
                    f"Candidate collision: {other_choice!r} and {choice!r} "
                    f"both resolve to first token id {token_id}"
                )
        candidates[choice] = token_id
    return candidates


def score_fields_full_sequence(model, tokenizer, text, schema):
    """Score every schema field for `text` by full-sequence, length-normalized
    candidate scoring, in one additional batched forward pass.

    For every (field, candidate) pair, builds one sequence: the field's
    chat-templated prompt (token ids, generation-prompt newline intact)
    followed by that candidate's own bare-form token ids. Runs a single
    `model(**batch)` over all (field, candidate) sequences together. For each
    sequence, takes the log-softmax over the vocabulary at every position
    that predicts one of the candidate's own tokens (never a prompt token),
    reads off the log-probability of the token actually there, and averages
    those log-probabilities over the candidate's own tokens. That mean is the
    candidate's score. Each field's candidate scores are then softmaxed
    across that field's candidates to get probabilities.

    Returns {field: {choice: probability}}.
    """
    fields, prompts = build_field_prompts(tokenizer, text, schema)

    sequences = []
    row_meta = []  # (field, choice, candidate_ids) per row, same order as sequences
    for field, prompt in zip(fields, prompts):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for choice in schema[field]:
            candidate_ids = tokenizer.encode(choice, add_special_tokens=False)
            sequences.append(prompt_ids + candidate_ids)
            row_meta.append((field, choice, candidate_ids))

    batch = tokenizer.pad(
        {"input_ids": sequences}, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**batch)
    # Cast to float32 before log_softmax, same as candidate_probabilities does,
    # so float16 logits do not distort small probabilities.
    log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)

    seq_len = batch["input_ids"].shape[1]
    field_scores = {field: {} for field in fields}
    for row, (field, choice, candidate_ids) in enumerate(row_meta):
        candidate_len = len(candidate_ids)
        # Left padding puts every sequence's real tokens flush against the
        # right-hand end, so the candidate's own tokens are always the last
        # `candidate_len` positions — computed from the right, never from an
        # absolute left-based offset, regardless of this row's prompt length
        # or how much left padding it received.
        candidate_start = seq_len - candidate_len
        total_log_prob = 0.0
        for offset, token_id in enumerate(candidate_ids):
            token_position = candidate_start + offset
            predicting_position = token_position - 1
            total_log_prob += log_probs[row, predicting_position, token_id].item()
        field_scores[field][choice] = total_log_prob / candidate_len

    results = {}
    for field in fields:
        choices = schema[field]
        scores = torch.tensor(
            [field_scores[field][choice] for choice in choices], dtype=torch.float32
        )
        probs = torch.softmax(scores, dim=0)
        results[field] = {
            choice: float(prob) for choice, prob in zip(choices, probs)
        }
    return results


def _current_rss_gb():
    """Return this process's CURRENT resident set size in GB (not peak).

    Shells out to `ps` rather than adding a dependency (e.g. psutil). Peak
    RSS (`resource.getrusage(...).ru_maxrss`) is not used here: it includes
    the transient safetensors load buffer and overstates the steady-state
    footprint the benchmark target cares about.
    """
    import os
    import subprocess

    output = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rss_kb = int(output)
    return rss_kb / (1024**2)


def _peak_rss_gb():
    """Return this process's PEAK resident set size in GB, for reference only."""
    import resource

    # On macOS, ru_maxrss is reported in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


if __name__ == "__main__":
    import json

    BENCHMARK_TEXT = (
        "My account was double charged for last month's subscription, "
        "fix this immediately!"
    )

    result = run_jev_decision(BENCHMARK_TEXT, SCHEMA)
    print(json.dumps(result, indent=2))
    print(f"Scoring latency: {result['latency_ms']:.1f} ms")
    print(f"Resident memory (current): {_current_rss_gb():.2f} GB")
    print(f"Resident memory (peak): {_peak_rss_gb():.2f} GB")
