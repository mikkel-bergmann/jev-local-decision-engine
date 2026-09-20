# tool-routing-head
Status: verified

## Idea

Add a tool-routing head over a ~100-tool catalogue and a top-k return, so an assistant can narrow
its tools to a handful before its main model sees any tool description.

### Motivation

An assistant carrying ~100 tools spends context on every tool description and a round-trip on
dynamic selection. The constrained engine cannot replace that: 102 namespaced names collapse to 64
distinct first tokens, leaving 37% unreachable, and full-sequence scoring takes 33.7 s and fails
with a 24.94 GiB allocation. A classification head costs one extra output column at no extra
forward pass.

### Details

- Add `data/tools.json`: a catalogue of roughly 105 tools mixing general-purpose developer and
  SaaS tools with PCI-compliance and food-safety tools for a quick-service-restaurant operator.
- Add ~3 hand-authored natural-language queries per tool as training data, and a held-out
  evaluation set authored last.
- Extend `run_heads_decision` with an optional top-k return per field, defaulting to 5, leaving
  the existing `decision` and `probabilities` keys untouched.
- Evaluate on recall@5 as the headline, with top-1 secondary and a count of tools the head never
  predicts.

Affected capabilities: `decision-heads` (requirements added, `comparable-return-shape` modified).
Impact: `decision_heads.py`, `train_heads.py`, `data/tools.json`, tool training and holdout files,
`tests/test_tool_routing.py`. The constrained engine and its `tool-routing-schema` requirement are
untouched. No new dependencies.

### Non-goals

- No argument or parameter extraction; the head returns a tool name and probabilities only.
- No top-k for `category`, `urgency` or `sentiment`; those keep returning an argmax decision.
- No change to `jev_local_engine.py` or to the engine's own tool-routing path.
- No integration with a real assistant runtime, and no use of a real tool catalogue.
- No retrieval, embedding index, or nearest-centroid alternative to the head.

## Implementation

**Files.** `decision_heads.py`, `train_heads.py`, `data/tools.json`, `data/tool_train_*.json`,
`data/tool_holdout.json`, `tests/test_tool_routing.py`.

**Interfaces.**

- `load_tool_catalogue() -> list[str]` — the ordered tool names from `data/tools.json`.
- `TOOL_SCHEMA = {"tool": load_tool_catalogue()}` — a schema the existing `DecisionHeads` accepts
  unchanged, since it already builds one `Linear(1536, n)` per field.
- `run_heads_decision(text, heads, top_k=5)` — each field's record gains a `top_k` list of
  `{"choice": str, "probability": float}` ordered by descending probability, alongside the existing
  `decision` and `probabilities` keys.
- `evaluate_tool_head(heads, holdout_path) -> dict` — returns `recall_at_5`, `recall_at_1`,
  `unpredicted_tool_count`, and the list of tools never predicted.

**Decisions.**

- **Recall@5 is the headline, not top-1.** The product narrows ~100 tools to a handful, so the head
  only has to put the right tool in a short list. Measured on the existing four-class data, top-2
  reaches 0.839 where top-1 reaches 0.628 at three examples per class — the looser bar is what makes
  three queries per tool viable. Rejected: top-1 accuracy as headline, which would understate
  fitness for the actual use.
- **Three hand-authored queries per tool.** Measured data efficiency: one example per class gives
  top-1 0.522 and top-2 0.750; three gives 0.628 and 0.839; five gives 0.717 and 0.872. Three sits
  where the curve turns, for roughly 315 authored items. Rejected: one per tool, too thin; five per
  tool, about 70% more authoring for a smaller increment.
- **Hand labels, never distillation.** The preceding change measured 48 hand-labelled examples
  beating 432 engine-labelled ones (0.867 against 0.667 on category) because the engine is the
  ceiling. The engine cannot label tool routing at all, so distillation is doubly unavailable.
- **Every cross-validation or fold split shuffles first.** A stride-based fold in the preceding
  change aliased with a repeating class cycle in the data, making each fold omit an entire class and
  reporting four-way urgency at 0.200 when the true figure is near 0.40. Any fold construction here
  shuffles with a seeded generator.
- **The catalogue carries deliberate near-neighbour families.** PCI scanning, HACCP logging, audit
  and reporting tools each contribute several similarly-named members, because a catalogue of
  unrelated tools would make routing look easier than it is.
- **Top-k is additive.** The existing `decision` and `probabilities` keys stay, so
  `comparable-return-shape` still holds for the fields it covers; that requirement is modified only
  to permit the extra key.

**Risks and trade-offs.**

- Three queries per tool over ~105 classes is sparse, and some tools may never be predicted. Guarded
  by reporting the unpredicted-tool count as a first-class metric rather than hiding it inside an
  average.
- Near-neighbour families may prove unroutable at this data density. Guarded by reporting per-family
  behaviour so a failure is attributable to specific tools rather than to the approach.
- Authored queries may echo tool names literally, teaching string matching instead of intent.
  Guarded by requiring a share of queries that avoid the tool's own words.

## Questions and answers

### Q1: Whose tool catalogue?
- **Question:** Should the head train on the user's real tool list, an invented catalogue, or an
  invented one documented for later retraining? Options: (1) real list; (2) invented;
  (3) invented plus retraining notes. Recommendation: (1).
- **Verdict:** INSUFFICIENT
- **Answered by:** USER
- **Answer:** An invented catalogue — option 2 — extended with API calls for a SaaS that manages
  business PCI and safety compliance from a quick-service-restaurant perspective. That domain
  supplies dense near-neighbour families, which is where routing is actually hard.
- **Queued:** none

### Q2: How many queries per tool?
- **Question:** Should training carry three, five, or one hand-authored query per tool? Options:
  (1) three, about 315 items; (2) five, about 525; (3) one, about 105. Recommendation: (1).
- **Verdict:** INSUFFICIENT
- **Answered by:** USER
- **Answer:** Three per tool — option 1. The measured curve turns there: top-2 rises from 0.750 at
  one example per class to 0.839 at three, then only to 0.872 at five.
- **Queued:** none

### Q3: What does the API return?
- **Question:** Should top-k be added for the tool field with k configurable, applied to every
  field, or replaced by a full sorted distribution the caller truncates? Options: (1) configurable
  top-k, default 5, existing fields unchanged; (2) top-k everywhere; (3) full distribution.
  Recommendation: (1).
- **Verdict:** INSUFFICIENT
- **Answered by:** USER
- **Answer:** Configurable top-k defaulting to 5 — option 1 — with `category`, `urgency` and
  `sentiment` still returning an argmax decision, so existing callers are unaffected.
- **Queued:** none

### Q4: What is the headline metric?
- **Question:** Should evaluation lead with recall@5, with top-1 accuracy, or with a recall curve
  and no single headline? Options: (1) recall@5 primary plus top-1 and unpredicted-tool count;
  (2) top-1 primary; (3) curve only. Recommendation: (1).
- **Verdict:** INSUFFICIENT
- **Answered by:** USER
- **Answer:** Recall@5 primary, top-1 secondary, plus the count of tools never predicted — option 1.
  The product narrows a catalogue to a shortlist, so recall@5 is what fitness means here, and the
  unpredicted count exposes dead classes an average would hide.
- **Queued:** none
