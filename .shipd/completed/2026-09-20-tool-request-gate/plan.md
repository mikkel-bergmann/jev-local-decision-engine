# tool-request-gate
Status: verified

## Idea

Add a binary head that decides whether a query is a tool request at all, so the router can abstain
instead of always naming a tool.

### Motivation

A softmax over the 110-tool catalogue must sum to one, so it cannot express "none of these": asked
`2+2` it reports a tool at probability 0.777, and `hello` at 0.607, overlapping the 0.837 median of
correct in-scope answers. A confidence threshold cannot separate them — at 0.25 it catches 44% of
out-of-scope queries while already discarding 3% of correct ones.

### Details

- Add a binary gate head over the same frozen shared features, trained on the existing tool queries
  as positives and hand-authored non-requests as negatives.
- Choose its decision threshold by measurement on a holdout, and record the number chosen.
- Add `route_tool`, which consults the gate first and returns no tool when the gate says the query
  is not a request.
- Fold in `data/tool_train_d.json`, twenty inspection queries already authored, and its path
  registration in `train_heads.py`.

Affected capabilities: `decision-heads` (requirements added). Impact: `decision_heads.py`,
`train_heads.py`, new gate data files, `data/gate_head.pt`, `tests/test_tool_gate.py`. No new
dependencies. `run_heads_decision` keeps its present contract untouched.

### Non-goals

- No `none` entry in the tool catalogue; abstention is the gate's job, not a 111th class.
- No gating of `category`, `urgency` or `sentiment` — those apply to any text, requested or not.
- No inspection-readiness tool, though its absence is the known cause of three misroutes.
- No change to `run_heads_decision`'s return shape, and no change to the constrained engine.
- No calibration of the tool head's own probabilities; the gate replaces that approach.

## Implementation

**Files.** `decision_heads.py`, `train_heads.py`, `data/gate_negatives_a.json`,
`data/gate_negatives_b.json`, `data/gate_holdout.json`, `data/gate_head.pt`,
`tests/test_tool_gate.py`.

**Interfaces.**

- `GATE_SCHEMA = {"is_tool_request": ["no", "yes"]}` — a schema the existing `DecisionHeads` accepts
  unchanged.
- `train_gate_head()` — trains on the tool queries as positives and the authored negatives, using a
  class weight to offset the imbalance, and saves to `data/gate_head.pt`.
- `choose_gate_threshold(gate, holdout) -> dict` — sweeps candidate thresholds and returns the
  selected value with its measured precision, recall and negative-rejection rate.
- `GATE_THRESHOLD` — the selected constant, carrying a comment naming the measurement that chose it.
- `route_tool(text, tool_heads, gate_head, threshold=GATE_THRESHOLD) -> dict` — returns
  `{"is_tool_request": bool, "gate_probability": float, "tool": str | None, "top_k": list}`, with
  `tool` `None` and `top_k` empty when the gate rejects.

**Decisions.**

- **A separate gate head, not a `none` class.** Routing quality and in-scope detection are different
  questions; coupling them means every catalogue change disturbs abstention. A separate head costs
  0.041 ms on features already computed, so the split is nearly free.
- **The design is measured feasible before being specified.** A binary `Linear(1536, 2)` over these
  frozen features, under 5-fold shuffled cross-validation, reached precision 0.977, recall 0.983 and
  F1 0.980, rejecting 33 of 41 negatives — and that was at a 350-to-41 imbalance.
- **Author at least 150 negatives.** The probe's 0.805 rejection rate came from 41 negatives against
  350 positives. Raising the negative count is the cheapest available improvement, and 150 brings
  the ratio under 3:1.
- **The threshold is measured, never chosen by hand.** A hand-picked constant is how the previous
  advice failed: thresholding the tool head's own confidence looked reasonable and was refuted by
  measurement. The spec therefore requires the number to come from a recorded sweep.
- **Abstention is additive, not a `None` decision.** `run_heads_decision` declares `decision` as a
  string; returning `None` there would break `comparable-return-shape`. `route_tool` is a separate
  entry point, so no merged requirement changes.
- **Every fold split shuffles with a seeded generator.** An unshuffled stride fold aliased with a
  repeating class cycle earlier in this project and reported 0.200 where the truth was near 0.40.
- **Tests exercise the shipped checkpoints.** Two tests in this repository have passed while
  measuring something other than what ships — randomly initialized weights, and a head whose schema
  did not match its holdout. Gate tests load `data/gate_head.pt` and `data/tool_head.pt`.

**Risks and trade-offs.**

- Negatives authored by one hand may cover a narrower space than real traffic, making the gate look
  better than it is. Guarded by a holdout authored last and checked for paraphrase leakage rather
  than only exact-text overlap.
- A gate tuned for high rejection will reject borderline real requests. Guarded by reporting both
  directions — negative rejection rate and in-scope retention — at the chosen threshold, so the
  trade is visible rather than implied.
- The twenty inspection queries slightly moved tool-head numbers: holdout recall@5 held at 0.9545
  while recall@1 fell from 0.8455 to 0.8364 and never-predicted rose from 14 to 15. Recorded here so
  the regression is not rediscovered as a mystery.
