# distilled-decision-heads
Status: verified

## Idea

Train one linear classification head per schema field on the frozen encoder's pooled hidden
state, so every field resolves from a single shared forward pass instead of one prompt each.

### Motivation

Constrained decoding builds a separate prompt per field and a separate row per candidate, so
cost grows with both: 202 ms for the three-field schema, and 33.7 s for a 102-way choice because
the prompt itself enumerates every option. Classification heads read all fields off one encode,
making latency independent of field count and class count.

### Details

- Add `decision_heads.py`: frozen-encoder feature extraction and a `DecisionHeads` module holding
  one `Linear(1536, n_classes)` per field.
- Add `train_heads.py`: builds a synthetic corpus, labels it with the existing constrained engine
  as teacher, trains the heads, and writes the weights plus a metrics report.
- Add a hand-authored labelled holdout that never enters training, and report teacher accuracy on
  it alongside student accuracy.
- Add `tests/test_decision_heads.py` covering the shared-pass contract, latency, and agreement.

Affected capabilities: `decision-heads` (added). `local-decision-engine` is untouched — this is a
parallel path, and the engine remains both the zero-shot fallback and the distillation teacher.
Impact: `decision_heads.py`, `train_heads.py`, `tests/test_decision_heads.py`, `data/`. No new
dependencies.

### Non-goals

- No tool-routing head and no 102-way classification in this change.
- No general "train a head for any schema" pipeline; the three existing fields only.
- No replacement or removal of the constrained-decoding engine.
- No fix to `score_fields_full_sequence`'s memory blowup; that path is not touched.
- No fine-tuning of the encoder — it stays frozen, and only the heads are trained.

## Implementation

**Files.** `decision_heads.py`, `train_heads.py`, `tests/test_decision_heads.py`, `data/`.

**Interfaces.**

- `encode(texts) -> Tensor` — mean-pooled last hidden state over the attention mask, L2-normalized,
  shape `(n, 1536)`. Uses `model.model(...)`, not the LM head.
- `DecisionHeads(schema)` — `torch.nn.Module` holding `Linear(1536, len(choices))` per field;
  `forward(features)` returns `{field: probabilities}` for every field from one feature tensor.
- `run_heads_decision(text) -> dict` — same return shape as `run_jev_decision`, so the two paths
  are directly comparable: `{"decisions": {field: {decision, probabilities}}, "latency_ms": ...}`.
- `save_heads(path)` / `load_heads(path)` — persist and restore trained weights with their schema.

**Decisions.**

- **Freeze the encoder; train only the heads.** Verified: a `Linear(1536, 3)` head over mean-pooled
  frozen features, trained on 18 hand-written sentences, classified 6 of 6 held-out sentences
  correctly, including the `neutral` cases the constrained engine handles worst. Fine-tuning 1.5B
  parameters is unnecessary for this and would need far more data.
- **Pool the hidden state; do not use cosine similarity over it.** Measured and rejected: ranking
  102 tools by cosine similarity over these same pooled features returned an identical top-3 for
  four unrelated queries. A trained head learns which dimensions carry the signal; similarity
  weights all 1536 equally and drowns it. Same features, opposite outcome.
- **Distil from the constrained engine, and measure the teacher too.** The engine labels the
  synthetic corpus, so no manual labelling gates the change. The student cannot exceed its teacher,
  and the teacher has two measured defects — injected instructions capture its decision
  (`sales` 0.0025 to 0.9993) and it is more certain on ambiguous input than specific input
  (entropy 0.0012 against 0.2847). The holdout therefore scores teacher AND student, so the report
  shows whether a gap is the student's error or the teacher's ceiling.
- **Hand-author the holdout with known labels.** Authoring a sentence for a target class yields a
  label without a separate annotation pass, and holding it out entirely keeps it honest.
- **Build the corpus combinatorially, not by generation.** Templates crossed with slot fillers give
  a deterministic, reproducible corpus with no sampling temperature and no generation cost.
- **Keep the return shape identical to `run_jev_decision`.** Identical shapes let the same
  assertions and the same comparison harness run against both paths.

**Risks and trade-offs.**

- The student inherits the teacher's defects. Guarded by scoring both on the holdout and reporting
  disagreements rather than reporting student accuracy alone.
- A combinatorial corpus may be narrower than real traffic, so heads could overfit template
  phrasing. Guarded by a hand-authored holdout written in free prose, sharing no template with the
  corpus.
- Pooled features may separate `sentiment` far better than `category`, whose classes are
  semantically closer. Guarded by reporting per-field accuracy rather than one aggregate number.

## Questions and answers

### Q1: Where does the training data come from?
- **Question:** Should heads be trained by self-distillation from the existing engine, from a
  hand-labelled seed set, from public intent datasets, or from synthesised-then-distilled inputs?
  Options: (1) self-distillation; (2) hand-labelled seed; (3) public datasets; (4) synthesise then
  distil. Recommendation: (1).
- **Verdict:** INSUFFICIENT
- **Answered by:** USER
- **Answer:** Self-distillation — option 1. The existing engine already produces correct labels for
  this small schema, so it becomes its own teacher and no manual labelling gates the change. A
  hand-authored holdout that never enters training scores both teacher and student, so an inherited
  defect shows up as a teacher ceiling rather than hiding inside the student's numbers.
- **Queued:** none

### Q2: How wide should the first change be?
- **Question:** Should the change cover heads for the existing three fields, those plus a 102-way
  tool head, or a general train-a-head-for-any-schema pipeline? Options: (1) three fields;
  (2) three fields plus the tool head; (3) general pipeline. Recommendation: (1).
- **Verdict:** INSUFFICIENT
- **Answered by:** USER
- **Answer:** The three existing fields only — option 1. Proving the architecture end to end against
  an already-measured baseline is what settles whether heads are worth building on; the tool head
  and a general pipeline both rest on that answer.
- **Queued:** none

### Q3: What happens to the constrained-decoding engine?
- **Question:** Should the engine be kept as the zero-shot path and distillation teacher, replaced
  by heads, or have its fate decided after measuring head accuracy? Options: (1) keep it;
  (2) replace it; (3) decide later. Recommendation: (1).
- **Verdict:** INSUFFICIENT
- **Answered by:** USER
- **Answer:** Keep it — option 1. It is the teacher this change depends on, and it needs no training
  data, so it stays the path that works on a schema no head has been trained for.
- **Queued:** none
