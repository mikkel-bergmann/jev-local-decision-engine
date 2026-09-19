# jev-local-decision-engine
Status: ready

## Idea

Build a local Jev-style classifier that scores every schema field in one batched
forward pass and reads calibrated class probabilities straight from the logits.

### Motivation

The supplied draft cannot classify anything: `next_token_logits[id].softmax(dim=0)`
softmaxes a 0-dim scalar, which returns `1.0` for every choice, so the result is
always the first option regardless of input. It also runs one forward pass per
field and calls `device_map="mps"`, which hangs indefinitely on this toolchain.

### Details

- Add `requirements.txt` pinning `torch`, `transformers`, `accelerate`, `pydantic`.
- Add `jev_local_engine.py`: loads `Qwen/Qwen2.5-1.5B-Instruct` in `float16` on MPS,
  resolves candidate token ids, scores all fields in one batched forward pass, and
  emits Pydantic-validated JSON with per-choice probabilities and a latency figure.
- Add `tests/test_jev_engine.py` covering candidate resolution, the gather-then-softmax
  fix, batch invariance, and schema validation.

Affected capabilities: `local-decision-engine` (added). Impact: three new files at the
repo root; no existing code to modify. One ~3.1 GB model download into
`~/.cache/huggingface` on first run.

### Non-goals

- No HTTP server, no daemon, no persistent process.
- No fine-tuning, LoRA, quantization, or training of any kind.
- No full-sequence rescoring of multi-token choices — first-token scoring only.
- No `hf_transfer` — it is deprecated and no longer used by `huggingface_hub`.

## Implementation

**Files.** `requirements.txt`, `jev_local_engine.py`, `tests/test_jev_engine.py`.

**Interfaces.**

- `IntentDecision(BaseModel)` with `category`, `urgency`, `sentiment`, each a
  `Literal[...]` over its allowed choices so Pydantic rejects an off-schema value.
- `resolve_candidates(tokenizer, choices) -> dict[str, int]` — maps each choice to the
  first token id of its space-prefixed form.
- `score_fields(model, tokenizer, text, schema) -> dict[str, dict[str, float]]` — one
  batched forward pass, returns per-field choice probabilities.
- `run_jev_decision(text, schema) -> dict` — returns `{"decisions": ..., "latency_ms": ...}`.

**Decisions.**

- **Load with `.to("mps")`, never `device_map="mps"`.** Verified: `from_pretrained(
  "hf-internal-testing/tiny-random-gpt2", dtype=torch.float16, device_map="mps")` hung at
  `Loading weights: 0%|  | 0/64` for 60s across three runs and had to be killed, leaking a
  multiprocessing semaphore. The same call without `device_map`, followed by `.to("mps")`,
  printed `loaded cpu in 1.2s` / `param device: mps:0` and exited 0. Rejected: `device_map`,
  which the draft used — it does not complete on this stack.
- **Use `dtype=`, not `torch_dtype=`.** `transformers` 5.17 logs
  `torch_dtype is deprecated! Use dtype instead!`; the verified load above passed `dtype=`
  and emitted no deprecation warning.
- **Gather candidate logits into one vector, then softmax across it.** This is the fix for
  the draft's dead line. Verified: `torch.softmax(last[0].float().index_select(0, cand), dim=0)`
  returned `[0.3078, 0.3731, 0.3191]` summing to `1.0`, where the draft's per-scalar softmax
  returned `1.0` for every candidate. Cast to `float32` before the softmax so `float16`
  logits do not distort small probabilities.
- **Batch the fields along the batch dimension, left-padded.** Build one prompt per field,
  tokenize with `padding=True` and `padding_side="left"`, run a single `model(**batch)`, and
  read `logits[:, -1, :]`. Left padding keeps position `-1` the true last token of every row.
  Verified: a 3-prompt batch produced `input_ids (3, 14)` and `logits[:, -1, :] (3, 1000)`.
  Rejected: reading several slots from one shared sequence — causal attention makes each
  later field's readout depend on whatever filler occupies the earlier slots, which biases it.
- **Score the first token of the space-prefixed choice.** Verified against the real Qwen2.5
  vocabulary: first-token ids are distinct within every field, `COLLISION=False` for all six
  bare and space-prefixed variants. Space-prefixed ids differ from bare ones (` low`=3347 vs
  `low`=10303), so the prompt must end without trailing whitespace. `technical_support` and
  `general_inquiry` are multi-token, so this is an approximation — sound here only because
  their first tokens (`technical`, `general`) are unambiguous.
- **Apply the tokenizer chat template.** `tokenizer.chat_template is not None` returned
  `True`; the draft fed a raw string, prompting the instruct-tuned model off-distribution.

**Risks and trade-offs.**

- A future schema could introduce a first-token collision, silently conflating two choices.
  Guarded by a startup check that raises on a duplicate candidate id.
- Left padding could shift position ids on some architectures. Guarded by a batch-invariance
  test asserting a field scored inside the batch matches the same field scored alone.
- First-token scoring degrades if choices are renamed to share a prefix. The collision check
  turns that into a loud failure rather than a wrong answer.

## Questions and answers

### Q1: Reproduce the broken draft, or ship only the fix?
- **Question:** Should the change build only the corrected engine, also run the draft verbatim
  to demonstrate the failure, or keep both as selectable modes? Options: (1) fix only;
  (2) demonstrate then fix; (3) keep both permanently. Recommendation: (1).
- **Verdict:** INSUFFICIENT
- **Answered by:** USER
- **Answer:** Fix only — option 1. The defect is already verified by direct probe, so a
  reproduction task would add cost without adding evidence.
- **Queued:** none

### Q2: How should multi-token choices be scored?
- **Question:** Should `technical_support` and `general_inquiry` be scored by first token,
  by full-sequence length-normalized log-likelihood, or by first token plus a collision
  assertion? Options: (1) first token; (2) full sequence; (3) first token plus assertion.
  Recommendation: (1).
- **Verdict:** INSUFFICIENT
- **Answered by:** USER
- **Answer:** First token only — option 1. Probing the real vocabulary showed no first-token
  collisions in this schema, so a second scoring pass would buy no accuracy here. The
  collision guard is still implemented, since it costs one comparison at startup.
- **Queued:** none

### Q3: Which dtype on MPS?
- **Question:** Should the model load in `float16`, `bfloat16`, or `float32` on MPS?
  Options: (1) float16; (2) bfloat16; (3) float32. Recommendation: (1).
- **Verdict:** INSUFFICIENT
- **Answered by:** USER
- **Answer:** `float16` — option 1. It holds the resident set near 3.1 GB, inside the stated
  3-4 GB envelope, where `float32` would need roughly 6.2 GB. Probabilities are computed in
  `float32` after the gather, so the narrower storage dtype does not distort them.
- **Queued:** none

### Q4: When should the model be downloaded?
- **Question:** Should the ~3.1 GB model download happen during build, or during planning so
  measured latency can be reported before approval? Options: (1) at build time;
  (2) now. Recommendation: (1).
- **Verdict:** INSUFFICIENT
- **Answered by:** USER
- **Answer:** At build time — option 1. Planning verified every mechanism against a tiny
  stand-in model, so the large download belongs to the run that actually needs the weights.
- **Queued:** none
