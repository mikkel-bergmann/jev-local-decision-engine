# batch-and-benchmark
Status: verified

## Idea

Add batch entry points to every scoring path, and a benchmark that reports how ten prompts behave
in time and memory.

### Motivation

Ten prompts cost 363.7 ms one at a time and 108.9 ms through a single batched encode, but no caller
can reach that gain because every entry point takes one string. The repository also carries no
benchmark, so every figure in its documentation came from a script that was written, read once, and
thrown away.

### Details

- Add `run_heads_decision_batch` and `route_tool_batch` to `decision_heads.py`.
- Add `run_jev_decision_batch` to `jev_local_engine.py`.
- Add `benchmark.py` at the repository root, reporting both modes over ten prompts.
- Add `tests/test_batch_scoring.py`, asserting each batch entry point agrees with its single-input
  counterpart.

Affected capabilities: `decision-heads` and `local-decision-engine` gain batch requirements, and a
new `benchmarking` capability holds the benchmark's contract. Impact: `decision_heads.py`,
`jev_local_engine.py`, `benchmark.py`, `tests/test_batch_scoring.py`. No new dependencies.

### Non-goals

- No batching of `score_fields_full_sequence`, which needs 1.2 GB for its log-probabilities at ten
  prompts and grows linearly from there.
- No scaling curve across batch sizes; the benchmark reports ten prompts.
- No pytest threshold on timing, which would fail on a loaded machine rather than on a regression.
- No threads, no processes, and no asynchronous execution.
- No change to any existing single-input entry point.

## Implementation

**Files.** `decision_heads.py`, `jev_local_engine.py`, `benchmark.py`, `tests/test_batch_scoring.py`.

**Interfaces.**

- `run_heads_decision_batch(texts, heads, top_k=5) -> list[dict]` — encodes every text in one pass
  and returns one result per text, each matching `run_heads_decision`'s shape.
- `route_tool_batch(texts, tool_heads, gate_head, threshold=GATE_THRESHOLD) -> list[dict]` — one
  encode shared by the gate head and the tool head, returning one result per text.
- `run_jev_decision_batch(texts, schema) -> list[dict]` — builds every text's field prompts into one
  batch and reads each row's final position.
- Each batch function reports `latency_ms` per result as that batch's total divided by its length,
  so a caller reads a comparable per-prompt figure.

**Decisions.**

- **Batch by sharing one encode, not by threading.** The measured gain comes from one forward pass
  over ten rows rather than ten passes over one row: 363.7 ms falls to 108.9 ms, a factor of 3.3.
  Threads would contend for the same device and add no parallelism.
- **Expect little from the constrained path.** Measured, its ten prompts fall from 1941 ms to
  1554 ms, a factor of 1.2. That path builds one prompt per field per text and enumerates every
  choice inside each prompt, so it is already compute-bound where the head path was
  overhead-bound. The requirement asks for the batch entry point, not for a speedup it cannot give.
- **Leave the full-sequence path alone.** It computes log-probabilities across the whole vocabulary
  for every position, which at ten prompts needs 1.2 GB and rises with the batch. Batching it would
  scale a known defect rather than a feature.
- **Assert agreement, never timing, in the tests.** A batch result must match its single-input
  counterpart within a tolerance; a timing assertion would fail on a busy machine and teach the
  reader to ignore it.
- **Report memory around the run, not once.** The benchmark reads resident size before and after,
  because the useful question is whether repeated scoring grows it. Measured across twenty
  decisions, it moved 6 MB, from 1.711 GB to 1.717 GB.

**Risks and trade-offs.**

- A long text in a batch pads every shorter row, so a mixed batch costs more than its average text.
  The benchmark uses prompts of realistic and varied length rather than ten copies of one string.
- Per-prompt latency derived from a batch total hides the spread. The benchmark reports the
  sequential minimum, median and maximum alongside it, so the reader sees both.
- A benchmark measuring a warm process understates a caller's first request, which pays about 4.9
  seconds of model load. The benchmark reports that load separately.
