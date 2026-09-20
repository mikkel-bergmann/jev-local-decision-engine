## 1. Batch the head path

- [x] 1.1 [req: batch-head-scoring] In `tests/test_batch_scoring.py`, add a test that
      `run_heads_decision_batch` returns ten results for ten texts in the order given, a test
      counting encoder forward passes for one batch and asserting exactly one, and a test that a
      text scored alone matches the same text scored inside a batch within 0.01 per probability.
      Mark them `requires_model`. Run them and observe them fail.
- [x] 1.2 [req: batch-head-scoring] In `decision_heads.py`, add
      `run_heads_decision_batch(texts, heads, top_k=5)` calling `encode(texts)` once, applying every
      head to the resulting feature tensor, and returning one result per row in input order. Set each
      result's `latency_ms` to the batch total divided by the number of texts. Leave
      `run_heads_decision` untouched. Confirm the tests from 1.1 pass.
- [x] 1.3 [req: batch-head-scoring] In `tests/test_batch_scoring.py`, add a test that a list holding
      one text returns one result matching the single-input entry point. Confirm it passes.

## 2. Batch tool routing

- [x] 2.1 [req: batch-tool-routing] In `tests/test_batch_scoring.py`, add a test that
      `route_tool_batch` runs exactly one encoder forward pass for ten texts, a test that a batch
      mixing a genuine request with text the gate head rejects returns no tool and an empty
      shortlist for the rejected one, and a test that batch routing matches single routing. Mark
      them `requires_model`. Run them and observe them fail.
- [x] 2.2 [req: batch-tool-routing] In `decision_heads.py`, add
      `route_tool_batch(texts, tool_heads, gate_head, threshold=GATE_THRESHOLD)` encoding once and
      reusing those features for both heads. Return the same keys `route_tool` returns, per text.
      Leave `route_tool` untouched. Confirm the tests from 2.1 pass.

## 3. Batch the constrained path

- [x] 3.1 [req: batch-constrained-scoring] In `tests/test_batch_scoring.py`, add a test that
      `run_jev_decision_batch` returns ten results in order, and a test that a text scored alone
      matches the same text inside a batch within 0.01 per probability. Mark them `requires_model`.
      Run them and observe them fail.
- [x] 3.2 [req: batch-constrained-scoring] In `jev_local_engine.py`, add
      `run_jev_decision_batch(texts, schema)` that builds every text's field prompts, pads them into
      one batch, runs one forward pass, and reads each row's final position through
      `candidate_probabilities`. Do not call `score_fields_full_sequence`. Confirm the tests pass.
- [x] 3.3 [req: batch-constrained-scoring] In `tests/test_batch_scoring.py`, add a test asserting no
      batch entry point exists for the full-sequence path and that its single-input function is
      unchanged. Confirm it passes.

## 4. The benchmark

- [x] 4.1 [req: ten-prompt-benchmark] Create `benchmark.py` at the repository root holding ten
      prompts that differ in length and subject, drawn from the domains the catalogue covers.
      Include a short fragment and a long sentence, so padding cost shows.
- [x] 4.2 [req: benchmark-measures] In `benchmark.py`, time the model load on its own, then for each
      of the three paths score one prompt and discard it, then time the sequential run and the
      batched run. Call `torch.mps.synchronize()` around every timed region.
- [x] 4.3 [req: benchmark-measures] In `benchmark.py`, report per path and mode the total, the time
      per prompt, and the throughput in prompts per second; report the shortest, median and longest
      sequential prompt; and report resident memory before the run, after it, and at its peak.
- [x] 4.4 [req: ten-prompt-benchmark] In `tests/test_batch_scoring.py`, add a test asserting
      `benchmark.py` holds ten prompts and that they are not ten copies of one string. Assert no
      timing threshold. Confirm it passes.

## 5. Run and report

- [x] 5.1 [req: *] Run `benchmark.py` and report its full output, including whether the batched head
      path reproduces the roughly threefold gain measured during planning, and whether the
      constrained path reproduces its roughly 1.2-fold gain.
- [x] 5.2 [req: *] Re-run the full suite in two separate processes and confirm green both times.

## 6. Answer the memory question the report asks

- [x] 6.1 [req: benchmark-measures] In `benchmark.py`, repeat the whole prompt workload three times
      after the measured run, reporting resident memory after each repeat and the growth between
      repeats. A single before-and-after pair cannot distinguish one-time allocation from a leak,
      which is the question the report exists to answer.
- [x] 6.2 [req: benchmark-measures] In `benchmark.py`, state alongside those figures that the first
      pass carries the model load and the first allocation for each tensor shape, so a reader reads
      the later passes as the steady state. Measured for reference, verify rather than assume: pass
      one grew about 1550 MB and later passes grew about 0.1 MB each.
- [x] 6.3 [req: *] Run the benchmark again and report its full output, including the repeat-pass
      growth figures.
- [x] 6.4 [req: *] Re-run the full suite in two separate processes and confirm green both times.

## Token usage breakdown

| Tool | Calls | Output tokens |
| --- | --- | --- |
| Bash | 96 | 41.0k |
| Edit | 17 | 31.4k |
| Write | 6 | 12.9k |
| (no tool) | 0 | 8.0k |
| Read | 25 | 4.9k |
| Agent | 2 | 2.0k |
| SendMessage | 1 | 1.1k |
| **Total** | 147 | 101.3k |
