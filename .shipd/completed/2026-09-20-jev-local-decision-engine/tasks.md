## 1. Environment

- [x] 1.1 [req: env-bootstrap] Create `requirements.txt` at the repo root pinning
      `torch==2.14.0`, `transformers==5.17.0`, `accelerate`, and `pydantic==2.13.5`.
      Do not list `hf_transfer` — `huggingface_hub` 1.32 no longer uses it.
- [x] 1.2 [req: env-bootstrap] Create the environment with
      `uv venv .venv --python 3.14 && uv pip install --python .venv/bin/python -r requirements.txt`,
      then confirm `.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"`
      prints `True`.

## 2. Candidate resolution

- [x] 2.1 [req: candidate-resolution] Add `tests/test_jev_engine.py` with a test that
      `resolve_candidates` returns four distinct ids for the `urgency` choices, and a test
      that a field holding two same-first-token choices raises an error naming both. Run the
      tests and observe them fail — `jev_local_engine.py` does not exist yet.
- [x] 2.2 [req: candidate-resolution] Create `jev_local_engine.py` with
      `resolve_candidates(tokenizer, choices)` returning `{choice: tokenizer.encode(" " + choice,
      add_special_tokens=False)[0]}`, raising `ValueError` naming both choices and the shared id
      when two ids collide. Confirm the tests from 2.1 pass.

## 3. Probability normalization

- [x] 3.1 [req: candidate-normalized-probabilities] In `tests/test_jev_engine.py`, add a test
      that feeds a known logit vector and candidate id list to the probability helper and
      asserts the returned values sum to 1.0 within 0.001 and are not all equal. This is the
      regression test for the scalar-softmax defect. Run it and observe it fail.
- [x] 3.2 [req: candidate-normalized-probabilities] In `jev_local_engine.py`, add
      `candidate_probabilities(row_logits, candidate_ids)` that calls `.float()`,
      `index_select(0, candidate_ids)`, then `torch.softmax(..., dim=0)`. Never softmax a
      single scalar. Confirm the test from 3.1 passes.

## 4. Model loading

- [x] 4.1 [req: mps-model-load, env-bootstrap] In `jev_local_engine.py`, add `load_model()`
      that exits non-zero naming MPS when `torch.backends.mps.is_available()` is false, then
      calls `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
      dtype=torch.float16)` with no `device_map` argument, followed by `.to("mps")` and
      `.eval()`. Also load the tokenizer and set `padding_side = "left"`, falling back to
      `eos_token` when `pad_token` is unset.

## 5. Batched scoring

- [x] 5.1 [req: batched-field-scoring] In `tests/test_jev_engine.py`, add a test that builds
      a three-field batch through the prompt builder and asserts the tokenized batch holds
      exactly three rows. Run it and observe it fail.
- [x] 5.2 [req: batched-field-scoring] In `jev_local_engine.py`, add
      `build_field_prompts(text, schema)` that renders one prompt per field through
      `tokenizer.apply_chat_template`, ending each prompt with no trailing whitespace so the
      space-prefixed candidate ids match.
- [x] 5.3 [req: batched-field-scoring] In `jev_local_engine.py`, add `score_fields(...)` that
      tokenizes all field prompts with `padding=True`, runs exactly one `model(**batch)` under
      `torch.no_grad()`, reads `logits[:, -1, :]`, and applies `candidate_probabilities` per
      row. Confirm the test from 5.1 passes.

## 6. Schema-validated output

- [x] 6.1 [req: schema-validated-output] In `tests/test_jev_engine.py`, add a test that
      constructing `IntentDecision` with `category="not_a_category"` raises a Pydantic
      `ValidationError`. Run it and observe it fail.
- [x] 6.2 [req: schema-validated-output] In `jev_local_engine.py`, define `IntentDecision`
      with `category`, `urgency`, and `sentiment` typed as `Literal[...]` over their allowed
      choices, and add `run_jev_decision(text, schema)` returning `{"decisions": {field:
      {"decision": ..., "probabilities": ...}}, "latency_ms": ...}` with `latency_ms` timing
      scoring only, not model load. Confirm the test from 6.1 passes.

## 7. Batch invariance

- [x] 7.1 [req: batched-field-scoring] In `tests/test_jev_engine.py`, add a test that scores
      the `sentiment` field inside the full three-field batch and again as a single-row batch,
      asserting every choice probability agrees within 0.01. Mark it as requiring the model.

## 8. Benchmark and verification

- [x] 8.1 [req: benchmark-report] In `jev_local_engine.py`, add a `__main__` block that loads
      the model, scores the input `"My account was double charged for last month's
      subscription, fix this immediately!"`, prints the JSON result, and prints scoring
      latency in milliseconds plus resident set size in gigabytes.
- [x] 8.2 [req: *] Run `.venv/bin/python jev_local_engine.py`, confirm it prints valid JSON,
      and record the observed latency and resident memory against the sub-500 ms and 3-4 GB
      targets.
- [x] 8.3 [req: candidate-normalized-probabilities] Run the engine on a calm positive message
      and confirm its decisions differ from the billing-complaint result, proving the engine
      responds to input rather than returning a fixed answer.

## 9. Full-sequence candidate scoring

- [x] 9.1 [req: full-sequence-scoring] In `tests/test_jev_engine.py`, add a test that scores the
      `category` field on a calm positive thank-you note and asserts `technical_support`
      receives a lower probability than the 0.99 it receives under first-token scoring. Mark it
      `requires_model`. Run it and observe it fail.
- [x] 9.2 [req: full-sequence-scoring] In `jev_local_engine.py`, add
      `score_fields_full_sequence(model, tokenizer, text, schema)` that builds one sequence per
      (field, candidate) pair as the field prompt followed by that candidate's token ids, runs a
      single batched forward pass over all of them, computes each candidate's mean per-token
      log-probability over its own tokens only, and softmaxes those means across each field's
      candidates.
- [x] 9.3 [req: full-sequence-scoring] In `jev_local_engine.py`, switch `run_jev_decision` to use
      `score_fields_full_sequence`, keeping `latency_ms` measured over scoring only. Confirm the
      test from 9.1 passes.
- [x] 9.4 [req: full-sequence-scoring] In `tests/test_jev_engine.py`, add a test asserting the
      candidate-scoring stage runs exactly one forward pass for the full three-field schema, and
      a test asserting each field's probabilities sum to 1.0 within 0.001. Confirm both pass.
- [x] 9.5 [req: *] Re-run the benchmark three times and record latency, current resident memory,
      and the full probability maps for both the billing complaint and the calm positive note.

## Token usage breakdown

| Tool | Calls | Output tokens |
| --- | --- | --- |
| Bash | 220 | 113.3k |
| (no tool) | 0 | 23.1k |
| Edit | 24 | 15.0k |
| Read | 11 | 7.9k |
| SendMessage | 5 | 5.6k |
| Agent | 3 | 2.4k |
| AskUserQuestion | 2 | 2.4k |
| Write | 5 | 2.1k |
| ListAgents | 1 | 278 |
| ToolSearch | 1 | 97 |
| **Total** | 272 | 172.2k |
