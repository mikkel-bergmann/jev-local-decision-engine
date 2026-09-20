## 1. Fold in the inspection queries

- [x] 1.1 [req: inspection-training-data] Confirm `data/tool_train_d.json` is present with its twenty
      inspection queries and that `train_heads.py` registers it in the tool training paths. Add a
      test asserting the loaded tool training data includes those queries and that each of
      `audit_checklist_run`, `health_inspection_schedule`, `health_inspection_report`,
      `health_violation_log` and `audit_schedule_create` has at least four queries.

## 2. Gate negatives

- [x] 2.1 [req: gate-training-data] Create `data/gate_negatives_a.json` with 80 hand-authored
      non-requests: greetings, questions with no action, factual questions, opinions, statements
      about the business, and small talk. Vary length and register; reuse no sentence frame.
- [x] 2.2 [req: gate-training-data] Create `data/gate_negatives_b.json` with 80 more, weighted toward
      hard cases: fragments, ambiguous requests naming no action, domain vocabulary used
      conversationally such as "our PCI audit went fine last year", and requests for things the
      catalogue cannot do such as "write a policy document for us".
- [x] 2.3 [req: gate-evaluation] Create `data/gate_holdout.json`, written last, with 40 positives
      drawn from newly-authored tool requests and 40 newly-authored negatives, sharing no text with
      any training file.
- [x] 2.4 [req: gate-training-data, gate-evaluation] Create `tests/test_tool_gate.py` with tests
      asserting the negatives number at least 150, that no four-word phrase appears in more than a
      tenth of them, and that the gate holdout is disjoint from all gate training data. Confirm they
      pass.

## 3. Gate head

- [x] 3.1 [req: tool-request-gate] In `decision_heads.py`, add
      `GATE_SCHEMA = {"is_tool_request": ["no", "yes"]}`.
- [x] 3.2 [req: tool-request-gate] In `train_heads.py`, add `load_gate_training_data()` taking the
      tool queries as positives and the authored negatives as negatives, and `train_gate_head()`
      training a `DecisionHeads(GATE_SCHEMA)` with a class weight offsetting the imbalance. Assert no
      encoder parameter is trainable. Save to `data/gate_head.pt`.
- [x] 3.3 [req: tool-request-gate] In `tests/test_tool_gate.py`, add a test loading the shipped
      `data/gate_head.pt` that asserts a clear tool request is accepted and a greeting, an arithmetic
      question and a poem request are all rejected. Add a test counting encoder forward passes for
      one gated routing decision and asserting exactly one. Confirm they pass.

## 4. Threshold

- [x] 4.1 [req: gate-threshold] In `train_heads.py`, add `choose_gate_threshold(gate, holdout_path)`
      sweeping thresholds from 0.10 to 0.90 in steps of 0.05, returning for each the share of
      negatives rejected and the share of genuine requests retained, and selecting the value that
      maximizes their harmonic mean.
- [x] 4.2 [req: gate-threshold] Run the sweep, then set `GATE_THRESHOLD` in `decision_heads.py` to
      the selected value with a comment recording the measured precision, recall and
      negative-rejection rate at that value, and the date of the sweep.
- [x] 4.3 [req: gate-threshold] In `tests/test_tool_gate.py`, add a test asserting the sweep returns
      both directions of the trade for every candidate, and that `GATE_THRESHOLD` lies within the
      swept range. Confirm they pass.

## 5. Gated routing

- [x] 5.1 [req: gated-routing] In `tests/test_tool_gate.py`, add tests that `route_tool` on a
      rejected query reports `is_tool_request` false, `tool` None and an empty `top_k`; that on an
      accepted query it returns the same tool the tool head alone returns; and that
      `gate_probability` is present either way. Run them and observe them fail.
- [x] 5.2 [req: gated-routing] In `decision_heads.py`, add
      `route_tool(text, tool_heads, gate_head, threshold=GATE_THRESHOLD)` encoding once and reusing
      those features for both heads. Leave `run_heads_decision` untouched. Confirm the tests pass and
      the existing suite stays green.

## 6. Evaluate and report

- [x] 6.1 [req: gate-evaluation] In `train_heads.py`, add `evaluate_gate(gate, holdout_path)`
      reporting negative-rejection rate, genuine-request retention, precision, recall and F1. Where
      any fold split is used, shuffle with a seeded generator first.
- [x] 6.2 [req: *] Train the gate end to end and report the chosen threshold, all evaluation figures
      at it, and the gate's verdict on these nine probes: "2+2", "hello", "what time is it", "tell me
      a joke", "what's the capital of France", "I really love this product", "explain quantum
      entanglement", "my dog ate my homework", "can you write me a poem about the sea". Every one
      should be rejected; report honestly which are not.
- [x] 6.3 [req: *] Retrain the tool head so it reflects the inspection queries, then re-run the tool
      holdout and report recall@5, recall@1 and never-predicted count against the recorded baseline
      of 0.9545, 0.8364 and 15.
- [x] 6.4 [req: *] Re-run the full suite in two separate processes and confirm green both times.

## 7. Close the narrative-incident gap

- [x] 7.1 [req: gate-training-data] In `data/gate_negatives_b.json`, add 30 negatives of the shape
      the gate currently fails: narrative past-tense statements about mundane incidents that ask for
      nothing. Cover domestic mishaps, workplace anecdotes, equipment annoyances and social
      recounting — text where something happened and the speaker is merely reporting it in
      conversation, not asking for it to be logged. Vary tense and register; reuse no frame.
- [x] 7.2 [req: tool-request-gate] Retrain the gate with the enlarged negative set, re-run the
      threshold sweep, and update `GATE_THRESHOLD` and its comment to the newly measured values.
      Do not keep the old number if the sweep selects a different one.
- [x] 7.3 [req: tool-request-gate] Re-test against these exact nine, all of which must now be
      rejected: "my dog ate my homework", "the cat knocked my coffee off the desk this morning",
      "yeah that email thread got out of hand fast", "the printer jammed again yesterday",
      "someone left the freezer door open last night", "we had a power cut on Sunday",
      "the delivery driver turned up late", "a customer spilled a drink near the till",
      "the alarm went off by accident at closing". Report each verdict with its probability.
- [x] 7.4 [req: gate-evaluation] Re-run the gate holdout evaluation and report whether the enlarged
      negatives moved the genuine-request retention rate. A gate that now rejects narrative text may
      also start rejecting real incident-logging requests such as "log that the freezer failed
      overnight" — test that exact phrasing and report it.
- [x] 7.5 [req: *] Re-run the full suite in two separate processes and confirm green both times.

