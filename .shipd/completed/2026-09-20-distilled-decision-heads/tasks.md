## 1. Feature extraction

- [x] 1.1 [req: frozen-feature-extraction] Create `tests/test_decision_heads.py` with a test that
      `encode(["a", "b", "c"])` returns shape (3, 1536) with unit-norm rows, and a test that a text
      encoded alone matches the same text encoded inside a padded batch within 1e-3 per dimension.
      Mark both `requires_model`. Run them and observe them fail — `decision_heads.py` does not
      exist yet.
- [x] 1.2 [req: frozen-feature-extraction] Create `decision_heads.py` with `encode(texts, batch_size=16)`
      that runs `model.model(**batch)` under `torch.no_grad()`, takes the attention-mask-weighted
      mean of `last_hidden_state`, casts to float32, and L2-normalizes. Reuse the engine's cached
      model via `jev_local_engine._get_model_and_tokenizer`. Confirm the tests from 1.1 pass.

## 2. Heads module

- [x] 2.1 [req: shared-prefill-inference, comparable-return-shape] In `tests/test_decision_heads.py`,
      add a test that `DecisionHeads(SCHEMA)` scoring one input returns all three fields with
      probabilities summing to 1.0 within 0.001, and a test counting encoder forward passes for one
      decision, asserting exactly one. Run them and observe them fail.
- [x] 2.2 [req: shared-prefill-inference] In `decision_heads.py`, add `class DecisionHeads(torch.nn.Module)`
      holding a `torch.nn.ModuleDict` of `Linear(1536, len(choices))` per field, whose `forward(features)`
      applies every head to the same feature tensor and returns `{field: probabilities}` via softmax.
- [x] 2.3 [req: comparable-return-shape] In `decision_heads.py`, add `run_heads_decision(text, heads)`
      returning `{"decisions": {field: {"decision", "probabilities"}}, "latency_ms"}`, timing the
      encode plus head forward only. Confirm the tests from 2.1 pass.

## 3. Corpus and labelling

- [x] 3.1 [req: hand-labelled-training] Create `train_heads.py` with `build_corpus()` that crosses a
      list of message templates with slot fillers to produce a deterministic ordered list of at
      least 300 texts spanning all four categories, all four urgencies, and all three sentiments.
      Use no randomness; identical output on every run.
- [x] 3.2 [req: hand-labelled-training] In `tests/test_decision_heads.py`, add a test asserting
      `build_corpus()` returns an identical list across two calls, and that it contains at least 300
      items. Confirm it passes.
- [x] 3.3 [req: hand-labelled-training] In `train_heads.py`, add `label_corpus(texts)` that scores each
      text with `jev_local_engine.run_jev_decision` and returns one label per field per text, each
      drawn from that field's declared choices. Cache labels to `data/corpus_labels.json` so a rerun
      does not re-score.

## 4. Holdout

- [x] 4.1 [req: holdout-evaluation] Create `data/holdout.json` holding at least 60 hand-authored
      items, each with free-prose text and a label for all three fields, covering every choice of
      every field. Write them in prose that shares no template with `build_corpus()`.
- [x] 4.2 [req: holdout-evaluation] In `tests/test_decision_heads.py`, add a test asserting no
      holdout text appears in `build_corpus()` output and that every holdout label is a declared
      choice for its field. Confirm it passes.

## 5. Training

- [x] 5.1 [req: hand-labelled-training, frozen-feature-extraction] In `train_heads.py`, add `train(...)`
      that encodes the labelled corpus once, then trains the heads with AdamW and cross-entropy.
      Assert before training that no encoder parameter has `requires_grad` set and that the optimizer
      holds only head parameters.
- [x] 5.2 [req: frozen-feature-extraction] In `tests/test_decision_heads.py`, add a test asserting
      that after a short training run every parameter with a gradient belongs to a head and none
      belongs to the encoder. Confirm it passes.

## 6. Evaluation

- [x] 6.1 [req: holdout-evaluation] In `train_heads.py`, add `evaluate(heads)` that scores the
      holdout with BOTH the trained heads and `jev_local_engine.run_jev_decision`, returning
      per-field accuracy for each and the count of items where they disagree.
- [x] 6.2 [req: holdout-evaluation] In `tests/test_decision_heads.py`, add a test asserting the
      evaluation report carries a per-field accuracy for both paths and a disagreement count, and
      that reporting student accuracy without teacher accuracy is not possible from the returned
      structure. Confirm it passes.

## 7. Persistence

- [x] 7.1 [req: head-persistence] In `tests/test_decision_heads.py`, add a test that saving heads,
      reloading them into a fresh instance and scoring the same input reproduces every probability
      within 1e-5, and a test that loading weights against a schema with different fields raises an
      error naming the mismatch. Run them and observe them fail.
- [x] 7.2 [req: head-persistence] In `decision_heads.py`, add `save_heads(heads, path)` and
      `load_heads(path, schema)` persisting weights alongside the schema, raising on mismatch.
      Confirm the tests from 7.1 pass.

## 8. Latency and report

- [x] 8.1 [req: head-latency] In `tests/test_decision_heads.py`, add a test that scores once to warm,
      then asserts a second `run_heads_decision` reports `latency_ms` below 100. Confirm it passes.
- [x] 8.2 [req: *] Run the full training end to end, then report: per-field holdout accuracy for the
      heads and for the constrained engine, their disagreement count, warm head latency, and the
      constrained engine's latency on the same inputs. State plainly whether any per-field gap is
      the student's error or the teacher's ceiling.

## 9. Padding-invariance contract

- [x] 9.1 [req: frozen-feature-extraction] In `tests/test_decision_heads.py`, replace the
      per-dimension padding assertion with the revised contract: encode a four-token text alone and
      again batched against a sixty-token text, and assert cosine similarity is at least 0.9999 and
      that every field's selected decision is identical between the two encodings. Keep the test
      `requires_model`. Include the measured drift in the assertion message.
- [x] 9.2 [req: *] Re-run the full suite and confirm it is green, reporting the measured cosine and
      per-dimension drift for that pair.

## 10. Scope the padding contract to what holds

- [x] 10.1 [req: frozen-feature-extraction] In `tests/test_decision_heads.py`, split the padding
      test to match the revised scenarios: assert decision-invariance for EVERY anchor including the
      single-token ".", ":)" and "!", and assert the cosine floor of 0.9999 only for anchors of two
      or more tokens. Put each anchor's measured cosine and max per-dimension diff in the assertion
      messages, and comment that single-token anchors fall below the floor by measurement.
- [x] 10.2 [req: *] Re-run the full suite and report, per anchor, its token count, measured cosine,
      max per-dimension diff, and whether any decision flipped.

## 11. Test the artifact that ships

- [x] 11.1 [req: frozen-feature-extraction] In `tests/test_decision_heads.py`, change
      `test_padding_never_changes_a_decision` to score with the TRAINED heads loaded from
      `data/heads.pt` rather than a fresh `DecisionHeads(SCHEMA)`, skipping the test when that file
      is absent. Assert decision-invariance for all six anchors and report the worst top-1 to top-2
      margin in the assertion message.
- [x] 11.2 [req: frozen-feature-extraction] Add `torch.manual_seed(0)` before every random
      `DecisionHeads(...)` construction in the test module, so no test samples an unseeded
      hyperplane. Note in a comment that 7 of 200 seeds flip an anchor with untrained heads.
- [x] 11.3 [req: *] Re-run the full suite five times in five separate processes and confirm it is
      green every time, reporting the worst trained-head margin observed.

## 12. Hand-labelled training set

- [x] 12.1 [req: heads-schema] In `decision_heads.py`, define `HEADS_SCHEMA` with `category` and
      `sentiment` copied from `jev_local_engine.SCHEMA` and `urgency` as exactly
      `["not_urgent", "urgent"]`. Do not modify `jev_local_engine.SCHEMA`. Add a test asserting the
      heads schema has two urgency choices and that the engine's still has four.
- [x] 12.2 [req: hand-labelled-training] Create `data/train_a.json` with 70 hand-authored items,
      each carrying `text` and a label for `category`, `urgency` (using the two-class scheme) and
      `sentiment`. Write genuinely varied prose: vary length from a few words to several sentences,
      vary register between terse, formal and chatty, and never reuse a sentence frame. Cover every
      choice of every field.
- [x] 12.3 [req: hand-labelled-training] Create `data/train_b.json` with 70 more items under the
      same rules, deliberately using different sentence openings and vocabulary from `train_a.json`.
- [x] 12.4 [req: hand-labelled-training] Create `data/train_c.json` with 60 more items under the
      same rules, weighted toward cases you expect to be hard: mixed sentiment, implicit urgency
      with no urgency words, and category boundaries such as a billing question phrased as a
      complaint.
- [x] 12.5 [req: hand-labelled-training] Create `data/holdout_v2.json` with 60 fresh items written
      last, under the same rules, covering every choice of every field. This is the holdout and must
      share no text with any training file or with `data/holdout.json`.
- [x] 12.6 [req: hand-labelled-training] In `tests/test_decision_heads.py`, add a test asserting the
      combined training set holds at least 200 items, that no fixed phrase of four or more words
      appears in more than one tenth of the items for any class, and that `data/holdout_v2.json` is
      disjoint from every training file and from `data/holdout.json`.

## 13. Retrain and re-measure

- [x] 13.1 [req: hand-labelled-training] In `train_heads.py`, replace the distillation path: load
      the hand-labelled training files instead of calling `label_corpus`, and train against
      `HEADS_SCHEMA`. Keep `build_corpus` and `label_corpus` in the file but no longer call them
      from the training path. Retain the frozen-encoder assertion.
- [x] 13.2 [req: holdout-evaluation] Update `evaluate` to score `data/holdout_v2.json`, mapping the
      constrained engine's four urgency choices onto the two-class scheme (`low` and `medium` to
      `not_urgent`, `high` and `critical` to `urgent`) so the engine comparison is defined. Keep
      reporting heads accuracy, engine accuracy and disagreement count together per field.
- [x] 13.3 [req: *] Retrain end to end, save the new weights, and report per-field heads accuracy,
      engine accuracy and disagreement on `data/holdout_v2.json`, plus warm latency. Compare against
      the distilled baseline: category 0.667, urgency 0.333 four-way, sentiment 0.617.
- [x] 13.4 [req: frozen-feature-extraction] Re-run the padding test against the newly trained heads
      and report the worst top-1 to top-2 margin, confirming no decision flips.

## 14. Test the shipped heads against the engine

- [x] 14.1 [req: comparable-return-shape] In `tests/test_decision_heads.py`, change the
      shape-comparison test to score with the SHIPPED heads loaded from `data/heads.pt` under
      `HEADS_SCHEMA`, not a fresh `DecisionHeads(SCHEMA)`. Assert both paths return the same field
      names and a numeric `latency_ms`, that choice keys match for `category` and `sentiment`, and
      that every engine `urgency` choice maps onto a heads choice through
      `ENGINE_URGENCY_TO_HEADS_URGENCY`. Skip when `data/heads.pt` is absent.
- [x] 14.2 [req: comparable-return-shape] Audit `tests/test_decision_heads.py` for any other test
      that constructs `DecisionHeads(SCHEMA)` where it should exercise `HEADS_SCHEMA` or the shipped
      checkpoint, and correct each one. Report every site you changed and every site you judged
      correct as written, with the reason.
- [x] 14.3 [req: *] Re-run the full suite in two separate processes and confirm green both times.

## 15. Document the heads path

- [x] 15.1 [req: hand-labelled-training, heads-schema] Update `README.md` to cover the heads path
      alongside the constrained engine: what `decision_heads.py` provides, how to train and run it,
      the measured per-field accuracy against the engine on `data/holdout_v2.json`, and the measured
      latency. State plainly which path a reader should use and why.
- [x] 15.2 [req: holdout-evaluation] In `README.md`, record the two evaluation caveats: the holdout's
      urgency split is 46 not_urgent to 14 urgent, so quote minority-class F1 0.839 alongside the
      0.917 accuracy; and the engine's category accuracy differs between holdouts (0.683 on the old
      set, 0.533 on holdout_v2), so neither figure should be quoted as the engine's accuracy without
      naming which set it came from. Also note that `data/holdout.json` is now a validation set that
      shares authorship patterns with `data/train_a.json` (Jaccard up to 0.75), so numbers from it
      would be optimistic.
- [x] 15.3 [req: heads-schema] In `README.md`, explain why the heads use a two-class urgency while
      the engine keeps four, citing the measurement: four-way urgency fits 60 labels at 1.000 train
      accuracy and cross-validates at 0.200 against a 0.250 chance floor.

## Token usage breakdown

| Tool | Calls | Output tokens |
| --- | --- | --- |
| Bash | 420 | 300.8k |
| (no tool) | 0 | 45.3k |
| Write | 14 | 31.5k |
| Edit | 23 | 19.8k |
| Read | 43 | 8.6k |
| SendMessage | 7 | 7.4k |
| Agent | 7 | 7.0k |
| AskUserQuestion | 3 | 3.9k |
| ToolSearch | 1 | 111 |
| **Total** | 518 | 424.3k |
