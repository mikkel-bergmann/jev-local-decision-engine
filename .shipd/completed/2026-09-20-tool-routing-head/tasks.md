## 1. Tool catalogue

- [x] 1.1 [req: tool-catalogue] Create `data/tools.json` holding at least 105 uniquely-named tools in
      `area_action` form. Include roughly 60 general-purpose developer and SaaS tools (for example
      `web_search`, `git_commit`, `k8s_deploy`, `stripe_refund`, `slack_send`) and at least 40 for a
      quick-service-restaurant operator's PCI and food-safety compliance (for example
      `pci_asv_scan_start`, `pci_saq_generate`, `haccp_checklist_submit`, `temp_log_fetch`,
      `allergen_matrix_update`, `health_inspection_report`). Build at least five near-neighbour
      families of three or more members sharing a prefix and purpose.
- [x] 1.2 [req: tool-catalogue] Create `tests/test_tool_routing.py` with a test asserting the
      catalogue holds at least 100 uniquely-named tools, at least 30 of which concern PCI or food
      safety, and that at least five prefix groups hold three or more members. Run it and confirm it
      passes.
- [x] 1.3 [req: tool-routing-head] In `decision_heads.py`, add `load_tool_catalogue()` returning the
      ordered names from `data/tools.json`, and `TOOL_SCHEMA = {"tool": load_tool_catalogue()}`.

## 2. Top-k return

- [x] 2.1 [req: top-k-output] In `tests/test_tool_routing.py`, add tests that `run_heads_decision`
      returns a `top_k` list of five `{"choice", "probability"}` entries per field by default,
      ordered by descending probability with its first choice equal to that field's `decision`; that
      passing a length of three returns three entries; and that `decision` and `probabilities` are
      unchanged. Mark them `requires_model`. Run them and observe them fail.
- [x] 2.2 [req: top-k-output, comparable-return-shape] In `decision_heads.py`, add a `top_k=5`
      parameter to `run_heads_decision` and populate each field's `top_k` from its probability map,
      leaving `decision` and `probabilities` untouched. Confirm the tests from 2.1 pass and the
      existing suite stays green.

## 3. Tool training data

- [x] 3.1 [req: tool-training-data] Create `data/tool_train_a.json` with three hand-authored queries
      for each of the first third of the catalogue, each carrying `text` and a `tool` label. Write
      natural queries a user would actually type, varying length and register. For each tool, make at
      least one query share no word with the tool's own name.
- [x] 3.2 [req: tool-training-data] Create `data/tool_train_b.json` covering the second third under
      the same rules.
- [x] 3.3 [req: tool-training-data] Create `data/tool_train_c.json` covering the final third under
      the same rules, taking extra care on the near-neighbour families so members are distinguished
      by intent rather than by borrowed wording.
- [x] 3.4 [req: tool-training-data] Create `data/tool_holdout.json`, written last, with one query per
      tool under the same rules, sharing no text with any training file.
- [x] 3.5 [req: tool-training-data] In `tests/test_tool_routing.py`, add tests asserting every
      catalogue tool has at least three training queries with labels drawn from the catalogue, that
      at least one query per tool shares no word with its tool name, and that the holdout is disjoint
      from every training file. Confirm they pass.

## 4. Train the tool head

- [x] 4.1 [req: tool-routing-head] In `train_heads.py`, add `load_tool_training_data()` reading the
      three tool training files, and `train_tool_head()` that encodes the queries once and trains a
      `DecisionHeads(TOOL_SCHEMA)` with the existing AdamW settings. Assert no encoder parameter is
      trainable. Save to `data/tool_head.pt`.
- [x] 4.2 [req: tool-routing-head] In `tests/test_tool_routing.py`, add a test counting encoder
      forward passes for one tool-routing decision and asserting exactly one, and a test asserting
      the returned decision is a catalogue name with probability keys equal to the catalogue. Confirm
      they pass.
- [x] 4.3 [req: tool-routing-head] In `tests/test_tool_routing.py`, add a test asserting the tool
      result carries no key named `argument`, `arguments`, `parameter` or `parameters` at any depth.
      Confirm it passes.

## 5. Evaluation

- [x] 5.1 [req: tool-evaluation] In `train_heads.py`, add `evaluate_tool_head(heads, holdout_path)`
      returning `recall_at_5`, `recall_at_1`, `unpredicted_tool_count`, and the list of tool names
      never predicted across the holdout. Where any fold split is used, shuffle with a seeded
      generator first — a stride-based split in the preceding change aliased with class ordering and
      produced a false result.
- [x] 5.2 [req: tool-evaluation] In `tests/test_tool_routing.py`, add tests asserting the report
      carries all four keys and that the never-predicted tools are listed by name rather than only
      counted. Confirm they pass.
- [x] 5.3 [req: *] Train the tool head end to end and report recall@5, recall@1, the number and names
      of tools never predicted, warm latency for a tool decision, and how the near-neighbour families
      behaved relative to the rest of the catalogue. Report the measured numbers whatever they are;
      if recall@5 is poor, say so and say which tools drove it.
- [x] 5.4 [req: *] Re-run the full suite in two separate processes and confirm green both times.

## 6. Document what the numbers mean

- [x] 6.1 [req: tool-evaluation] In `README.md`, add a tool-routing section covering what the head
      does, how to train and run it, and the measured recall@5, recall@1, warm latency, and the
      count and names of tools never predicted. Use only figures you re-measure yourself.
- [x] 6.2 [req: tool-evaluation] In `README.md`, record two caveats that materially qualify the
      headline. First, holdout items lexically closer to their training queries score far better:
      splitting the holdout at the median word-overlap gives recall@5 1.000 on the more-similar half
      against 0.909 on the less-similar half, so roughly 0.91 is the figure to expect on genuinely
      novel phrasing. Second, accuracy splits sharply by tool kind: the QSR and PCI compliance tools
      reach recall@1 0.949 and recall@5 0.974, while the general developer and SaaS tools reach
      0.789 and 0.944 — a catalogue of mostly generic tools would perform worse than this headline
      suggests. Re-measure both splits yourself rather than transcribing these numbers.
- [x] 6.3 [req: *] Re-run the full suite in two separate processes and confirm green both times.

## Token usage breakdown

| Tool | Calls | Output tokens |
| --- | --- | --- |
| Bash | 117 | 69.2k |
| Write | 5 | 26.5k |
| Edit | 17 | 15.0k |
| (no tool) | 0 | 8.8k |
| Read | 18 | 7.7k |
| Agent | 2 | 2.3k |
| SendMessage | 1 | 1.2k |
| **Total** | 160 | 130.7k |
