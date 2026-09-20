<!-- doc-type: how-to -->

# Using the engine

This guide installs the engine, runs each path, and retrains a head on your
own examples.

## Install

The engine needs Python 3.10 or later and an Apple Silicon machine. It refuses
to run on a processor without Metal support, rather than falling back to the
processor in silence.

1. Create the environment and install the dependencies.

   ```bash
   uv venv .venv --python 3.14
   uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt
   ```

2. Confirm Metal is available.

   ```bash
   .venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
   ```

The first run downloads about 3.1 GB of model weights. Set
`HF_HUB_DISABLE_XET=1` before any command that reaches the model host, because
the newer transfer path stalls on some machines.

## Classify text against the default schema

Run the head path for category, urgency, and sentiment.

```python
import decision_heads as D

heads = D.load_heads("data/heads.pt", D.HEADS_SCHEMA)
result = D.run_heads_decision("My card was charged twice", heads)

print(result["decisions"]["category"]["decision"])
print(result["latency_ms"])
```

Each field returns a chosen answer, a probability for every choice, and a
ranked shortlist under `top_k`.

## Route a request to a tool

Consult the gate head first, so the engine can abstain on text that asks for
nothing.

```python
import decision_heads as D

tools = D.load_heads("data/tool_head.pt", D.TOOL_SCHEMA)
gate = D.load_heads("data/gate_head.pt", D.GATE_SCHEMA)

result = D.route_tool("the card reader keeps declining chip payments", tools, gate)
```

`route_tool` returns four keys. Read `is_tool_request` first: when it is
false, `tool` is `None` and `top_k` is empty.

| key | meaning |
|---|---|
| `is_tool_request` | Whether the gate head accepted the text. |
| `gate_probability` | The gate head's confidence, reported either way. |
| `tool` | The chosen tool name, or `None` on a refusal. |
| `top_k` | The ranked shortlist, empty on a refusal. |

Pass `top_k=10` to widen the shortlist. Narrow 110 tools to five, because the
shortlist reads far more reliably than the single best answer.

## Classify without training data

Use the constrained decoding path for a schema that has no trained head.

```python
import jev_local_engine as J

result = J.run_jev_decision("My card was charged twice", J.SCHEMA)
```

This path accepts any schema of short answers. It runs about six times slower
than the head path, and scores lower on every field measured.

## Train a head on your own examples

Write examples by hand. Labels from the constrained decoding path score worse,
because a student never passes the teacher it copies.

1. Write at least three examples for each choice, as JSON records holding
   `text` and a label. Vary the length and the wording, and never reuse a
   sentence frame.
2. Write one example per choice that shares no word with that choice's name.
   Without it the head learns to match strings.
3. Hold back a further set, written last, as the test.
4. Train and save.

   ```bash
   PYTHONPATH=. .venv/bin/python -c "
   import train_heads as T, decision_heads as D
   heads = T.train_tool_head()
   D.save_heads(heads, 'data/tool_head.pt')"
   ```

5. Score the held-back set and read both directions of the result.

   ```bash
   PYTHONPATH=. .venv/bin/python -c "
   import train_heads as T, decision_heads as D
   print(T.evaluate_tool_head(D.load_heads('data/tool_head.pt', D.TOOL_SCHEMA)))"
   ```

Accuracy still climbs steeply at three examples per choice. Write five or ten
where the choice matters.

## Run the tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/
```

Tests that need the model weights skip cleanly when the weights are absent.
