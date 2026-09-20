<!-- doc-type: concept -->

# jev-local-engine

A local decision engine for short text. It reads a small language model's
internal state and returns typed decisions, rather than generating words.

The engine classifies an input by category, urgency, and sentiment, and routes
it to one of 110 tools. A decision takes about 33 milliseconds on an Apple
Silicon laptop, and the model holds 1.81 GB of memory.

```python
import decision_heads as D

tools = D.load_heads("data/tool_head.pt", D.TOOL_SCHEMA)
gate = D.load_heads("data/gate_head.pt", D.GATE_SCHEMA)

result = D.route_tool("the walk-in freezer packed up overnight", tools, gate)
# {'is_tool_request': True, 'tool': 'cold_chain_breach_report', ...}
```

## Documentation

- **[What this engine is](docs/what-it-is.md)** — the two paths, and why one
  runs six times faster than the other.
- **[Using the engine](docs/using-the-engine.md)** — install, run, and train
  on your own examples.
- **[Measured results](docs/measured-results.md)** — accuracy, latency,
  caveats, and known limits.

## Install

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt
PYTHONPATH=. .venv/bin/python -m pytest tests/
```

The first run downloads about 3.1 GB of model weights. The engine needs an
Apple Silicon machine, and it exits rather than falling back to the processor.

## What it does well, and what it does not

The engine puts the right tool in a shortlist of five about 95 percent of the
time. It names the right tool outright about 85 percent of the time, and 96
percent on the compliance tools.

The engine triages; it does not adjudicate. Do not place it behind an
automated refund or a compliance sign-off.

Three limits deserve attention before you build on it:

- An injected instruction captures the decision. The engine never invents a
  label outside the schema, but a caller must not read that as resistance.
- Confidence alone cannot detect text that no choice fits, which is why a
  separate gate head answers that question.
- The category field favours `technical_support` on text that mentions help.

[Measured results](docs/measured-results.md) carries the figures behind each
of these, with the conditions that produced them.
