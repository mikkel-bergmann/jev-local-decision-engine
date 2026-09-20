## ADDED Requirements

### Requirement: A tool catalogue spans general and compliance-domain tools
id: tool-catalogue

The system SHALL carry a catalogue of at least one hundred tool names in `data/tools.json`, mixing
general-purpose developer and SaaS tools with tools for a quick-service-restaurant operator's PCI
and food-safety compliance. The catalogue SHALL include several near-neighbour families whose
members share a prefix and a purpose, so routing is exercised where it is hardest. Names SHALL be
namespaced in `area_action` form.

#### Scenario: The catalogue is large and mixed
- **WHEN** the catalogue is loaded
- **THEN** it holds at least one hundred uniquely-named tools, of which at least thirty concern PCI
  or food-safety compliance

#### Scenario: Near-neighbour families are present
- **WHEN** the catalogue is grouped by the segment before the first underscore
- **THEN** at least five groups hold three or more members

### Requirement: A tool head routes a query to a tool name
id: tool-routing-head

The system SHALL score a query against the tool catalogue using one head over the shared encoder
features, adding no encoder forward pass beyond the one a decision already costs. The head SHALL
return a tool name drawn from the catalogue and a probability for every tool, and SHALL NOT return
arguments, parameters, or any generated text.

#### Scenario: A query routes to a catalogue tool
- **WHEN** the head scores a natural-language query
- **THEN** the returned decision is a name from the catalogue, and the probability map's keys are
  exactly the catalogue's names

#### Scenario: Routing costs one encoder pass
- **WHEN** a query is scored against the full tool catalogue
- **THEN** exactly one encoder forward pass executes

#### Scenario: Routing returns no arguments
- **WHEN** the head scores any query
- **THEN** the returned record carries no argument or parameter key at any depth

### Requirement: Scoring returns a ranked shortlist
id: top-k-output

The system SHALL return, per field, a `top_k` list of the highest-probability choices as
`{"choice", "probability"}` entries ordered by descending probability, with the list length
configurable and defaulting to five. The system SHALL retain the existing `decision` and
`probabilities` keys unchanged, and the first `top_k` entry SHALL equal `decision`.

#### Scenario: The shortlist defaults to five and is ordered
- **WHEN** a query is scored without a length given
- **THEN** each field's `top_k` holds five entries for a field with at least five choices, ordered
  by descending probability, and its first entry's choice equals that field's `decision`

#### Scenario: The shortlist length is configurable
- **WHEN** a query is scored with a length of three
- **THEN** each field's `top_k` holds three entries

#### Scenario: Existing keys are unchanged
- **WHEN** a query is scored
- **THEN** every field still carries `decision` and `probabilities` with their previous meaning

### Requirement: Tool training data is hand-authored and intent-bearing
id: tool-training-data

The system SHALL train the tool head on at least three hand-authored natural-language queries per
tool, and SHALL NOT derive any tool label from the constrained engine. At least one query per tool
SHALL avoid every word appearing in that tool's own name, so the head learns intent rather than
string matching.

#### Scenario: Every tool carries at least three queries
- **WHEN** the tool training data is loaded
- **THEN** every catalogue tool has at least three queries, and every query's label is a catalogue name

#### Scenario: Some queries avoid the tool's own words
- **WHEN** each tool's queries are compared against the words in its name
- **THEN** at least one query per tool shares no word with that name

### Requirement: Tool evaluation leads with recall at five
id: tool-evaluation

The system SHALL evaluate the tool head on a holdout authored after the training data and disjoint
from it, reporting recall@5 as the headline figure, recall@1 alongside it, and the number of
catalogue tools the head never predicts across the holdout. Any cross-validation or fold split
SHALL shuffle with a seeded generator before partitioning.

#### Scenario: The report leads with recall at five
- **WHEN** evaluation runs on the tool holdout
- **THEN** the report carries `recall_at_5`, `recall_at_1`, and a count of tools never predicted

#### Scenario: Dead tools are named, not averaged away
- **WHEN** the head never predicts some catalogue tools across the holdout
- **THEN** the report lists those tool names rather than only counting them

#### Scenario: The tool holdout is disjoint from tool training data
- **WHEN** the holdout queries are compared against the training queries
- **THEN** no query text appears in both

## MODIFIED Requirements

### Requirement: The return shape matches the constrained engine
id: comparable-return-shape
base: 23f19275fa0f

The system SHALL return `{"decisions": {field: {"decision": str, "probabilities": {choice: float},
"top_k": list}}, "latency_ms": float}`, carrying the same field names as `run_jev_decision`, so both
paths can be compared by one harness. Each field's probabilities SHALL sum to 1.0 within 0.001. The
`top_k` key is additive: `decision` and `probabilities` keep their previous meaning, and the engine's
own return shape is unchanged.

Choice keys match per field only where the two schemas declare the same choices. They diverge on
`urgency` by design: the heads collapse it to two classes while the engine keeps four, per
`heads-schema`. Where they diverge, a declared mapping from the engine's choices onto the heads'
SHALL exist so the comparison stays defined; `ENGINE_URGENCY_TO_HEADS_URGENCY` is that mapping. A
tool schema has no engine counterpart at all, so the comparison applies only to fields both paths
declare.

#### Scenario: Shapes are interchangeable
- **WHEN** the same input is scored by the head path using the shipped trained heads and by the
  constrained engine
- **THEN** both results carry the same field names and a numeric `latency_ms`, the choice keys match
  for every field whose schemas declare the same choices, and every engine choice for a diverging
  field maps onto a heads choice through the declared mapping

#### Scenario: Probabilities normalize
- **WHEN** the head path scores any field
- **THEN** that field's probabilities sum to 1.0 within a tolerance of 0.001
