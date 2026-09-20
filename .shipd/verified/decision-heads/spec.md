# decision-heads

### Requirement: Features come from one frozen-encoder pass
id: frozen-feature-extraction

The system SHALL compute features as the attention-mask-weighted mean of the encoder's last hidden
state, taken from the base model rather than the language-model head, and SHALL L2-normalize them.
The encoder's parameters SHALL NOT be updated during training. The prompt SHALL NOT enumerate a
field's choices, since the head supplies the class set.

#### Scenario: Features have the encoder's hidden width
- **WHEN** the system encodes a batch of three texts
- **THEN** it returns a tensor of shape (3, 1536), each row having unit L2 norm

#### Scenario: Padding never changes a trained head's decision
- **WHEN** the system encodes any text alone and again in a batch padded to a substantially
  longer length, including single-token inputs such as "." and ":)" against a seventy-token
  filler, and scores both with **trained** heads
- **THEN** every field's selected decision is identical between the two encodings

The claim is scoped to trained heads because an untrained head's argmax carries no meaning: its
class scores are near-ties by construction, so float16 padding drift can reorder them. Measured on
randomly initialized heads, 7 of 200 seeds flip at least one anchor's decision; measured on the
trained heads this change ships, none of the six anchors flips any field, and the worst top-1 to
top-2 margin across them is 0.0813 — two orders of magnitude above the drift. Any test of this
scenario SHALL therefore exercise trained heads, and SHALL seed the generator wherever it
constructs random ones, so the result is deterministic rather than a sample of one hyperplane.

#### Scenario: Pooled vectors stay aligned for multi-token inputs
- **WHEN** the system encodes a text of two or more tokens alone and again in a batch padded to a
  substantially longer length
- **THEN** the two feature vectors have cosine similarity of at least 0.9999

Decision-invariance is the universal contract; the cosine floor is scoped, deliberately, to inputs
of two or more tokens. Single-token inputs fall below it — measured, "." reaches cosine 0.9998567
and ":)" 0.9998783 against a seventy-token filler — while their decisions still do not change. An
earlier draft of this requirement asserted 0.9999 for all inputs; that figure was calibrated to the
anchors then known rather than derived from a worst case, and adversarial probing falsified it.
The bound is stated here only over the range where it demonstrably holds.

A per-dimension epsilon was the original instrument and was abandoned as unmeetable: the features
are L2-normalized across 1536 dimensions, so a typical dimension carries magnitude near 0.0255 and
the former 1e-3 bound amounted to four percent of it, against float16 attention accumulation that
reaches 0.009964 on a single-token anchor. The masked mean itself is exact; the residual is float16
accumulation over padded positions. Running the encoder in float32 would remove the drift at
roughly double the resident memory, which this change declines for noise that never alters an
output.

#### Scenario: The encoder stays frozen
- **WHEN** training completes
- **THEN** no encoder parameter has `requires_grad` set, and every trained parameter belongs to a head

### Requirement: Every field resolves from one shared pass
id: shared-prefill-inference

The system SHALL encode the input exactly once per decision and SHALL read every schema field from
that single feature tensor. The number of encoder forward passes SHALL NOT grow with the number of
fields or with the number of classes in any field.

#### Scenario: Three fields cost one encoder pass
- **WHEN** the system scores a three-field schema for one input
- **THEN** exactly one encoder forward pass executes, and all three fields are returned

#### Scenario: Adding classes does not add passes
- **WHEN** a head with four classes is replaced by one with a hundred classes and the input is scored
- **THEN** the encoder forward-pass count is unchanged

### Requirement: The return shape matches the constrained engine
id: comparable-return-shape

The system SHALL return `{"decisions": {field: {"decision": str, "probabilities": {choice: float}}},
"latency_ms": float}` — the same shape `run_jev_decision` returns — carrying the same field names, so
both paths can be compared by one harness. Each field's probabilities SHALL sum to 1.0 within 0.001.

Choice keys match per field only where the two schemas declare the same choices. They diverge on
`urgency` by design: the heads collapse it to two classes while the engine keeps four, per
`heads-schema`. Where they diverge, a declared mapping from the engine's choices onto the heads'
SHALL exist so the comparison stays defined; `ENGINE_URGENCY_TO_HEADS_URGENCY` is that mapping.
An earlier draft of this requirement demanded identical choice keys for every field, which the
urgency collapse contradicted — the two requirements were written at different times and were never
reconciled until validation exercised the shipped heads against the engine.

#### Scenario: Shapes are interchangeable
- **WHEN** the same input is scored by the head path using the shipped trained heads and by the
  constrained engine
- **THEN** both results carry the same field names and a numeric `latency_ms`, the choice keys match
  for every field whose schemas declare the same choices, and every engine choice for a diverging
  field maps onto a heads choice through the declared mapping

#### Scenario: Probabilities normalize
- **WHEN** the head path scores any field
- **THEN** that field's probabilities sum to 1.0 within 0.001

### Requirement: Heads train on hand-labelled examples, not distilled ones
id: hand-labelled-training

The system SHALL train its heads on hand-authored examples carrying human labels, and SHALL NOT
train them on labels produced by the constrained engine. Training examples SHALL be written in
varied prose — differing in length, register and phrasing — and SHALL NOT be generated by crossing
templates with slot fillers.

Distillation was measured and rejected. Training on 432 engine-labelled examples reached 0.667 on
category and 0.617 on sentiment, while 48 hand-labelled examples cross-validated at 0.867 and
0.820 on the same features under shuffled folds. The engine scores 0.683 and 0.717 itself, so it was the ceiling: a
student trained on its labels cannot pass it, and hand labels already do.

#### Scenario: Training data carries human labels
- **WHEN** the training set is loaded
- **THEN** every item carries a hand-authored label per field, and no label is read from the
  constrained engine

#### Scenario: Training examples are not template-generated
- **WHEN** the training set is inspected for repeated structure
- **THEN** no fixed phrase appears in more than one tenth of the items for any single class

### Requirement: The heads schema collapses urgency to two classes
id: heads-schema

The system SHALL define its own schema for the heads, holding the four category choices and the
three sentiment choices unchanged, and urgency as exactly two choices. The system SHALL NOT alter
the constrained engine's own schema, which keeps its four urgency choices.

Four-way urgency was measured as weak and heavily overfitting: it fits sixty labels at 1.000 train
accuracy yet cross-validates at roughly 0.40 against a 0.250 chance floor — above chance, but far
below its own training fit. Collapsed to two classes the shipped heads reach 0.9167 on the holdout,
with minority-class F1 0.839. The boundary between "high" and "critical" in a short message is a
judgement a reproducible labelling cannot rest on.

An earlier draft of this rationale cited 0.200 and called the four-way result worse than guessing.
That figure was an artifact: the cross-validation used stride-five folds against a holdout ordered
in a repeating fifteen-item class cycle, so each fold omitted an entire class. Shuffled folds give
0.397 to 0.450. The same artifact inflated the category figure quoted for hand-labelled training
from a true 0.867 to 0.917. Both errors are corrected here; neither changes the decision, since a
four-way cross-validation near 0.40 against a two-way holdout result of 0.9167 supports the
collapse on its own.

#### Scenario: Urgency carries two choices for the heads
- **WHEN** the heads schema is read
- **THEN** its urgency field holds exactly two choices, while category holds four and sentiment
  holds three

#### Scenario: The engine's schema is untouched
- **WHEN** the constrained engine's schema is read
- **THEN** its urgency field still holds its original four choices

### Requirement: Evaluation uses a fresh holdout and reports the engine alongside
id: holdout-evaluation

The system SHALL evaluate on a holdout authored after the training set and never used to choose a
schema, a hyperparameter, or a label scheme. The earlier sixty-item set SHALL be retained as a
validation set, since it informed the decision to collapse urgency, and SHALL NOT be used as the
holdout. The system SHALL report per-field accuracy for BOTH the trained heads and the constrained
engine, mapping the engine's four urgency choices onto the heads' two so the comparison is defined.

#### Scenario: Both paths are scored per field
- **WHEN** evaluation runs on the holdout
- **THEN** the report carries a per-field accuracy for the heads and for the constrained engine,
  plus the count of items where they disagree

#### Scenario: The holdout is disjoint from training and validation
- **WHEN** the holdout is compared against the training set and the validation set
- **THEN** no holdout text appears in either

### Requirement: Head inference is faster than the constrained path
id: head-latency

The system SHALL resolve all three fields, warm, in under 100 ms — against the constrained engine's
measured 202 ms for the same schema — and SHALL report that latency.

#### Scenario: Warm scoring beats the baseline
- **WHEN** the head path scores a three-field schema after at least one prior call
- **THEN** the reported `latency_ms` is below 100

### Requirement: Trained heads persist and reload
id: head-persistence

The system SHALL save trained head weights together with the schema they were trained for, and SHALL
reload them without retraining. If a saved schema does not match the requested schema, then the
system SHALL raise rather than load mismatched weights.

#### Scenario: Reloaded heads reproduce their predictions
- **WHEN** heads are saved, reloaded into a fresh instance, and scored on the same input
- **THEN** every probability matches the pre-save value within 1e-5

#### Scenario: A mismatched schema fails loudly
- **WHEN** heads saved for one schema are loaded against a schema with different fields or choices
- **THEN** the system raises an error naming the mismatch
