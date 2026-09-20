## ADDED Requirements

### Requirement: Head scoring accepts many texts in one pass
id: batch-head-scoring

The system SHALL provide a batch entry point that scores a list of texts against the head schema,
encoding every text in one encoder forward pass. It SHALL return one result per text, in the order
given, each carrying the shape a single-input decision returns. The system SHALL NOT alter the
single-input entry point.

#### Scenario: A batch costs one encoder pass
- **WHEN** the system scores ten texts through the batch entry point
- **THEN** exactly one encoder forward pass executes, and ten results return in the order given

#### Scenario: A batch result matches a single result
- **WHEN** one text is scored alone and again inside a batch of ten
- **THEN** every field's chosen answer is identical, and every probability agrees within 0.01

#### Scenario: A single-text batch still works
- **WHEN** the batch entry point receives a list holding one text
- **THEN** it returns one result, matching what the single-input entry point returns for that text

### Requirement: Tool routing accepts many texts in one pass
id: batch-tool-routing

The system SHALL provide a batch routing entry point that consults the gate head and the tool head
for a list of texts, sharing one encoder forward pass between both heads and every text. Each result
SHALL carry the keys the single-text routing entry point returns, and a rejected text SHALL carry no
tool name and an empty shortlist.

#### Scenario: One pass serves both heads and every text
- **WHEN** the system routes ten texts through the batch entry point
- **THEN** exactly one encoder forward pass executes

#### Scenario: A rejected text abstains inside a batch
- **WHEN** a batch holds both a genuine request and text the gate head rejects
- **THEN** the rejected text's result reports no tool and an empty shortlist, while the genuine
  request returns a tool name

#### Scenario: Batch routing matches single routing
- **WHEN** one text is routed alone and again inside a batch
- **THEN** both results carry the same verdict and the same chosen tool
