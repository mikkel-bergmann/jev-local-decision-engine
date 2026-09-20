## ADDED Requirements

### Requirement: Constrained scoring accepts many texts in one pass
id: batch-constrained-scoring

The system SHALL provide a batch entry point that scores a list of texts against a schema through
the first-token path, building every text's field prompts into one batch and reading each row's
final position. The system SHALL NOT batch the full-sequence path, whose log-probabilities need
1.2 GB at ten texts and grow with the batch.

#### Scenario: A batch returns one result per text
- **WHEN** the system scores ten texts through the batch entry point
- **THEN** ten results return in the order given, each carrying the shape a single-input decision
  returns

#### Scenario: A batch result matches a single result
- **WHEN** one text is scored alone and again inside a batch of ten
- **THEN** every field's chosen answer is identical, and every probability agrees within 0.01

#### Scenario: The full-sequence path stays unbatched
- **WHEN** a reader looks for a batch entry point on the full-sequence path
- **THEN** none exists, and the single-input function is unchanged
