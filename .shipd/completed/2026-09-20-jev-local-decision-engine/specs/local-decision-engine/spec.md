## ADDED Requirements

### Requirement: Reproducible Apple Silicon environment
id: env-bootstrap

The system SHALL pin its runtime dependencies in `requirements.txt` covering `torch`,
`transformers`, `accelerate`, and `pydantic`, and SHALL NOT depend on `hf_transfer`.
Where `torch.backends.mps.is_available()` returns false, the engine SHALL exit non-zero
with a message naming MPS as unavailable rather than silently falling back to CPU.

#### Scenario: Dependencies install into a fresh virtual environment
- **WHEN** a user creates a virtual environment and installs `requirements.txt`
- **THEN** the install completes with a zero exit status and `import torch, transformers, pydantic` succeeds

#### Scenario: Missing MPS is reported, not hidden
- **WHEN** the engine starts on a machine where `torch.backends.mps.is_available()` is false
- **THEN** it exits non-zero and its message names MPS as unavailable

### Requirement: Model loads onto MPS without device_map
id: mps-model-load

The system SHALL load `Qwen/Qwen2.5-1.5B-Instruct` by calling `from_pretrained` with
`dtype=torch.float16` and no `device_map` argument, then moving the model with `.to("mps")`.
The system SHALL NOT pass `device_map="mps"`, which does not complete on this toolchain.

#### Scenario: Weights land on the MPS device in float16
- **WHEN** the engine loads the model
- **THEN** loading returns rather than hanging, and the first model parameter reports device
  `mps` and dtype `torch.float16`

### Requirement: Candidate token ids resolve without collision
id: candidate-resolution

The system SHALL map each schema choice to the first token id of the form that matches the
prompt's final token boundary: the bare form where the prompt ends in a newline, and the
space-prefixed form where it ends mid-line. The system SHALL NOT strip the trailing newline
that terminates a chat-template generation prompt. If two choices within one field resolve to
the same first token id, then the system SHALL raise an error naming both colliding choices
and the shared id.

#### Scenario: Distinct choices resolve to distinct ids
- **WHEN** the engine resolves candidates for the `urgency` choices `low`, `medium`, `high`, `critical`
- **THEN** it returns four distinct token ids and raises nothing

#### Scenario: Candidate form matches the prompt boundary
- **WHEN** the engine builds a field prompt through the chat template with a generation prompt
- **THEN** the prompt retains its terminating newline, and the candidate ids scored at that position
  are the bare-form first token ids rather than the space-prefixed ones

#### Scenario: Colliding choices fail loudly
- **WHEN** the engine resolves candidates for a field holding two choices that share a first token id
- **THEN** it raises an error naming both choices and the shared id

### Requirement: All fields score in one batched forward pass
id: batched-field-scoring

The system SHALL build one prompt per schema field, tokenize them together with left padding,
and obtain the logits for every field from a single `model(**batch)` call. The system SHALL
read each field's distribution from the final position of that field's own batch row.

#### Scenario: One forward pass covers every field
- **WHEN** the engine scores a three-field schema for one input
- **THEN** exactly one forward pass executes, and the batched logits carry one row per field

#### Scenario: Batched scoring matches unbatched scoring
- **WHEN** a field is scored inside the batch and again alone
- **THEN** the two probability distributions for that field agree within a tolerance of 0.01 per choice

### Requirement: Probabilities normalize across candidates
id: candidate-normalized-probabilities

The system SHALL gather a field's candidate logits into a single vector, cast it to float32,
and apply softmax across that vector, so the returned probabilities sum to 1 across the
field's choices. The system SHALL NOT apply softmax to an individual scalar logit.

#### Scenario: Probabilities are normalized and discriminating
- **WHEN** the engine scores any field
- **THEN** that field's probabilities sum to 1.0 within a tolerance of 0.001, and they are not all equal

#### Scenario: Input changes the selected class
- **WHEN** the engine scores a clearly negative, urgent billing complaint and then a calm positive
  thank-you note
- **THEN** the two inputs do not yield identical decisions across all three fields

### Requirement: Candidates score by length-normalized full-sequence likelihood
id: full-sequence-scoring

The system SHALL score each candidate by the mean per-token log-probability of its complete
token sequence continuing the field prompt, not by its first token alone. The system SHALL
obtain every field's every candidate from a single additional batched forward pass, and SHALL
convert the per-candidate mean log-probabilities of one field into probabilities by softmax
across that field's candidates.

#### Scenario: Multi-token choices compete on equal footing
- **WHEN** the engine scores the `category` field on a message with no technical-support content,
  such as a calm thank-you note
- **THEN** `technical_support` is not selected purely on its first token's prior, and its
  probability is below the probability it receives under first-token scoring

#### Scenario: Scoring uses one additional batched pass
- **WHEN** the engine scores a schema whose fields hold eleven candidates in total
- **THEN** the candidate-scoring stage executes exactly one forward pass covering all eleven
  candidate continuations, not one pass per candidate

#### Scenario: Per-field probabilities still normalize
- **WHEN** the engine scores any field under full-sequence scoring
- **THEN** that field's probabilities sum to 1.0 within a tolerance of 0.001

### Requirement: Output is schema-validated JSON with a latency figure
id: schema-validated-output

The system SHALL validate the selected decisions through a Pydantic model whose fields are
typed as literals over the allowed choices, and SHALL emit JSON carrying each field's chosen
decision, its per-choice probabilities, and an end-to-end `latency_ms` measured over scoring
only, excluding model load.

#### Scenario: Valid decisions serialize to JSON
- **WHEN** the engine finishes scoring a test input
- **THEN** it prints JSON holding a decision and a probability map per field, plus a numeric `latency_ms`

#### Scenario: Off-schema value is rejected
- **WHEN** a decision value outside a field's declared choices is passed to the Pydantic model
- **THEN** validation raises rather than accepting the value

### Requirement: Benchmark reports latency and resident memory
id: benchmark-report

The system SHALL report scoring latency in milliseconds and the process **current** resident set
size in gigabytes after the model is loaded, so both can be checked against the latency and 3-4 GB
targets. The latency budget for the full-sequence scoring path is **under 1000 ms**, measured
warm: the second batched pass over every candidate continuation costs roughly 3.5x the
first-token path, which the change accepts in exchange for length-normalized candidate scores. The system SHALL NOT report peak resident size in place of current, since peak
includes transient load-time buffers and overstates the steady-state footprint.

#### Scenario: A run prints both figures
- **WHEN** a user runs the engine on the test input
- **THEN** the output carries a scoring latency in milliseconds and a resident memory figure in gigabytes
