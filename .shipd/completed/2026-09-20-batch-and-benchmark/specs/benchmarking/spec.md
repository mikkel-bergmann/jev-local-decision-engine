## ADDED Requirements

### Requirement: A benchmark reports ten prompts in both modes
id: ten-prompt-benchmark

The system SHALL provide `benchmark.py` at the repository root, which scores ten prompts one at a
time and again as one batch, for the head path, the tool path, and the constrained path. It SHALL
print its results to standard output and SHALL exit zero on success.

#### Scenario: The benchmark runs both modes for every path
- **WHEN** a user runs the benchmark
- **THEN** it reports a sequential total and a batched total for each of the three paths

#### Scenario: The benchmark uses varied prompts
- **WHEN** the benchmark's prompts are inspected
- **THEN** they differ in length and subject, rather than repeating one string ten times

### Requirement: The benchmark reports time and memory together
id: benchmark-measures

The system SHALL report, for each path and mode, the total elapsed time, the time per prompt, and
the throughput in prompts per second. It SHALL report the shortest, median, and longest single
prompt from the sequential run. It SHALL report resident memory before and after the run, and the
peak, so a reader sees whether repeated scoring grows it.

#### Scenario: Every measure appears
- **WHEN** the benchmark finishes
- **THEN** its output carries a total, a per-prompt time, a throughput, a shortest, median and
  longest prompt time, and resident memory before, after, and at its peak

#### Scenario: Model load is reported apart from scoring
- **WHEN** the benchmark reports its timings
- **THEN** the model load time appears as its own figure, excluded from every scoring total

#### Scenario: Timing excludes the first call
- **WHEN** the benchmark times any mode
- **THEN** it scores at least one prompt first and discards that result, so no figure carries
  kernel compilation
