"""Benchmark: ten prompts, scored one at a time and again as one batch, over
the head path, the tool-routing path, and the constrained (first-token)
path.

Reports, per path and mode, the total elapsed time, the time per prompt, and
throughput in prompts per second; the shortest, median, and longest single
prompt from the sequential run; and resident memory before, after, and at
its peak. Model load is timed and reported separately, excluded from every
scoring total, and the first call on every path is scored once and
discarded before any timed region, so no figure carries Metal's one-time
kernel-compilation cost.
"""

import statistics
import time

import torch

# Ten prompts differing in length and subject, drawn from the domains the
# schema and tool catalogue cover: customer-support billing, technical
# support, sales, general inquiry, positive feedback, and cross-cutting
# tool-request domains (source control, container orchestration, payments,
# compliance). Includes a short fragment ("Thanks!") and a long, multi-clause
# sentence, so padding cost between the shortest and longest row shows.
PROMPTS = [
    "Thanks!",
    "Double charged again, fix this!",
    "App keeps crashing on export.",
    "What are your pricing plans?",
    "Push the release branch, tag v2.3.0.",
    "Roll back the deployment, checks failing.",
    "Refund my last charge, please.",
    "Log a cooler temperature violation.",
    "Great work on my ticket, thanks!",
    "I've been a customer for five years and my payment keeps failing.",
]


def _synchronize():
    """Block until every queued MPS operation has actually finished.

    MPS submits work to an asynchronous queue: without this barrier,
    `time.perf_counter()` measures how fast work was *submitted*, not how
    long it took to *run*. Called on both sides of every timed region.
    """
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _timed(fn, *args, **kwargs):
    """Call `fn(*args, **kwargs)` inside an MPS synchronize barrier on both
    sides, so queued-but-not-yet-executed work is never mistaken for
    elapsed compute time. Returns `(result, elapsed_ms)`.
    """
    _synchronize()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    _synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def time_model_load():
    """Load the base Qwen model and tokenizer, timing the load on its own
    so it is never folded into any scoring total. Loading is one-time and
    cached (`jev_local_engine._get_model_and_tokenizer`), so this must be
    the first call made against the model in the process.
    """
    from jev_local_engine import _get_model_and_tokenizer

    _, load_ms = _timed(_get_model_and_tokenizer)
    return load_ms


def measure_path(score_one, score_batch, prompts):
    """Measure one scoring path (head, tool, or constrained) over `prompts`,
    both one prompt at a time and as a single batch.

    Each mode's kernels are compiled by Metal the first time that mode
    runs, at that mode's own batch shape (1 for sequential, len(prompts)
    for batched) — a cost real callers pay once but a benchmark must not
    attribute to steady-state scoring. So each mode is warmed with one
    scored-and-discarded call of its own before anything is timed.

    Returns a dict with the per-prompt sequential latencies (ms), the
    sequential total (ms), and the batched total (ms).
    """
    # Warm-up: one single-input call, discarded, compiling batch-size-1
    # kernels before the sequential run is timed.
    score_one(prompts[0])

    sequential_latencies_ms = []
    for prompt in prompts:
        _, elapsed_ms = _timed(score_one, prompt)
        sequential_latencies_ms.append(elapsed_ms)
    sequential_total_ms = sum(sequential_latencies_ms)

    # Warm-up: one batched call, discarded, compiling batch-size-N kernels
    # before the batched run is timed.
    score_batch(prompts)

    _, batched_total_ms = _timed(score_batch, prompts)

    return {
        "sequential_latencies_ms": sequential_latencies_ms,
        "sequential_total_ms": sequential_total_ms,
        "batched_total_ms": batched_total_ms,
    }


def build_paths():
    """Load every checkpoint the three paths need and return
    {name: (score_one, score_batch)} for each: head, tool, and constrained.
    """
    from decision_heads import (
        GATE_SCHEMA,
        GATE_THRESHOLD,
        HEADS_SCHEMA,
        TOOL_SCHEMA,
        load_heads,
        route_tool,
        route_tool_batch,
        run_heads_decision,
        run_heads_decision_batch,
    )
    from jev_local_engine import SCHEMA, run_jev_decision, run_jev_decision_batch

    heads = load_heads("data/heads.pt", HEADS_SCHEMA)
    gate = load_heads("data/gate_head.pt", GATE_SCHEMA)
    tool_heads = load_heads("data/tool_head.pt", TOOL_SCHEMA)

    return {
        "head": (
            lambda text: run_heads_decision(text, heads),
            lambda texts: run_heads_decision_batch(texts, heads),
        ),
        "tool": (
            lambda text: route_tool(text, tool_heads, gate, GATE_THRESHOLD),
            lambda texts: route_tool_batch(texts, tool_heads, gate, GATE_THRESHOLD),
        ),
        "constrained": (
            lambda text: run_jev_decision(text, SCHEMA),
            lambda texts: run_jev_decision_batch(texts, SCHEMA),
        ),
    }


def run_full_workload(paths, prompts):
    """Score every prompt through every path, once sequentially and once
    batched — the same shapes `measure_path` already warmed for each path
    (batch-size-1 for the sequential loop, batch-size-len(prompts) for the
    batched call) — with no timing.

    Used to repeat the whole benchmark's workload after the measured run,
    so the benchmark can report resident memory after each repeat and the
    growth between repeats: a single before-and-after pair cannot show
    whether repeated scoring keeps growing memory, only whether it grew
    once. Because every shape here was already exercised by `measure_path`,
    a repeat's growth reflects steady-state scoring, not first-time
    allocation.
    """
    for score_one, score_batch in paths.values():
        for prompt in prompts:
            score_one(prompt)
        score_batch(prompts)


def format_path_report(path_name, measurement, num_prompts):
    """Render one path's sequential and batched figures: total, per-prompt
    time, and throughput for both modes; shortest, median, and longest
    prompt from the sequential run.
    """
    seq_latencies_ms = measurement["sequential_latencies_ms"]
    seq_total_ms = measurement["sequential_total_ms"]
    batched_total_ms = measurement["batched_total_ms"]

    lines = [f"\n=== {path_name} path ==="]
    lines.append(
        f"  sequential: total {seq_total_ms:.1f} ms, "
        f"{seq_total_ms / num_prompts:.1f} ms/prompt, "
        f"{num_prompts / (seq_total_ms / 1000):.2f} prompts/sec"
    )
    lines.append(
        f"              shortest {min(seq_latencies_ms):.1f} ms, "
        f"median {statistics.median(seq_latencies_ms):.1f} ms, "
        f"longest {max(seq_latencies_ms):.1f} ms"
    )
    lines.append(
        f"  batched:    total {batched_total_ms:.1f} ms, "
        f"{batched_total_ms / num_prompts:.1f} ms/prompt, "
        f"{num_prompts / (batched_total_ms / 1000):.2f} prompts/sec"
    )
    lines.append(f"  batched is {seq_total_ms / batched_total_ms:.2f}x the sequential speed")
    return "\n".join(lines)


def main():
    from jev_local_engine import _current_rss_gb, _peak_rss_gb

    print(f"Benchmarking {len(PROMPTS)} prompts over the head, tool, and constrained paths.\n")

    load_ms = time_model_load()
    print(f"Model load: {load_ms:.1f} ms (one-time; excluded from every scoring total)")

    mem_before_gb = _current_rss_gb()

    paths = build_paths()
    for name, (score_one, score_batch) in paths.items():
        measurement = measure_path(score_one, score_batch, PROMPTS)
        print(format_path_report(name, measurement, len(PROMPTS)))

    mem_after_gb = _current_rss_gb()

    print("\n=== Resident memory ===")
    print(f"  before: {mem_before_gb:.3f} GB")
    print(f"  after:  {mem_after_gb:.3f} GB")

    # The measured run above already paid for the model load and for the
    # first allocation of every shape scored (sequential batch-of-1 and
    # batched batch-of-len(prompts), for each of the three paths) — a
    # one-time cost, not evidence about repeated scoring. A single
    # before-and-after pair can't tell those apart, so the whole workload
    # is run three more times here, with memory sampled after each repeat,
    # to show directly whether it keeps growing. Read these repeats, not
    # the "after" figure above, as the steady state.
    print(
        "\n=== Repeated workload (steady-state check) ===\n"
        "  the run above already paid for model load and first-shape "
        "allocation; treat these repeats as the steady state, not it"
    )
    repeat_mem_gb = mem_after_gb
    for repeat_index in range(1, 4):
        run_full_workload(paths, PROMPTS)
        mem_now_gb = _current_rss_gb()
        growth_mb = (mem_now_gb - repeat_mem_gb) * 1024
        print(f"  repeat {repeat_index}: {mem_now_gb:.3f} GB (growth {growth_mb:+.1f} MB)")
        repeat_mem_gb = mem_now_gb

    mem_peak_gb = _peak_rss_gb()
    print(f"\n  peak: {mem_peak_gb:.3f} GB")


if __name__ == "__main__":
    main()
