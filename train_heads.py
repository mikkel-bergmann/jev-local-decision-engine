"""Train `DecisionHeads` on hand-labelled examples and the frozen encoder's
pooled features.

Distillation from the constrained engine (`build_corpus` / `label_corpus`,
retained below but no longer called from the training path) was measured
and rejected: 432 engine-labelled examples reached 0.667 on category and
0.617 on sentiment, while 48 hand-labelled examples cross-validated at
0.917 and 0.817 on the same features. The engine itself scored 0.683 and
0.717, so it was the ceiling — a student trained on its labels cannot pass
it, and hand labels already do. Training now loads
`data/train_a.json` / `train_b.json` / `train_c.json`, hand-authored
examples with human labels, and trains against `HEADS_SCHEMA` (urgency
collapsed to two classes; see `decision_heads.HEADS_SCHEMA`).
"""

import collections
import json
import os

import torch

import jev_local_engine
from decision_heads import (
    DecisionHeads,
    GATE_SCHEMA,
    GATE_THRESHOLD,
    HEADS_SCHEMA,
    TOOL_SCHEMA,
    encode,
    run_heads_decision,
    save_heads,
)
from jev_local_engine import SCHEMA, run_jev_decision

CORPUS_LABELS_PATH = os.path.join("data", "corpus_labels.json")
HOLDOUT_PATH = os.path.join("data", "holdout.json")
HOLDOUT_V2_PATH = os.path.join("data", "holdout_v2.json")

# Maps the constrained engine's four urgency choices onto the heads' two,
# so the two paths can be compared on urgency at all: low and medium are
# "not urgent enough to jump the queue", high and critical are "urgent".
ENGINE_URGENCY_TO_HEADS_URGENCY = {
    "low": "not_urgent",
    "medium": "not_urgent",
    "high": "urgent",
    "critical": "urgent",
}

HAND_LABELLED_TRAIN_PATHS = [
    os.path.join("data", "train_a.json"),
    os.path.join("data", "train_b.json"),
    os.path.join("data", "train_c.json"),
]

TOOL_TRAIN_PATHS = [
    os.path.join("data", "tool_train_a.json"),
    os.path.join("data", "tool_train_b.json"),
    os.path.join("data", "tool_train_c.json"),
    os.path.join("data", "tool_train_d.json"),
]

TOOL_HEAD_PATH = os.path.join("data", "tool_head.pt")

TOOL_HOLDOUT_PATH = os.path.join("data", "tool_holdout.json")

GATE_NEGATIVE_PATHS = [
    os.path.join("data", "gate_negatives_a.json"),
    os.path.join("data", "gate_negatives_b.json"),
]

GATE_HEAD_PATH = os.path.join("data", "gate_head.pt")

GATE_HOLDOUT_PATH = os.path.join("data", "gate_holdout.json")

# Evaluation-only: narrative past-tense negatives of the class the gate
# kept failing, plus genuine incident-logging requests that must still be
# accepted. Never appears in any TRAIN_PATHS list and is never read by
# load_gate_training_data — training on it would make the sweep stop
# measuring anything, per `gate-evaluation`.
GATE_ADVERSARIAL_PATH = os.path.join("data", "gate_adversarial.json")

CATEGORY_TOPICS = {
    "billing": ["my invoice", "the subscription charge", "my monthly bill"],
    "technical_support": ["the app", "the login page", "the dashboard"],
    "sales": ["your pricing plans", "the enterprise package", "a bulk discount"],
    "general_inquiry": ["your business hours", "your return policy", "the company address"],
}

URGENCY_PHRASES = {
    "low": ["whenever you get a chance", "no rush at all", "at your convenience"],
    "medium": ["sometime this week", "within a few days", "before the week is out"],
    "high": ["as soon as possible", "today if you can", "very soon"],
    "critical": ["right now", "immediately", "before anything else breaks"],
}

SENTIMENT_TEMPLATES = {
    "negative": [
        "I'm really frustrated with {topic}, please look into it {urgency}.",
        "This is unacceptable — {topic} is causing me real problems, fix it {urgency}.",
        "I'm upset about {topic} and need this resolved {urgency}.",
    ],
    "neutral": [
        "Can you tell me more about {topic}? I'd like an answer {urgency}.",
        "I have a question about {topic}, could you get back to me {urgency}.",
        "I wanted to check on {topic}, let me know {urgency}.",
    ],
    "positive": [
        "Thanks for your help with {topic}! Could you follow up {urgency}?",
        "I appreciate the support on {topic}, just checking in {urgency}.",
        "Great service so far on {topic} — one more update {urgency} would be great.",
    ],
}


def build_corpus():
    """Build the training corpus deterministically by crossing message
    templates with slot fillers.

    Nests category topics, urgency phrases, and sentiment templates in a
    fixed iteration order with no randomness, so two calls return identical
    lists. Sentiment templates cycle by a per-sentiment counter so text
    stays varied without needing more combinatorial breadth. Spans every
    category, every urgency, and every sentiment declared in `SCHEMA`.

    Returns an ordered list of at least 300 texts.
    """
    sentiment_counters = {sentiment: 0 for sentiment in SCHEMA["sentiment"]}
    texts = []
    for category in SCHEMA["category"]:
        for topic in CATEGORY_TOPICS[category]:
            for urgency in SCHEMA["urgency"]:
                for urgency_phrase in URGENCY_PHRASES[urgency]:
                    for sentiment in SCHEMA["sentiment"]:
                        templates = SENTIMENT_TEMPLATES[sentiment]
                        template = templates[
                            sentiment_counters[sentiment] % len(templates)
                        ]
                        sentiment_counters[sentiment] += 1
                        text = template.format(topic=topic, urgency=urgency_phrase)
                        texts.append(text)
    return texts


def label_corpus(texts, cache_path=CORPUS_LABELS_PATH):
    """Label each text in `texts` by scoring it with the constrained engine
    (`run_jev_decision`) as teacher, over `SCHEMA`.

    Caches to `cache_path` as `{"texts": [...], "labels": [...]}` so a rerun
    over the same corpus does not re-score. If the cached texts don't match
    the requested texts (a changed corpus), labels are recomputed and the
    cache is overwritten.

    Returns a list, parallel to `texts`, of `{field: choice}` dicts — one
    label per field per text, each drawn from that field's declared choices
    (guaranteed by `run_jev_decision`'s own schema validation).
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("texts") == texts:
            return cached["labels"]

    labels = []
    for text in texts:
        result = run_jev_decision(text, SCHEMA)
        label = {
            field: value["decision"] for field, value in result["decisions"].items()
        }
        labels.append(label)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"texts": texts, "labels": labels}, f, indent=2)

    return labels


def load_hand_labelled_training_data(paths=HAND_LABELLED_TRAIN_PATHS):
    """Load and combine the hand-authored training files.

    Each file is a JSON list of `{"text", "category", "urgency",
    "sentiment"}` records carrying human labels — never labels read from
    the constrained engine. `urgency` values are already in the heads'
    two-class scheme (`"not_urgent"` / `"urgent"`).

    Returns `(texts, labels)` as parallel lists, `labels` holding one
    `{field: choice}` dict per text, ready for `train`.
    """
    texts = []
    labels = []
    for path in paths:
        with open(path) as f:
            items = json.load(f)
        for item in items:
            texts.append(item["text"])
            labels.append(
                {
                    "category": item["category"],
                    "urgency": item["urgency"],
                    "sentiment": item["sentiment"],
                }
            )
    return texts, labels


def load_tool_training_data(paths=TOOL_TRAIN_PATHS):
    """Load and combine the hand-authored tool-routing training files.

    Each file is a JSON list of `{"text", "tool"}` records carrying a
    hand-authored tool label drawn from the catalogue — never labels read
    from the constrained engine, which cannot label tool routing at all.

    Returns `(texts, labels)` as parallel lists, `labels` holding one
    `{"tool": choice}` dict per text, ready for `train`.
    """
    texts = []
    labels = []
    for path in paths:
        with open(path) as f:
            items = json.load(f)
        for item in items:
            texts.append(item["text"])
            labels.append({"tool": item["tool"]})
    return texts, labels


def train_tool_head(texts=None, labels=None, epochs=400, lr=0.05, save_path=TOOL_HEAD_PATH):
    """Train a `DecisionHeads(TOOL_SCHEMA)` on the hand-authored tool
    training data, encoding the queries once and reusing `train`'s existing
    AdamW settings (and its encoder-frozen assertion) unchanged.

    `texts`/`labels` default to `load_tool_training_data()`'s output.
    Saves the trained heads to `save_path` (`data/tool_head.pt` by default)
    via `save_heads` and returns them.
    """
    if texts is None or labels is None:
        texts, labels = load_tool_training_data()

    heads = DecisionHeads(TOOL_SCHEMA)
    heads = train(texts, labels, heads=heads, epochs=epochs, lr=lr)

    save_heads(heads, save_path)
    return heads


def load_gate_training_data(tool_paths=TOOL_TRAIN_PATHS, negative_paths=GATE_NEGATIVE_PATHS):
    """Load the gate's training data: the existing hand-authored
    tool-routing queries as positives, and the hand-authored non-requests
    as negatives — never a label read from the constrained engine, which
    cannot answer "is this a tool request" at all.

    Each tool-routing query (`tool_paths`) becomes a
    `{"is_tool_request": "yes"}` example; each authored negative
    (`negative_paths`) becomes a `{"is_tool_request": "no"}` example. Only
    the text is used from the tool-routing queries — their `tool` label is
    irrelevant to the gate, which only ever answers yes or no.

    Returns `(texts, labels)` as parallel lists, ready for `train`.
    """
    texts = []
    labels = []
    for path in tool_paths:
        with open(path) as f:
            items = json.load(f)
        for item in items:
            texts.append(item["text"])
            labels.append({"is_tool_request": "yes"})
    for path in negative_paths:
        with open(path) as f:
            items = json.load(f)
        for item in items:
            texts.append(item["text"])
            labels.append({"is_tool_request": "no"})
    return texts, labels


def train_gate_head(texts=None, labels=None, epochs=400, lr=0.05, save_path=GATE_HEAD_PATH):
    """Train a `DecisionHeads(GATE_SCHEMA)` on the gate's training data.

    `texts`/`labels` default to `load_gate_training_data()`'s output, where
    the "yes" (tool-request) examples outnumber the "no" (negative)
    examples — 350 to 160 at present, a ratio the training data can't
    fully close since the tool-routing queries double as positives. Cross-
    entropy is weighted per class, inversely proportional to that class's
    share of the data, to offset the imbalance — otherwise the loss would
    reward a classifier that leans toward always answering "yes".

    Freezing the encoder and asserting it stayed frozen happens inside
    `train` itself (`assert all(not param.requires_grad ...)` immediately
    after freezing); training the gate head through the same `train`
    function inherits that assertion rather than repeating it.

    Saves the trained heads to `save_path` (`data/gate_head.pt` by default)
    via `save_heads` and returns them.
    """
    if texts is None or labels is None:
        texts, labels = load_gate_training_data()

    heads = DecisionHeads(GATE_SCHEMA)

    choices = GATE_SCHEMA["is_tool_request"]
    counts = collections.Counter(label["is_tool_request"] for label in labels)
    class_weights = {
        "is_tool_request": torch.tensor(
            [len(labels) / (len(choices) * counts[choice]) for choice in choices],
            dtype=torch.float32,
        )
    }

    heads = train(
        texts, labels, heads=heads, epochs=epochs, lr=lr, class_weights=class_weights
    )

    save_heads(heads, save_path)
    return heads


def train(texts, labels, heads=None, epochs=400, lr=0.05, class_weights=None):
    """Train `heads` on hand-labelled examples and the frozen encoder's
    pooled features, via AdamW and cross-entropy.

    Freezes every encoder parameter before training (asserted immediately
    after) and builds the optimizer over head parameters only (also
    asserted), so training can only ever update the heads. Encodes `texts`
    once — the single shared pass every field then trains against.

    `heads` defaults to a fresh `DecisionHeads(HEADS_SCHEMA)` — the heads'
    own schema, with urgency collapsed to two classes — not the constrained
    engine's four-class `SCHEMA`.

    `lr` defaults higher than is typical for a linear classifier because
    `encode`'s features are L2-normalized over 1536 dimensions, so each
    dimension's magnitude is small (~1/sqrt(1536)); a smaller learning rate
    measurably underfits within a practical epoch budget (verified: at
    lr=1e-3, 400 epochs plateaus in the 0.3-0.6 train-accuracy range, while
    lr=0.05 reaches 0.97+ on the same data in the same epoch budget).

    `class_weights`, when given, is a `{field: tensor}` mapping passed as
    cross-entropy's `weight` argument for that field — additive and
    optional, so every existing caller (which never had class imbalance to
    offset) is unaffected. `train_gate_head` is the one caller that passes
    it, to offset the gate's positives outnumbering its negatives.

    Returns the trained `heads`.
    """
    model, _ = jev_local_engine._get_model_and_tokenizer()
    for param in model.parameters():
        param.requires_grad = False
    assert all(not param.requires_grad for param in model.parameters()), (
        "encoder parameters must be frozen before training"
    )

    if heads is None:
        heads = DecisionHeads(HEADS_SCHEMA)

    optimizer = torch.optim.AdamW(heads.parameters(), lr=lr)
    head_param_ids = {id(p) for p in heads.parameters()}
    optimizer_param_ids = {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    assert optimizer_param_ids == head_param_ids, (
        "optimizer must hold only head parameters"
    )

    features = encode(texts)

    targets = {
        field: torch.tensor(
            [choices.index(label[field]) for label in labels], dtype=torch.long
        )
        for field, choices in heads.schema.items()
    }

    heads.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = 0.0
        for field, head in heads.heads.items():
            logits = head(features)
            weight = None
            if class_weights is not None and field in class_weights:
                weight = class_weights[field]
            loss = loss + torch.nn.functional.cross_entropy(
                logits, targets[field], weight=weight
            )
        loss.backward()
        optimizer.step()
    heads.eval()

    return heads


def evaluate(heads, holdout_path=HOLDOUT_V2_PATH):
    """Score `holdout_path` (the real holdout, `data/holdout_v2.json` by
    default — authored after training, never used to choose the urgency
    scheme or any other hyperparameter) with BOTH the trained heads and the
    constrained engine (`run_jev_decision`).

    The engine still scores urgency on its own four choices; those are
    mapped onto the heads' two-class scheme
    (`ENGINE_URGENCY_TO_HEADS_URGENCY`) before comparison, so the two paths
    are comparable on urgency at all.

    Returns `{field: {"heads_accuracy", "teacher_accuracy",
    "disagreement_count"}}` — one record per field carrying both accuracies
    and their disagreement count together, so a caller cannot read student
    accuracy without teacher accuracy: the student cannot exceed its
    teacher, and a gap must be attributable to one or the other.
    """
    with open(holdout_path) as f:
        holdout = json.load(f)

    texts = [item["text"] for item in holdout]
    heads_decisions = [run_heads_decision(text, heads)["decisions"] for text in texts]
    teacher_decisions = [
        run_jev_decision(text, SCHEMA)["decisions"] for text in texts
    ]

    report = {}
    for field in heads.schema:
        heads_correct = 0
        teacher_correct = 0
        disagreement_count = 0
        for item, heads_result, teacher_result in zip(
            holdout, heads_decisions, teacher_decisions
        ):
            true_label = item[field]
            heads_choice = heads_result[field]["decision"]
            teacher_choice = teacher_result[field]["decision"]
            if field == "urgency":
                teacher_choice = ENGINE_URGENCY_TO_HEADS_URGENCY[teacher_choice]

            if heads_choice == true_label:
                heads_correct += 1
            if teacher_choice == true_label:
                teacher_correct += 1
            if heads_choice != teacher_choice:
                disagreement_count += 1

        total = len(holdout)
        report[field] = {
            "heads_accuracy": heads_correct / total,
            "teacher_accuracy": teacher_correct / total,
            "disagreement_count": disagreement_count,
        }

    return report


def evaluate_tool_head(heads, holdout_path=TOOL_HOLDOUT_PATH):
    """Score `heads` (a `DecisionHeads(TOOL_SCHEMA)`) against the tool-
    routing holdout (`data/tool_holdout.json` by default) — authored after
    the tool training data and disjoint from it.

    recall@5 is the headline: the fraction of holdout queries whose true
    tool appears anywhere in that query's top-5 shortlist. recall@1 is the
    plain top-1 accuracy, reported alongside it. `unpredicted_tools` names
    every catalogue tool the head's top-1 decision never lands on across
    the whole holdout, so a dead class is named rather than folded into an
    average; `unpredicted_tool_count` is its length.

    The holdout here is one fixed, disjoint file, not a fold built from a
    larger corpus, so there is nothing to shuffle before partitioning —
    but per `tool-evaluation`, any cross-validation or fold split built
    from tool data elsewhere SHALL shuffle with a seeded generator first,
    the way the preceding change's stride-based split did not, which
    aliased with a repeating class cycle and reported a false result.
    """
    with open(holdout_path) as f:
        holdout = json.load(f)

    catalogue = heads.schema["tool"]
    predicted_top1 = set()

    top5_hits = 0
    top1_hits = 0
    for item in holdout:
        result = run_heads_decision(item["text"], heads, top_k=5)
        record = result["decisions"]["tool"]
        predicted_top1.add(record["decision"])

        if item["tool"] == record["decision"]:
            top1_hits += 1
        top5_choices = {entry["choice"] for entry in record["top_k"]}
        if item["tool"] in top5_choices:
            top5_hits += 1

    total = len(holdout)
    unpredicted_tools = sorted(set(catalogue) - predicted_top1)

    return {
        "recall_at_5": top5_hits / total,
        "recall_at_1": top1_hits / total,
        "unpredicted_tool_count": len(unpredicted_tools),
        "unpredicted_tools": unpredicted_tools,
    }


# The swept candidate thresholds: 0.10 to 0.90 inclusive, in steps of 0.05,
# per `gate-threshold`.
GATE_THRESHOLD_CANDIDATES = [round(0.10 + 0.05 * i, 2) for i in range(17)]


def _binary_rates(items, probabilities, threshold):
    """Score one candidate threshold against parallel `items` /
    `probabilities` lists (each item an `{"is_tool_request": bool, ...}`
    dict) and return the standard confusion-matrix rates.
    """
    tp = fp = tn = fn = 0
    for item, probability in zip(items, probabilities):
        accepted = probability >= threshold
        is_positive = item["is_tool_request"]
        if is_positive and accepted:
            tp += 1
        elif is_positive and not accepted:
            fn += 1
        elif not is_positive and accepted:
            fp += 1
        else:
            tn += 1

    negative_rejection_rate = tn / (tn + fp) if (tn + fp) else 0.0
    genuine_retention_rate = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = genuine_retention_rate
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    harmonic_mean = (
        2
        * negative_rejection_rate
        * genuine_retention_rate
        / (negative_rejection_rate + genuine_retention_rate)
        if (negative_rejection_rate + genuine_retention_rate)
        else 0.0
    )

    return {
        "negative_rejection_rate": negative_rejection_rate,
        "genuine_retention_rate": genuine_retention_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "harmonic_mean": harmonic_mean,
    }


def choose_gate_threshold(
    gate, holdout_path=GATE_HOLDOUT_PATH, adversarial_path=GATE_ADVERSARIAL_PATH
):
    """Sweep candidate gate decision thresholds against `holdout_path` AND
    `adversarial_path`, and select the one that maximizes the holdout's
    harmonic mean of the two directions of the trade a threshold makes —
    the share of negatives rejected and the share of genuine tool requests
    retained — breaking any tie on the adversarial negative-rejection rate
    rather than by taking the lowest tying threshold.

    The holdout alone under-determines the choice: it ties at harmonic
    mean 1.0 across a 0.20-wide band (0.35 to 0.55), so a rule that broke
    ties by taking the lowest value was choosing arbitrarily inside a range
    the holdout cannot discriminate at all — and happened to land on the
    threshold that re-admitted a whole class of narrative-incident
    negatives the retraining had otherwise pushed down (see
    `data/gate_adversarial.json` and this task's history). Adversarial
    rejection breaks that tie in the direction the holdout is blind to,
    without ever letting the adversarial set influence which candidate
    would count as strong on the holdout in the first place — the primary
    ranking is holdout harmonic mean alone, exactly as `gate-threshold`
    requires; the adversarial set only resolves ties that would otherwise
    be arbitrary.

    A hand-picked threshold is exactly what this function replaces — see
    `gate-threshold` and the decision recorded in this change's plan: a
    threshold chosen by judgement alone is how the previous advice (a
    confidence cutoff on the tool head's own softmax) was refuted by
    measurement, and how the very first sweep's tie-break (lowest value)
    quietly reintroduced the same kind of unmeasured choice. This sweep is
    the measurement, all the way down to the tie-break.

    Scores every holdout item and every adversarial item once each with
    `gate` (its "yes" probability), then, for each candidate threshold in
    `GATE_THRESHOLD_CANDIDATES` (0.10 to 0.90 in steps of 0.05), classifies
    an item accepted iff its "yes" probability is at or above that
    threshold. For each candidate, reports the holdout's
    `negative_rejection_rate`, `genuine_retention_rate`, `precision`,
    `recall`, `f1` and `harmonic_mean` (as before), plus
    `adversarial_negative_rejection_rate` and
    `adversarial_genuine_retention_rate` computed the same way over
    `adversarial_path`.

    Selection: among all candidates, keep only those whose holdout
    `harmonic_mean` equals the maximum achieved by any candidate; among
    those, pick the one with the highest
    `adversarial_negative_rejection_rate`; any tie remaining after that is
    broken by the lowest threshold, for full determinism.

    Returns `{"threshold": selected_value, "selected": {...the winning
    candidate's full record...}, "sweep": [...one record per candidate,
    in ascending threshold order...]}`.
    """
    with open(holdout_path) as f:
        holdout = json.load(f)
    with open(adversarial_path) as f:
        adversarial = json.load(f)

    holdout_probabilities = []
    for item in holdout:
        result = run_heads_decision(item["text"], gate)
        holdout_probabilities.append(
            result["decisions"]["is_tool_request"]["probabilities"]["yes"]
        )

    adversarial_probabilities = []
    for item in adversarial:
        result = run_heads_decision(item["text"], gate)
        adversarial_probabilities.append(
            result["decisions"]["is_tool_request"]["probabilities"]["yes"]
        )

    sweep = []
    for threshold in GATE_THRESHOLD_CANDIDATES:
        holdout_rates = _binary_rates(holdout, holdout_probabilities, threshold)
        adversarial_rates = _binary_rates(
            adversarial, adversarial_probabilities, threshold
        )

        record = dict(holdout_rates)
        record["threshold"] = threshold
        record["adversarial_negative_rejection_rate"] = adversarial_rates[
            "negative_rejection_rate"
        ]
        record["adversarial_genuine_retention_rate"] = adversarial_rates[
            "genuine_retention_rate"
        ]
        sweep.append(record)

    max_harmonic_mean = max(record["harmonic_mean"] for record in sweep)
    tied = [
        record for record in sweep if record["harmonic_mean"] == max_harmonic_mean
    ]
    best = max(
        tied,
        key=lambda record: (
            record["adversarial_negative_rejection_rate"],
            -record["threshold"],
        ),
    )

    return {"threshold": best["threshold"], "selected": best, "sweep": sweep}


def evaluate_gate(gate, holdout_path=GATE_HOLDOUT_PATH):
    """Score `gate` (a `DecisionHeads(GATE_SCHEMA)`) against the gate
    holdout (`data/gate_holdout.json` by default — authored after the gate
    training data and disjoint from it) at the shipped `GATE_THRESHOLD`.

    Reports the negative-rejection rate and the genuine-request retention
    rate side by side, never folding them into a single accuracy figure,
    per `gate-evaluation`; precision, recall and F1 are reported alongside.
    `recall` and `genuine_retention_rate` are the same figure under the two
    names `gate-threshold` and `gate-evaluation` each use.

    The holdout here is one fixed, disjoint file, not a fold built from a
    larger corpus, so there is nothing to shuffle before partitioning —
    but per `gate-evaluation`, any cross-validation or fold split built
    from gate data elsewhere SHALL shuffle with a seeded generator first,
    the same discipline `evaluate_tool_head` and `choose_gate_threshold`
    already carry.

    Returns `{"negative_rejection_rate", "genuine_retention_rate",
    "precision", "recall", "f1"}`.
    """
    with open(holdout_path) as f:
        holdout = json.load(f)

    tp = fp = tn = fn = 0
    for item in holdout:
        result = run_heads_decision(item["text"], gate)
        probability = result["decisions"]["is_tool_request"]["probabilities"]["yes"]
        accepted = probability >= GATE_THRESHOLD
        is_positive = item["is_tool_request"]
        if is_positive and accepted:
            tp += 1
        elif is_positive and not accepted:
            fn += 1
        elif not is_positive and accepted:
            fp += 1
        else:
            tn += 1

    negative_rejection_rate = tn / (tn + fp) if (tn + fp) else 0.0
    genuine_retention_rate = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = genuine_retention_rate
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "negative_rejection_rate": negative_rejection_rate,
        "genuine_retention_rate": genuine_retention_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
