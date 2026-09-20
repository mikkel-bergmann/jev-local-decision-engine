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

import json
import os

import torch

import jev_local_engine
from decision_heads import DecisionHeads, HEADS_SCHEMA, encode, run_heads_decision
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


def train(texts, labels, heads=None, epochs=400, lr=0.05):
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
            loss = loss + torch.nn.functional.cross_entropy(logits, targets[field])
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
