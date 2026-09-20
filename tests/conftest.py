import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_model: needs network/cache access to Qwen/Qwen2.5-1.5B-Instruct "
        "(tokenizer or full model weights); skipped when unavailable",
    )


@pytest.fixture(scope="session")
def qwen_tokenizer():
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    except Exception as exc:  # network/cache unavailable
        pytest.skip(f"Qwen tokenizer unavailable: {exc}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


@pytest.fixture(scope="session")
def qwen_model_and_tokenizer():
    from jev_local_engine import load_model

    try:
        model, tokenizer = load_model()
    except Exception as exc:  # network/cache/MPS unavailable
        pytest.skip(f"Qwen model unavailable: {exc}")
    return model, tokenizer
