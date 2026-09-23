import pytest


@pytest.fixture(autouse=True)
def no_real_api_calls(monkeypatch):
    """Tests never reach OpenRouter: the key is removed (tests that exercise the
    AI path set a dummy key and stub the transport)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
