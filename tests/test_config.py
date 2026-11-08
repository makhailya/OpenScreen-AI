"""Проверки чтения и валидации конфигурации."""

import pytest

from openscreen_ai.config import AppConfig


def test_config_uses_documented_defaults() -> None:
    config = AppConfig.from_env({})

    assert config.ollama_base_url == "http://127.0.0.1:11434"
    assert config.model == "qwen2.5-coder:7b"
    assert config.timeout_seconds == 120.0


def test_config_reads_overrides_and_normalizes_url() -> None:
    config = AppConfig.from_env(
        {
            "OLLAMA_BASE_URL": "http://localhost:11434/",
            "OLLAMA_MODEL": "custom-model:latest",
            "OLLAMA_TIMEOUT_SECONDS": "42.5",
        }
    )

    assert config.ollama_base_url == "http://localhost:11434"
    assert config.model == "custom-model:latest"
    assert config.timeout_seconds == 42.5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"OLLAMA_BASE_URL": "localhost:11434"}, "OLLAMA_BASE_URL"),
        ({"OLLAMA_MODEL": "  "}, "OLLAMA_MODEL"),
        ({"OLLAMA_TIMEOUT_SECONDS": "fast"}, "OLLAMA_TIMEOUT_SECONDS"),
        ({"OLLAMA_TIMEOUT_SECONDS": "0"}, "OLLAMA_TIMEOUT_SECONDS"),
        ({"OLLAMA_TIMEOUT_SECONDS": "inf"}, "OLLAMA_TIMEOUT_SECONDS"),
        ({"OLLAMA_TIMEOUT_SECONDS": "nan"}, "OLLAMA_TIMEOUT_SECONDS"),
        ({"OLLAMA_BASE_URL": "http://user:secret@localhost:11434"}, "логин или пароль"),
        ({"OLLAMA_BASE_URL": "https://example.com:11434"}, "только к локальной Ollama"),
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AppConfig.from_env(overrides)
