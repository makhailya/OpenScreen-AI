import pytest

from openscreen_ai.config import AppConfig, ProviderKind
from openscreen_ai.providers.factory import ProviderConfig, ProviderFactory
from openscreen_ai.providers.ollama_provider import OllamaProvider


def test_default_provider_and_real_factory():
    config = AppConfig.from_env({})
    assert config.provider is ProviderKind.OLLAMA
    provider = ProviderFactory().create(ProviderConfig(config.provider, config))
    assert isinstance(provider, OllamaProvider)


def test_legacy_environment_configuration_works_without_network(monkeypatch):
    import socket

    def deny_network(*args, **kwargs):
        pytest.fail("Configuration must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    config = AppConfig.from_env({"OLLAMA_MODEL": "local", "OLLAMA_TIMEOUT_SECONDS": "30"})
    provider = ProviderFactory().create(ProviderConfig(config.provider, config))
    assert config.model == "local"
    assert config.timeout_seconds == 30
    assert isinstance(provider, OllamaProvider)


def test_sensitive_configuration_values_are_not_in_repr_or_errors():
    import traceback

    sensitive = "sensitive-marker-for-test"
    config = AppConfig(ollama_base_url=sensitive, model=sensitive)
    assert sensitive not in repr(config)
    assert sensitive not in repr(ProviderConfig(ProviderKind.OLLAMA, config))
    with pytest.raises(ValueError) as error:
        AppConfig.from_env({"OPENSCREEN_PROVIDER": sensitive})
    assert sensitive not in str(error.value)
    for values in (
        {"OLLAMA_TIMEOUT_SECONDS": sensitive},
        {"OLLAMA_BASE_URL": f"http://localhost:{sensitive}"},
    ):
        with pytest.raises(ValueError) as error:
            AppConfig.from_env(values)
        assert sensitive not in "".join(traceback.format_exception(error.value))


def test_factory_passes_existing_config_to_ollama():
    config = AppConfig(model="local-model")
    marker = object()
    seen = []

    def create(value):
        seen.append(value)
        return marker

    assert ProviderFactory(create).create(ProviderConfig(ProviderKind.OLLAMA, config)) is marker
    assert seen == [config]


def test_cloud_fails_closed_without_fallback():
    def unexpected(_config):
        pytest.fail("Ollama must not be used as a fallback")

    with pytest.raises(ValueError, match="недоступен"):
        ProviderFactory(unexpected).create(ProviderConfig(ProviderKind.CHATGPT, AppConfig()))


@pytest.mark.parametrize("value", ["chatgpt", "unknown"])
def test_unavailable_provider_rejected_in_environment(value):
    with pytest.raises(ValueError):
        AppConfig.from_env({"OPENSCREEN_PROVIDER": value})
