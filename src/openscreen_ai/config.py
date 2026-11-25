"""Загрузка и проверка настроек приложения."""

from __future__ import annotations

import ipaddress
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Настройки подключения к локальному серверу Ollama."""

    ollama_base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5-coder:7b"
    timeout_seconds: float = 120.0
    think: bool | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> AppConfig:
        """Создаёт конфигурацию из переменных окружения и проверяет значения."""
        values = os.environ if environ is None else environ
        defaults = cls()
        base_url = values.get("OLLAMA_BASE_URL", defaults.ollama_base_url).strip().rstrip("/")
        model = values.get("OLLAMA_MODEL", defaults.model).strip()
        raw_timeout = values.get("OLLAMA_TIMEOUT_SECONDS", str(defaults.timeout_seconds)).strip()
        raw_think = values.get("OLLAMA_THINK", "").strip().lower()
        if raw_think not in {"", "true", "false"}:
            raise ValueError("OLLAMA_THINK должен быть true, false или пустым.")

        try:
            parsed_url = urlsplit(base_url)
            hostname = parsed_url.hostname
            _ = parsed_url.port  # Accessing port validates its format and range.
        except ValueError as error:
            raise ValueError("OLLAMA_BASE_URL содержит некорректный адрес или порт.") from error

        if parsed_url.scheme not in {"http", "https"} or not hostname:
            raise ValueError("OLLAMA_BASE_URL должен быть полноценным URL с http:// или https://.")
        if parsed_url.username is not None or parsed_url.password is not None:
            raise ValueError("Не указывайте логин или пароль внутри OLLAMA_BASE_URL.")
        if parsed_url.path or "?" in base_url or "#" in base_url:
            raise ValueError("OLLAMA_BASE_URL должен содержать только схему, хост и порт.")
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = hostname.lower() == "localhost"
        if not is_loopback:
            raise ValueError("OpenScreen AI MVP разрешает подключение только к локальной Ollama.")
        if not model:
            raise ValueError("OLLAMA_MODEL не может быть пустым.")

        try:
            timeout_seconds = float(raw_timeout)
        except ValueError as error:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS должен быть числом больше нуля.") from error
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS должен быть числом больше нуля.")

        return cls(
            ollama_base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            think={"": None, "true": True, "false": False}[raw_think],
        )


def load_config() -> AppConfig:
    """Загружает локальный .env, если он есть, затем читает настройки."""
    load_dotenv()
    return AppConfig.from_env()
