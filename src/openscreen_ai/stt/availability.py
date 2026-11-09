"""Явная проверка готовности необязательного STT runtime."""

from __future__ import annotations

import importlib.util
import sys

from openscreen_ai.stt.config import SttConfig
from openscreen_ai.stt.errors import (
    SttCompatibilityError,
    SttConfigurationError,
    SttDependencyError,
    SttFilesError,
)

# distribution name -> import name. find_spec проверяет наличие пакета,
# но не импортирует native-модули Sherpa-ONNX.
_OPTIONAL_MODULES = (
    ("sherpa-onnx", "sherpa_onnx"),
    ("sounddevice", "sounddevice"),
    ("numpy", "numpy"),
)
_REQUIRED_MODEL_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)


def _module_is_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def check_stt_available(config: SttConfig) -> None:
    """Проверяет Python, optional-пакеты и заданные файлы, не импортируя STT.

    Вызывайте эту функцию только при явной подготовке STT. Обычный импорт
    приложения и пакета openscreen_ai.stt эту проверку не запускает.
    """
    python_version = sys.version_info[:2]
    if not ((3, 12) <= python_version < (3, 14)):
        version = ".".join(str(part) for part in sys.version_info[:3])
        raise SttCompatibilityError(
            f"Parakeet STT проверен на Python 3.12–3.13; сейчас используется {version}."
        )

    missing_distributions = [
        distribution
        for distribution, module in _OPTIONAL_MODULES
        if not _module_is_available(module)
    ]
    if missing_distributions:
        raise SttDependencyError(missing_distributions)

    if config.model_dir is None:
        raise SttConfigurationError(
            "Укажите каталог Parakeet через SttConfig.model_dir."
        )
    if config.vad_model_path is None:
        raise SttConfigurationError(
            "Укажите файл Silero VAD через SttConfig.vad_model_path."
        )

    missing_model_files = tuple(
        config.model_dir / filename
        for filename in _REQUIRED_MODEL_FILES
        if not (config.model_dir / filename).is_file()
    )
    missing_vad_model = (
        config.vad_model_path if not config.vad_model_path.is_file() else None
    )
    if missing_model_files or missing_vad_model is not None:
        raise SttFilesError(
            missing_model_files=missing_model_files,
            missing_vad_model=missing_vad_model,
        )
