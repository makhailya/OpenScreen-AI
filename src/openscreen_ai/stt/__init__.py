"""Необязательная инфраструктура распознавания речи OpenScreen AI.

Импорт этого пакета не загружает Sherpa-ONNX и не открывает модель.
"""

from openscreen_ai.stt.availability import check_stt_available
from openscreen_ai.stt.config import SttConfig
from openscreen_ai.stt.engine import SttEngine, SttResult
from openscreen_ai.stt.errors import (
    SttAudioError,
    SttCompatibilityError,
    SttConfigurationError,
    SttDecodeError,
    SttDependencyError,
    SttError,
    SttFilesError,
    SttInitializationError,
    SttLifecycleError,
)

__all__ = [
    "SttAudioError",
    "SttCompatibilityError",
    "SttConfig",
    "SttConfigurationError",
    "SttDecodeError",
    "SttDependencyError",
    "SttEngine",
    "SttError",
    "SttFilesError",
    "SttInitializationError",
    "SttLifecycleError",
    "SttResult",
    "check_stt_available",
]
