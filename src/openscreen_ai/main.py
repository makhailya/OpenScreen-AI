"""Точка входа приложения OpenScreen AI."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication, QMessageBox

from openscreen_ai.config import AppConfig, load_config
from openscreen_ai.core.assistant import Assistant
from openscreen_ai.providers.factory import ProviderConfig, ProviderFactory
from openscreen_ai.providers.ollama_provider import OllamaProvider
from openscreen_ai.stt.availability import check_stt_available
from openscreen_ai.stt.config import SttConfig
from openscreen_ai.stt.errors import SttError
from openscreen_ai.ui.main_window import MainWindow

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openscreen_ai.stt.worker import SttController


def create_window(
    config: AppConfig,
    *,
    stt_config: SttConfig | None = None,
    stt_controller_factory: Callable[[SttConfig], SttController] | None = None,
) -> MainWindow:
    """Собирает зависимости приложения без загрузки STT-моделей."""
    from openscreen_ai.stt.worker import SttController

    provider = ProviderFactory(OllamaProvider).create(ProviderConfig(config.provider, config))
    assistant = Assistant(provider)
    resolved_stt_config = stt_config or SttConfig(
        model_dir=os.environ.get("OPENSCREEN_STT_MODEL_DIR") or None,
        vad_model_path=os.environ.get("OPENSCREEN_STT_VAD_MODEL_PATH") or None,
    )

    try:
        check_stt_available(resolved_stt_config)
    except (SttError, ValueError, OSError):
        # This metadata-only check avoids presenting a dictation action that is
        # guaranteed to fail. Do not include paths or exception text in logs/UI.
        logger.warning("STT_OPTIONAL_NOT_READY")
        window = MainWindow(assistant)
        window.statusBar().showMessage(
            "Диктовка недоступна. Проверьте окружение командой "
            "«python -m openscreen_ai doctor»."
        )
        return window

    controller_factory = stt_controller_factory or SttController
    stt_controller = controller_factory(resolved_stt_config)
    return MainWindow(assistant, stt_controller)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    application = QApplication(sys.argv)
    application.setApplicationName("OpenScreen AI")

    try:
        config = load_config()
    except ValueError as error:
        QMessageBox.critical(None, "Ошибка конфигурации", str(error))
        return 2

    window = create_window(config)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
