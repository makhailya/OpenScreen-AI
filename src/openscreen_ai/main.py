"""Точка входа приложения OpenScreen AI."""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from openscreen_ai.config import AppConfig, load_config
from openscreen_ai.core.assistant import Assistant
from openscreen_ai.providers.ollama_provider import OllamaProvider
from openscreen_ai.ui.main_window import MainWindow


def create_window(config: AppConfig) -> MainWindow:
    """Собирает зависимости приложения и возвращает главное окно."""
    provider = OllamaProvider(config)
    assistant = Assistant(provider)
    return MainWindow(assistant)


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
