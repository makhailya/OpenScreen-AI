"""Проверки необязательной конфигурации и диагностики STT."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

import openscreen_ai.stt.availability as availability
from openscreen_ai.stt import SttConfig, check_stt_available
from openscreen_ai.stt.errors import (
    SttCompatibilityError,
    SttDependencyError,
    SttFilesError,
)


def _create_valid_assets(tmp_path: Path) -> SttConfig:
    model_dir = tmp_path / "parakeet"
    model_dir.mkdir()
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (model_dir / name).touch()
    vad_model = tmp_path / "silero_vad.onnx"
    vad_model.touch()
    return SttConfig(model_dir=model_dir, vad_model_path=vad_model)


def _pretend_optional_modules_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        availability.importlib.util,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None),
    )


def test_stt_config_uses_poc_defaults_and_accepts_paths() -> None:
    defaults = SttConfig()

    assert defaults.model_dir is None
    assert defaults.vad_model_path is None
    assert defaults.audio_device is None
    assert defaults.sample_rate == 16_000
    assert defaults.num_threads == 4
    assert defaults.feature_dim == 80
    assert defaults.decoding_method == "greedy_search"
    assert defaults.provider == "cpu"
    assert defaults.vad_min_silence_duration == 0.25
    assert defaults.vad_min_speech_duration == 0.15
    assert defaults.vad_max_speech_duration == 15.0

    configured = SttConfig(
        model_dir="~/models/parakeet",
        vad_model_path="models/silero_vad.onnx",
        audio_device="Built-in Microphone",
        num_threads=2,
    )
    assert configured.model_dir == Path("~/models/parakeet").expanduser()
    assert configured.vad_model_path == Path("models/silero_vad.onnx")
    assert configured.audio_device == "Built-in Microphone"
    assert configured.num_threads == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_threads": 0}, "num_threads"),
        ({"sample_rate": 44_100}, "sample_rate=16000"),
        ({"vad_min_silence_duration": float("nan")}, "конечными числами"),
        ({"vad_max_speech_duration": 0.1}, "должен быть больше"),
        ({"audio_device": ""}, "не может быть пустым"),
    ],
)
def test_stt_config_rejects_invalid_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SttConfig(**overrides)


def test_missing_optional_dependencies_have_install_guidance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _create_valid_assets(tmp_path)
    monkeypatch.setattr(availability.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(SttDependencyError) as error:
        check_stt_available(config)

    assert error.value.missing_distributions == ("sherpa-onnx", "sounddevice", "numpy")
    assert "pip install -e '.[stt]'" in str(error.value)


def test_missing_parakeet_files_are_listed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pretend_optional_modules_exist(monkeypatch)
    config = SttConfig(
        model_dir=tmp_path / "missing-model",
        vad_model_path=tmp_path / "silero_vad.onnx",
    )

    with pytest.raises(SttFilesError) as error:
        check_stt_available(config)

    message = str(error.value)
    assert "encoder.int8.onnx" in message
    assert "decoder.int8.onnx" in message
    assert "joiner.int8.onnx" in message
    assert "tokens.txt" in message
    assert "silero_vad.onnx" in message
    assert len(error.value.missing_model_files) == 4


def test_missing_vad_is_reported_separately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pretend_optional_modules_exist(monkeypatch)
    config = _create_valid_assets(tmp_path)
    missing_vad = tmp_path / "not-installed" / "silero_vad.onnx"
    config = SttConfig(model_dir=config.model_dir, vad_model_path=missing_vad)

    with pytest.raises(SttFilesError) as error:
        check_stt_available(config)

    assert error.value.missing_model_files == ()
    assert error.value.missing_vad_model == missing_vad
    assert "Silero VAD" in str(error.value)


def test_availability_check_does_not_import_optional_modules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _create_valid_assets(tmp_path)
    _pretend_optional_modules_exist(monkeypatch)
    real_import = __import__

    def reject_stt_imports(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".", maxsplit=1)[0] in {"sherpa_onnx", "sounddevice", "numpy"}:
            raise AssertionError(f"optional dependency was imported: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_stt_imports)
    check_stt_available(config)


def test_unsupported_python_version_has_clear_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(availability.sys, "version_info", (3, 14, 0))
    with pytest.raises(SttCompatibilityError, match="Python 3.12–3.13"):
        check_stt_available(SttConfig())


def test_base_application_imports_and_runs_without_stt_dependencies(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).parents[1]
    script = """
import builtins
import asyncio
import sys

optional_modules = {"sherpa_onnx", "sounddevice", "numpy"}
original_import = builtins.__import__

def block_optional_imports(name, *args, **kwargs):
    if name.split(".", maxsplit=1)[0] in optional_modules:
        raise ModuleNotFoundError(f"blocked optional dependency: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = block_optional_imports

import openscreen_ai.main
assert "openscreen_ai.stt.worker" not in sys.modules
import openscreen_ai.stt
from openscreen_ai.stt import SttConfig
from openscreen_ai.stt.audio import MicrophoneSource
from openscreen_ai.stt.engine import SttEngine
from openscreen_ai.core.assistant import Assistant

from openscreen_ai.stt.worker import SttController

SttController(SttConfig())
SttEngine(SttConfig())
MicrophoneSource()

class Provider:
    async def stream(self, prompt):
        yield prompt

async def run_assistant():
    assistant = Assistant(Provider())
    return [chunk async for chunk in assistant.stream_response("test")]

assert asyncio.run(run_assistant()) == ["test"]
assert optional_modules.isdisjoint(sys.modules)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
