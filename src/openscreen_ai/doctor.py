"""Safe offline preflight checks for OpenScreen AI."""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, TextIO

_CONFIG_KEYS = frozenset(
    {
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_TIMEOUT_SECONDS",
        "OPENSCREEN_STT_MODEL_DIR",
        "OPENSCREEN_STT_VAD_MODEL_PATH",
    }
)
_MODEL_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
_CORE_MODULES = ("httpx", "dotenv")
_MAX_CONFIG_FILE_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    status: str
    category: str
    detail: str


def _check(status: str, category: str, detail: str) -> DoctorCheck:
    return DoctorCheck(status, category, detail)


def _dotenv_parsers() -> tuple[Any, Any] | None:
    """Load optional dotenv parsing helpers only when the dependency is present."""
    try:
        from dotenv.parser import parse_stream
        from dotenv.variables import parse_variables
    except ImportError:
        return None
    return parse_stream, parse_variables


def _read_config_file(
    path: Path, environ: Mapping[str, str]
) -> tuple[dict[str, str], DoctorCheck]:
    try:
        file_info = path.lstat()
    except FileNotFoundError:
        return {}, _check("PASS", "config_file", "not present; defaults and process settings apply")
    except (OSError, ValueError):
        return {}, _check("FAIL", "config_file", "present but unreadable or invalid")
    if not stat.S_ISREG(file_info.st_mode):
        return {}, _check("FAIL", "config_file", "not a regular file")
    if file_info.st_size > _MAX_CONFIG_FILE_BYTES:
        return {}, _check("FAIL", "config_file", "exceeds safe size limit")

    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened_info = os.fstat(descriptor)
        if not stat.S_ISREG(opened_info.st_mode):
            return {}, _check("FAIL", "config_file", "not a regular file")
        if (opened_info.st_dev, opened_info.st_ino) != (file_info.st_dev, file_info.st_ino):
            return {}, _check("FAIL", "config_file", "changed before reading")
        if opened_info.st_size > _MAX_CONFIG_FILE_BYTES:
            return {}, _check("FAIL", "config_file", "exceeds safe size limit")
        with os.fdopen(descriptor, "rb") as config_file:
            descriptor = None
            raw_contents = config_file.read(_MAX_CONFIG_FILE_BYTES + 1)
        if len(raw_contents) > _MAX_CONFIG_FILE_BYTES:
            return {}, _check("FAIL", "config_file", "exceeds safe size limit")
        contents = raw_contents.decode("utf-8")
    except (OSError, UnicodeError, ValueError):
        return {}, _check("FAIL", "config_file", "present but unreadable or invalid")
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

    dotenv_parsers = _dotenv_parsers()
    if dotenv_parsers is None:
        return {}, _check(
            "WARN", "config_file", "parser unavailable because a dependency is missing"
        )
    parse_stream, parse_variables = dotenv_parsers
    bindings = tuple(parse_stream(StringIO(contents)))
    if any(binding.error for binding in bindings):
        return {}, _check("FAIL", "config_file", "present but unreadable or invalid")
    resolved: dict[str, str | None] = {}
    for binding in bindings:
        if binding.key is None:
            continue
        if binding.value is None:
            resolved[binding.key] = None
            continue
        interpolation_context = {**resolved, **environ}
        atoms = parse_variables(binding.value)
        resolved[binding.key] = "".join(atom.resolve(interpolation_context) for atom in atoms)
    filtered = {
        key: value
        for key, value in resolved.items()
        if key in _CONFIG_KEYS and value is not None
    }
    return filtered, _check("PASS", "config_file", "readable")


def _is_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _path_is_readable(path: Path, *, directory: bool = False) -> bool:
    try:
        if directory:
            if not path.is_dir():
                return False
            with os.scandir(path):
                return True
        if not path.is_file():
            return False
        with path.open("rb"):
            return True
    except (OSError, RuntimeError, ValueError):
        return False


def collect_doctor_checks(
    *, environ: Mapping[str, str] | None = None, config_path: Path | None = None
) -> tuple[DoctorCheck, ...]:
    """Collect safe status categories without contacting services or opening devices."""
    process_values = os.environ if environ is None else environ
    config_file = config_path if config_path is not None else Path.cwd() / ".env"
    file_values, file_check = _read_config_file(config_file, process_values)
    values = {
        **file_values,
        **{key: value for key, value in process_values.items() if key in _CONFIG_KEYS},
    }

    checks = [
        _check(
            "PASS" if sys.version_info >= (3, 12) else "FAIL",
            "python",
            (
                f"Python {sys.version_info[0]}.{sys.version_info[1]}."
                f"{sys.version_info[2]} supported"
                if sys.version_info >= (3, 12)
                else f"Python {sys.version_info[0]}.{sys.version_info[1]}."
                f"{sys.version_info[2]} requires 3.12 or newer"
            ),
        ),
        file_check,
    ]

    missing_core = [module for module in _CORE_MODULES if not _is_available(module)]
    core_detail = "missing required modules" if missing_core else "available"
    checks.append(_check("FAIL" if missing_core else "PASS", "core_dependencies", core_detail))
    qt_available = _is_available("PySide6.QtWidgets")
    qt_detail = "bindings available" if qt_available else "PySide6 Qt widgets unavailable"
    checks.append(_check("PASS" if qt_available else "FAIL", "qt", qt_detail))

    try:
        from openscreen_ai.config import AppConfig
    except ImportError:
        checks.append(
            _check("WARN", "llm_config", "validator unavailable because a dependency is missing")
        )
    else:
        try:
            AppConfig.from_env(values)
        except (TypeError, ValueError):
            checks.append(_check("FAIL", "llm_config", "invalid local configuration"))
        else:
            checks.append(
                _check("PASS", "llm_config", "valid syntax; endpoint availability not checked")
            )

    model_raw = values.get("OPENSCREEN_STT_MODEL_DIR", "").strip()
    vad_raw = values.get("OPENSCREEN_STT_VAD_MODEL_PATH", "").strip()
    if not model_raw and not vad_raw:
        checks.append(_check("WARN", "stt", "optional speech recognition is not configured"))
        return tuple(checks)
    if not model_raw or not vad_raw:
        checks.append(
            _check("WARN", "stt", "optional speech recognition configuration is incomplete")
        )
        return tuple(checks)

    stt_python_supported = (3, 12) <= sys.version_info[:2] < (3, 14)
    checks.append(
        _check(
            "PASS" if stt_python_supported else "WARN",
            "stt_python",
            "supported runtime"
            if stt_python_supported
            else "optional STT runtime supports Python 3.12 and 3.13",
        )
    )

    try:
        model_dir = Path(model_raw).expanduser()
        vad_path = Path(vad_raw).expanduser()
    except (RuntimeError, ValueError):
        model_dir = Path()
        vad_path = Path()
        paths_readable = False
    else:
        paths_readable = _path_is_readable(model_dir, directory=True) and _path_is_readable(
            vad_path
        )
    model_files_readable = paths_readable and all(
        _path_is_readable(model_dir / name) for name in _MODEL_FILES
    )
    checks.append(
        _check(
            "PASS" if model_files_readable else "WARN",
            "stt_model_paths",
            "configured model files readable"
            if model_files_readable
            else "optional model files missing or unreadable",
        )
    )
    optional_available = all(
        _is_available(module) for module in ("sherpa_onnx", "sounddevice", "numpy")
    )
    dependency_detail = (
        "available"
        if optional_available
        else "optional runtime dependencies unavailable"
    )
    checks.append(
        _check("PASS" if optional_available else "WARN", "stt_dependencies", dependency_detail)
    )
    return tuple(checks)


def run_doctor(
    *,
    environ: Mapping[str, str] | None = None,
    config_path: Path | None = None,
    output: TextIO | None = None,
) -> int:
    """Print fixed safe statuses and return 0 for PASS/WARN or 1 for any FAIL."""
    stream = output if output is not None else sys.stdout
    checks = collect_doctor_checks(environ=environ, config_path=config_path)
    for item in checks:
        print(f"{item.status} {item.category}: {item.detail}", file=stream)
    return int(any(item.status == "FAIL" for item in checks))
