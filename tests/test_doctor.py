"""Offline and privacy-safe tests for the doctor command."""

from __future__ import annotations

import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from openscreen_ai import __main__ as module_entrypoint
from openscreen_ai import doctor


def check_by_category(checks: tuple[doctor.DoctorCheck, ...], category: str) -> doctor.DoctorCheck:
    return next(item for item in checks if item.category == category)


def test_unconfigured_stt_is_optional_warning_and_config_file_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary = "SYNTHETIC_SECRET_CANARY_9172"
    config_file = tmp_path / ".env"
    config_file.write_text(f"OLLAMA_MODEL={canary}\n", encoding="utf-8")
    before = config_file.read_bytes()
    monkeypatch.setattr(doctor, "_is_available", lambda _name: True)

    checks = doctor.collect_doctor_checks(environ={}, config_path=config_file)
    output = StringIO()
    exit_code = doctor.run_doctor(environ={}, config_path=config_file, output=output)

    assert check_by_category(checks, "stt").status == "WARN"
    assert check_by_category(checks, "llm_config").status == "PASS"
    assert exit_code == 0
    assert canary not in output.getvalue()
    assert str(config_file) not in output.getvalue()
    assert config_file.read_bytes() == before


def test_doctor_does_not_change_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "_is_available", lambda _name: True)
    before = dict(os.environ)

    doctor.collect_doctor_checks(
        environ={"OLLAMA_MODEL": "synthetic-model"},
        config_path=tmp_path / "absent.env",
    )

    assert dict(os.environ) == before


def test_doctor_reports_python_major_minor_and_micro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "sys", SimpleNamespace(version_info=(3, 13, 7)))
    monkeypatch.setattr(doctor, "_is_available", lambda _name: True)

    checks = doctor.collect_doctor_checks(
        environ={},
        config_path=tmp_path / "absent.env",
    )

    python_check = check_by_category(checks, "python")
    assert python_check.status == "PASS"
    assert python_check.detail == "Python 3.13.7 supported"


def test_module_entrypoint_reports_missing_dependencies_under_python_s_without_traceback(
    tmp_path: Path,
) -> None:
    source_dir = Path(__file__).resolve().parents[1] / "src"
    clean_env = {
        "PYTHONPATH": str(source_dir),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    result = subprocess.run(
        [sys.executable, "-S", "-m", "openscreen_ai", "doctor"],
        cwd=tmp_path,
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 1
    assert "FAIL core_dependencies: missing required modules" in result.stdout
    assert "FAIL qt: PySide6 Qt widgets unavailable" in result.stdout
    assert "WARN llm_config: validator unavailable because a dependency is missing" in result.stdout
    assert "WARN stt: optional speech recognition is not configured" in result.stdout
    assert result.stderr == ""
    assert "Traceback" not in result.stdout + result.stderr
    assert "ModuleNotFoundError" not in result.stdout + result.stderr


def test_dotenv_interpolation_matches_app_without_echoing_config_values(
    tmp_path: Path,
) -> None:
    canary = "SYNTHETIC_CONFIG_CANARY_417"
    config_file = tmp_path / ".env"
    config_file.write_text(
        f"BASE=http://127.0.0.1:11434\n"
        "OLLAMA_BASE_URL=${BASE}\n"
        f"OLLAMA_MODEL={canary}\n",
        encoding="utf-8",
    )
    output = StringIO()

    exit_code = doctor.run_doctor(environ={}, config_path=config_file, output=output)

    assert exit_code == 0
    assert "PASS llm_config: valid syntax; endpoint availability not checked" in output.getvalue()
    assert "http://127.0.0.1:11434" not in output.getvalue()
    assert "${BASE}" not in output.getvalue()
    assert canary not in output.getvalue()
    assert str(config_file) not in output.getvalue()


def test_unreadable_config_file_fails_without_echoing_location(tmp_path: Path) -> None:
    config_directory = tmp_path / "private-config"
    config_directory.mkdir()
    output = StringIO()

    exit_code = doctor.run_doctor(
        environ={}, config_path=config_directory, output=output
    )

    assert exit_code == 1
    assert "FAIL config_file: not a regular file" in output.getvalue()
    assert str(config_directory) not in output.getvalue()


def test_non_regular_config_file_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    source_dir = Path(__file__).resolve().parents[1] / "src"
    fifo_path = tmp_path / "private-config-pipe"
    os.mkfifo(fifo_path)
    code = (
        "import sys; from pathlib import Path; "
        "from openscreen_ai.doctor import run_doctor; "
        "raise SystemExit(run_doctor(config_path=Path(sys.argv[1])))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code, str(fifo_path)],
        cwd=tmp_path,
        env={"PYTHONPATH": str(source_dir), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 1
    assert "FAIL config_file: not a regular file" in result.stdout
    assert str(fifo_path) not in result.stdout + result.stderr
    assert result.stderr == ""


def test_symlink_config_file_is_rejected_without_reading_target(tmp_path: Path) -> None:
    canary = "SYNTHETIC_SYMLINK_CONFIG_CANARY"
    target = tmp_path / "actual-config"
    target.write_text(f"OLLAMA_MODEL={canary}\n", encoding="utf-8")
    config_link = tmp_path / ".env"
    config_link.symlink_to(target)
    output = StringIO()

    exit_code = doctor.run_doctor(environ={}, config_path=config_link, output=output)

    assert exit_code == 1
    assert "FAIL config_file: not a regular file" in output.getvalue()
    assert canary not in output.getvalue()
    assert str(config_link) not in output.getvalue()


def test_config_file_replacement_between_metadata_check_and_open_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary = "SYNTHETIC_CONFIG_REPLACEMENT_CANARY"
    config_file = tmp_path / ".env"
    replacement = tmp_path / "replacement.env"
    config_file.write_text("OLLAMA_MODEL=ordinary-model\n", encoding="utf-8")
    replacement.write_text(f"OLLAMA_MODEL={canary}\n", encoding="utf-8")
    real_open = os.open
    replaced = False

    def replace_before_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if Path(path) == config_file and not replaced:
            replaced = True
            replacement.replace(config_file)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(doctor.os, "open", replace_before_open)
    output = StringIO()

    exit_code = doctor.run_doctor(environ={}, config_path=config_file, output=output)

    assert exit_code == 1
    assert "FAIL config_file: changed before reading" in output.getvalue()
    assert canary not in output.getvalue()
    assert str(config_file) not in output.getvalue()


def test_oversized_config_file_is_rejected_without_echoing_content_or_path(
    tmp_path: Path,
) -> None:
    canary = b"SYNTHETIC_OVERSIZE_CANARY"
    config_file = tmp_path / ".env"
    config_file.write_bytes(b"x" * doctor._MAX_CONFIG_FILE_BYTES + canary)
    output = StringIO()

    exit_code = doctor.run_doctor(environ={}, config_path=config_file, output=output)

    assert exit_code == 1
    assert "FAIL config_file: exceeds safe size limit" in output.getvalue()
    assert canary.decode() not in output.getvalue()
    assert str(config_file) not in output.getvalue()


@pytest.mark.parametrize(
    "config_value",
    [
        "OLLAMA_BASE_URL=https://example.invalid/endpoint\n",
        "OLLAMA_BASE_URL=http://user:synthetic-secret@localhost:11434\n",
        "OLLAMA_TIMEOUT_SECONDS=nan\n",
        "OLLAMA_MODEL=\n",
    ],
)
def test_invalid_llm_config_fails_without_echoing_values(
    tmp_path: Path, config_value: str
) -> None:
    secret = "SYNTHETIC_CONFIG_VALUE_671"
    config_value = config_value.replace("synthetic-secret", secret)
    config_file = tmp_path / ".env"
    config_file.write_text(config_value, encoding="utf-8")
    output = StringIO()

    exit_code = doctor.run_doctor(environ={}, config_path=config_file, output=output)

    assert exit_code == 1
    assert "FAIL llm_config: invalid local configuration" in output.getvalue()
    assert secret not in output.getvalue()
    assert str(config_file) not in output.getvalue()


def test_partial_stt_configuration_is_warning_not_failure(tmp_path: Path) -> None:
    checks = doctor.collect_doctor_checks(
        environ={"OPENSCREEN_STT_MODEL_DIR": str(tmp_path / "private-model-path")},
        config_path=tmp_path / "absent.env",
    )

    stt = check_by_category(checks, "stt")
    assert stt.status == "WARN"
    assert "private-model-path" not in stt.detail
    assert not any(item.status == "FAIL" for item in checks if item.category.startswith("stt"))


def test_configured_stt_reports_model_prerequisites_without_echoing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_model_dir = tmp_path / "private-models"
    private_model_dir.mkdir()
    vad_file = tmp_path / "private-vad.onnx"
    vad_file.touch()
    monkeypatch.setattr(
        doctor,
        "_is_available",
        lambda name: name in {"httpx", "dotenv", "PySide6.QtWidgets"},
    )
    values = {
        "OPENSCREEN_STT_MODEL_DIR": str(private_model_dir),
        "OPENSCREEN_STT_VAD_MODEL_PATH": str(vad_file),
    }
    output = StringIO()

    exit_code = doctor.run_doctor(
        environ=values,
        config_path=tmp_path / "absent.env",
        output=output,
    )

    assert exit_code == 0
    assert "WARN stt_model_paths: optional model files missing or unreadable" in output.getvalue()
    assert str(private_model_dir) not in output.getvalue()
    assert str(vad_file) not in output.getvalue()


def test_configured_stt_python_outside_supported_range_is_optional_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "sys", SimpleNamespace(version_info=(3, 14, 0)))

    checks = doctor.collect_doctor_checks(
        environ={
            "OPENSCREEN_STT_MODEL_DIR": str(tmp_path / "models"),
            "OPENSCREEN_STT_VAD_MODEL_PATH": str(tmp_path / "vad.onnx"),
        },
        config_path=tmp_path / "absent.env",
    )

    assert check_by_category(checks, "python").status == "PASS"
    assert check_by_category(checks, "stt_python").status == "WARN"


def test_module_entrypoint_rejects_unknown_command_without_sensitive_data(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(module_entrypoint.sys, "argv", ["openscreen_ai", "secret-value"])

    exit_code = module_entrypoint.main()

    assert exit_code == 2
    assert capsys.readouterr().err == "Usage: python -m openscreen_ai doctor\n"
