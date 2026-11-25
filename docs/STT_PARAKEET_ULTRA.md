# Настройка Parakeet Ultra INT8

OpenScreen-AI использует существующий `SttEngine` на базе Sherpa-ONNX. Для
Parakeet Ultra не требуется отдельный runtime или новая Python-зависимость:
проверенный community export имеет тот же transducer layout, который уже
ожидает движок. TDT v2 остаётся доступен: достаточно указать каталог его
совместимого export вместо каталога Ultra.

## Совместимость и происхождение

Проверенная сборка — [`leogaudin/sherpa-onnx-nemo-parakeet-ultra-int8` на
закреплённой revision](https://huggingface.co/leogaudin/sherpa-onnx-nemo-parakeet-ultra-int8/tree/f7bc472677ff70be95cdb0ef9df509fee6f5ec0b),
revision `f7bc472677ff70be95cdb0ef9df509fee6f5ec0b`, для Sherpa-ONNX 2.0.1 на
macOS arm64. Это community conversion, а не официальный Sherpa-ONNX release и
не сборка, одобренная Moondream/NVIDIA. Model card указывает CC-BY-4.0; перед
распространением модели проверьте требования атрибуции и лицензию conversion.

Закреплённая ревизия содержит следующие файлы:

| Файл | Размер, байт | SHA-256 |
|---|---:|---|
| `encoder.int8.onnx` | 652282340 | `8bd3b8dc07056def6dfd19910e9f90c9e1a4e7e20901f58bdfe2a570a8081a2a` |
| `decoder.int8.onnx` | 11845275 | `94373c6405c8149d1c9d1e9862514e0e3e88dab1c7de18fac9d5df9c6ba5efdd` |
| `joiner.int8.onnx` | 6355277 | `8a71aaccfdba3d451775507a889afd2257d2e754463fd81095a8c8f290f924c2` |
| `tokens.txt` | 93939 | `d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d` |

Эти hashes сверены с `SHA256SUMS` закреплённой upstream revision. Небольшой
реальный прогон на публичном FLEURS наборе подтвердил техническую загрузку и
распознавание русской речи на M4/arm64, но выборка из трёх русских фраз не
является достаточной оценкой качества или production readiness. Подробные
методика, ограничения и результаты должны сопровождать этот документ в
исследовательском отчёте STT-004.

## Настройка

Сначала убедитесь, что выбранный каталог содержит все четыре файла выше и что
рядом доступна отдельная совместимая модель Silero VAD. Не помещайте модели в
Git checkout. В локальном `.env` задайте пути к уже подготовленным файлам:

```dotenv
OPENSCREEN_STT_MODEL_DIR=/path/to/parakeet-ultra-int8
OPENSCREEN_STT_VAD_MODEL_PATH=/path/to/silero_vad.onnx
```

Либо задайте их в окружении процесса. Эти имена уже читаются composition root
в `src/openscreen_ai/main.py`; конфигурация передаётся через `SttConfig`.
`src/openscreen_ai/stt/availability.py` проверяет наличие зависимостей и
четырёх ожидаемых файлов до создания STT controller. Сам `SttEngine` создаёт
`OfflineRecognizer.from_transducer` с `model_type="nemo_transducer"`, частотой
16 кГц, feature dimension 80 и текущими `num_threads`/`provider` из
`SttConfig`.

Перед запуском можно проверить только локальные предварительные условия:

```sh
python -m openscreen_ai doctor
```

Doctor не загружает модели и не проверяет фактическую совместимость inference.
При явном запуске STT веса загружаются лениво; не запускать модель в рамках
проверки конфигурации. Для управляемого реального испытания используйте
отдельное разрешение и утверждённый runtime test plan.

## Текущие границы

- Декодирование остаётся на CPU и использует параметры `SttConfig`; этот
  документ не включает выбор Metal/Core ML provider.
- Итоговая пользовательская диктовка формируется из VAD-сегментов и остаётся
  в поле ввода после Stop; LLM запускается только после ручного нажатия
  «Отправить». Отмена, ошибка или превышение лимита сбрасывают
  частичный результат.
- Агрегация ограничена 256 сегментами и 32 000 символами на сеанс.
- Точная модель и версия Sherpa должны оставаться закреплёнными; новая revision
  требует повторной проверки checksum и inference.
- Тесты с фейковыми моделями подтверждают wiring и lifecycle, но не заменяют
  измерения качества, latency, RSS, микрофона и полного GUI flow.
