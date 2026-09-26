# Практическое задание «Основы нейронных сетей»

Репозиторий соответствует отчёту и состоит из трёх частей:

1. `part1_mlp` — объектно-ориентированная реализация многослойного перцептрона без готового ML-фреймворка;
2. `part2_prepare` — CLI-утилита подготовки изображений/CSV в NumPy-массив;
3. `part3_parallel` — producer-consumer на потоках и CPU-аугментация через `multiprocessing.Pool`.

## Установка

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

```bash
python -m part1_mlp.app.demo
python -m part1_mlp.app.main
python -m part2_prepare.app.cli prepare --path data/input --ext png --output data/output/prepared.npy
python -m part2_prepare.app.cli doctor
python -m part3_parallel.app.main --path data/input --report data/output/performance_report.json
```

## Проверка

```bash
python -m compileall part1_mlp part2_prepare part3_parallel
```

В репозитории есть локальная история Git с отдельными коммитами по основным частям проекта.
