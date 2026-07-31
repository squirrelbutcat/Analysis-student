# Анализ влияния социальных сетей на студентов

Python-проект для разведочного анализа Kaggle-датасета
`shivasingh4945/student-social-media-and-mental-health-impact`.

Проект скачивает датасет, очищает данные с помощью Pandas, строит сводные
таблицы, графики, итоговый Markdown-отчет и красивый HTML-дашборд на русском языке.

## Структура проекта

```text
.
├── data/
│   ├── raw/                 # исходные CSV-файлы
│   └── processed/           # очищенные данные и сводные таблицы
├── reports/
│   ├── figures/             # сгенерированные графики
│   ├── analysis_report.md   # итоговый отчет
│   └── dashboard.html       # общий дашборд
├── scripts/
│   └── download_data.py     # загрузчик через KaggleHub
└── src/student_analysis/
    └── analyze.py           # пайплайн анализа на Pandas
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Загрузка датасета

```bash
python scripts/download_data.py
```

Скрипт использует:

```python
import kagglehub

path = kagglehub.dataset_download(
    "shivasingh4945/student-social-media-and-mental-health-impact"
)
print("Path to dataset files:", path)
```

## Запуск анализа

```bash
python src/student_analysis/analyze.py
```

Можно также передать конкретный CSV-файл:

```bash
python src/student_analysis/analyze.py --input data/raw/Students\ Social\ Media\ Addiction.csv
```

## Результаты

- `data/processed/cleaned_student_social_media.csv`
- `data/processed/numeric_summary.csv`
- таблицы распределений для найденных категориальных колонок
- `data/processed/key_correlations.csv`
- `reports/figures/*.png`
- `reports/analysis_report.md`
- `reports/dashboard.html`

## Вопросы для анализа

- Сколько часов в день студенты проводят в социальных сетях?
- Какие платформы встречаются чаще всего?
- Как связаны время в социальных сетях, сон, учеба, физическая активность и показатель ментального здоровья?
- Есть ли заметные различия по полу, стране, уровню обучения или уровню стресса?
