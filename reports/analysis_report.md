# Анализ влияния социальных сетей на студентов

Исходный файл: `/Users/homosanians/Documents/student-analysis/data/raw/Student Social Media And Mental Health Impact.csv`
Строк: 5,000
Колонок: 13

## Использованные колонки

- `age` -> `age`
- `gender` -> `gender`
- `academic_level` -> `academic_level`
- `country` -> `country`
- `usage_hours` -> `avg_daily_usage_hours`
- `daily_unlocks` -> `daily_unlocks`
- `platform` -> `most_used_platform`
- `purpose` -> `purpose_of_use`
- `study_hours` -> `study_hours`
- `physical_activity_hours` -> `physical_activity_hours`
- `sleep_hours` -> `sleep_hours_per_night`
- `mental_health_score` -> `mental_health_score`
- `stress_level` -> `stress_level`

## Краткие выводы

- Среднее ежедневное использование соцсетей: 5.08 часа.
- Средний показатель ментального здоровья: 6.23.
- Средняя длительность сна: 6.63 часа.
- Среднее время учебы: 3.01 часа.
- Средняя физическая активность: 1.75 часа.
- Среднее количество разблокировок телефона в день: 171.5.
- Самая распространенная платформа: Instagram.
- В данных есть сильная отрицательная корреляция: больше времени в соцсетях связано с меньшим сном (`-0.808`) и более низким показателем ментального здоровья (`-0.816`). Это связь в данных, а не доказательство причинности.
- Самая сильная ключевая корреляция: `avg_daily_usage_hours` и `daily_unlocks` = 0.960.

## Пропущенные значения

- `age`: 0.0%
- `gender`: 0.0%
- `country`: 0.0%
- `academic_level`: 0.0%
- `most_used_platform`: 0.0%
- `purpose_of_use`: 0.0%
- `avg_daily_usage_hours`: 0.0%
- `daily_unlocks`: 0.0%
- `study_hours`: 0.0%
- `physical_activity_hours`: 0.0%

## Числовые колонки

- `age`
- `avg_daily_usage_hours`
- `daily_unlocks`
- `study_hours`
- `physical_activity_hours`
- `sleep_hours_per_night`
- `mental_health_score`

## Сгенерированные таблицы

- `data/processed/numeric_summary.csv`
- `data/processed/gender_distribution.csv`
- `data/processed/academic_level_distribution.csv`
- `data/processed/country_distribution.csv`
- `data/processed/platform_distribution.csv`
- `data/processed/purpose_distribution.csv`
- `data/processed/stress_level_distribution.csv`
- `data/processed/key_correlations.csv`

## Сгенерированные графики

- `reports/figures/stress_level_donut.png`
- `reports/figures/purpose_of_use_donut.png`
- `reports/figures/key_correlations_heatmap.png`
- `reports/figures/usage_hours_distribution.png`
- `reports/figures/platform_distribution.png`
- `reports/figures/sleep_vs_mental_health.png`
- `reports/figures/usage_vs_mental_health.png`
- `reports/figures/usage_vs_study_hours.png`
- `reports/figures/activity_vs_mental_health.png`
