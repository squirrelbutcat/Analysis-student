from __future__ import annotations

import argparse
from html import escape
import re
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
import plotly.graph_objects as go

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


DISPLAY_NAMES = {
    "age": "Возраст",
    "gender": "Пол",
    "academic_level": "Уровень обучения",
    "country": "Страна",
    "avg_daily_usage_hours": "Соцсети, часов в день",
    "daily_unlocks": "Разблокировки телефона в день",
    "most_used_platform": "Основная платформа",
    "purpose_of_use": "Цель использования",
    "study_hours": "Учеба, часов в день",
    "physical_activity_hours": "Физическая активность, часов в день",
    "sleep_hours_per_night": "Сон, часов в сутки",
    "mental_health_score": "Показатель ментального здоровья",
    "stress_level": "Уровень стресса",
}

CHART_BG = "#f6f4ef"
CHART_PANEL = "#fffdf8"
CHART_GRID = "#d9ded7"
CHART_TEXT = "#17211d"
CHART_MUTED = "#66736d"
CHART_GREEN = "#197278"
CHART_BRICK = "#b85b4b"
CHART_DARK_GREEN = "#283d3b"
CHART_MOSS = "#4f6f52"
CHART_SAGE = "#8aa899"
CHART_GOLD = "#c59b4b"
CHART_PALETTE = [CHART_GREEN, CHART_SAGE, CHART_MOSS, CHART_GOLD, CHART_DARK_GREEN, "#6f8f72"]


COLUMN_CANDIDATES = {
    "student_id": ["student_id", "student id", "id"],
    "age": ["age"],
    "gender": ["gender"],
    "academic_level": ["academic_level", "academic level", "education_level", "education level"],
    "country": ["country"],
    "usage_hours": ["avg_daily_usage_hours", "daily social media time", "usage hours", "daily_usage"],
    "daily_unlocks": ["daily_unlocks", "daily unlocks", "unlocks"],
    "platform": ["most_used_platform", "dominant platform", "platform"],
    "purpose": ["purpose_of_use", "purpose of use", "purpose"],
    "academic_impact": ["affects_academic_performance", "academic impact", "performance"],
    "study_hours": ["study_hours", "study hours"],
    "physical_activity_hours": ["physical_activity_hours", "physical activity hours", "activity hours"],
    "sleep_hours": ["sleep_hours_per_night", "sleep duration", "sleep hours"],
    "mental_health_score": ["mental_health_score", "mental health score", "wellbeing", "stress"],
    "stress_level": ["stress_level", "stress level"],
    "relationship_status": ["relationship_status", "relationship status"],
    "conflicts": ["conflicts_over_social_media", "conflicts", "argument"],
    "addiction_score": ["addicted_score", "addiction score", "addicted"],
}


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def find_csv(input_path: Path | None) -> Path:
    if input_path:
        if input_path.is_dir():
            csv_files = sorted(input_path.rglob("*.csv"))
        else:
            csv_files = [input_path]
    else:
        csv_files = sorted(RAW_DIR.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "CSV-файл не найден. Сначала запустите `python scripts/download_data.py` "
            "или передайте путь к CSV через `--input`."
        )
    return csv_files[0]


def read_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [normalize_name(column) for column in df.columns]
    return df


def resolve_columns(columns: Iterable[str]) -> dict[str, str | None]:
    available = list(columns)
    resolved: dict[str, str | None] = {}

    for logical_name, candidates in COLUMN_CANDIDATES.items():
        normalized_candidates = {normalize_name(candidate) for candidate in candidates}
        exact_match = next((column for column in available if column in normalized_candidates), None)
        if exact_match:
            resolved[logical_name] = exact_match
            continue

        fuzzy_match = next(
            (
                column
                for column in available
                if any(candidate in column or column in candidate for candidate in normalized_candidates)
            ),
            None,
        )
        resolved[logical_name] = fuzzy_match

    return resolved


def clean_dataset(df: pd.DataFrame, columns: dict[str, str | None]) -> pd.DataFrame:
    cleaned = df.copy()

    numeric_fields = [
        "age",
        "usage_hours",
        "daily_unlocks",
        "study_hours",
        "physical_activity_hours",
        "sleep_hours",
        "mental_health_score",
        "conflicts",
        "addiction_score",
    ]
    for field in numeric_fields:
        column = columns.get(field)
        if column and column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    text_fields = [
        "gender",
        "academic_level",
        "country",
        "platform",
        "purpose",
        "academic_impact",
        "stress_level",
        "relationship_status",
    ]
    for field in text_fields:
        column = columns.get(field)
        if column and column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("string").str.strip()

    return cleaned


def save_summary_tables(df: pd.DataFrame, columns: dict[str, str | None]) -> dict[str, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        path = PROCESSED_DIR / "numeric_summary.csv"
        df[numeric_columns].describe().T.to_csv(path)
        outputs["numeric_summary"] = path

    for field in [
        "gender",
        "academic_level",
        "country",
        "platform",
        "purpose",
        "academic_impact",
        "stress_level",
        "relationship_status",
    ]:
        column = columns.get(field)
        if column and column in df.columns:
            path = PROCESSED_DIR / f"{field}_distribution.csv"
            df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="count").to_csv(path, index=False)
            outputs[f"{field}_distribution"] = path

    key_numeric = [
        columns.get(field)
        for field in [
            "usage_hours",
            "daily_unlocks",
            "study_hours",
            "physical_activity_hours",
            "sleep_hours",
            "mental_health_score",
            "conflicts",
            "addiction_score",
        ]
    ]
    key_numeric = [column for column in key_numeric if column and column in df.columns]
    if len(key_numeric) >= 2:
        path = PROCESSED_DIR / "key_correlations.csv"
        df[key_numeric].corr(numeric_only=True).round(3).to_csv(path)
        outputs["key_correlations"] = path

    return outputs


FONT_FAMILY = "Avenir Next, Avenir, Optima, Candara, 'Segoe UI', sans-serif"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, round(channel))) for channel in rgb))


def color_ramp(base_hex: str, n: int, light_hex: str = "#eaf2ee") -> list[str]:
    """Colors ordered from `base_hex` (index 0) fading to `light_hex`, for series sorted by size."""
    if n <= 1:
        return [base_hex]
    base = hex_to_rgb(base_hex)
    light = hex_to_rgb(light_hex)
    return [
        rgb_to_hex(tuple(base[c] + (light[c] - base[c]) * i / (n - 1) for c in range(3)))
        for i in range(n)
    ]


def gaussian_kde_line(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray] | None:
    values = values[~np.isnan(values)]
    n = values.size
    std = values.std(ddof=1) if n > 1 else 0
    if n < 2 or std == 0 or values.min() == values.max():
        return None

    bandwidth = 1.06 * std * n ** (-1 / 5)
    x_grid = np.linspace(values.min(), values.max(), 200)
    diffs = (x_grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * diffs**2).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))
    bin_width = (values.max() - values.min()) / bins
    return x_grid, density * n * bin_width


def base_layout(title: str, height: int = 420) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=CHART_TEXT, size=16, family=FONT_FAMILY), x=0.01, xanchor="left"),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_PANEL,
        font=dict(color=CHART_MUTED, family=FONT_FAMILY, size=12),
        margin=dict(l=54, r=24, t=56, b=48),
        height=height,
        hoverlabel=dict(bgcolor=CHART_PANEL, font_color=CHART_TEXT, bordercolor=CHART_GRID, font_family=FONT_FAMILY),
        legend=dict(font=dict(color=CHART_MUTED), bgcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )


def style_axes(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(gridcolor=CHART_GRID, zeroline=False, linecolor=CHART_GRID, tickfont=dict(color=CHART_MUTED))
    fig.update_yaxes(gridcolor=CHART_GRID, zeroline=False, linecolor=CHART_GRID, tickfont=dict(color=CHART_MUTED))
    return fig


def build_donut(counts: pd.Series, title: str) -> go.Figure:
    labels = counts.index.astype(str).tolist()
    values = counts.values.tolist()
    colors = color_ramp(CHART_GREEN, len(values))

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.58,
            sort=False,
            marker=dict(colors=colors, line=dict(color=CHART_PANEL, width=2)),
            textinfo="percent",
            textfont=dict(color=CHART_TEXT, size=12),
            hovertemplate="%{label}<br>%{value} студентов (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(**base_layout(title, height=380))
    fig.update_layout(showlegend=True, legend=dict(font=dict(color=CHART_MUTED)))
    return fig


def build_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    labels = [DISPLAY_NAMES.get(column, column) for column in corr.columns]
    z = corr.values
    text = [[f"{value:.2f}" for value in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[[0, "#9aa081"], [0.5, "#f3e7d0"], [1, CHART_GREEN]],
            text=text,
            texttemplate="%{text}",
            textfont=dict(color=CHART_TEXT, size=11),
            hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
            xgap=2,
            ygap=2,
            colorbar=dict(outlinewidth=0, tickfont=dict(color=CHART_MUTED)),
        )
    )
    fig.update_layout(**base_layout(title, height=460))
    fig.update_yaxes(autorange="reversed")
    return style_axes(fig)


def build_histogram(series: pd.Series, title: str, x_label: str) -> go.Figure:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    bins = 12

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            marker=dict(color=CHART_GREEN, line=dict(color=CHART_PANEL, width=1)),
            name="Студенты",
            hovertemplate=f"{x_label}: " + "%{x}<br>Студентов: %{y}<extra></extra>",
        )
    )
    kde = gaussian_kde_line(values, bins)
    if kde is not None:
        x_grid, y_kde = kde
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_kde,
                mode="lines",
                line=dict(color=CHART_BRICK, width=2),
                name="Плотность",
                hoverinfo="skip",
            )
        )

    fig.update_layout(**base_layout(title))
    fig.update_layout(bargap=0.05)
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="Количество студентов")
    return style_axes(fig)


def build_bar(counts: pd.Series, title: str) -> go.Figure:
    ordered = counts.sort_values(ascending=True)
    colors = list(reversed(color_ramp(CHART_GREEN, len(ordered))))

    fig = go.Figure(
        go.Bar(
            x=ordered.values,
            y=ordered.index.astype(str),
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}: %{x} студентов<extra></extra>",
        )
    )
    fig.update_layout(**base_layout(title, height=max(320, 34 * len(ordered) + 120)))
    fig.update_xaxes(title="Количество студентов")
    fig.update_yaxes(title=None, gridcolor=CHART_BG)
    return style_axes(fig)


def build_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    pair = df[[x_col, y_col]].dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pair[x_col],
            y=pair[y_col],
            mode="markers",
            marker=dict(color=CHART_GREEN, size=8, opacity=0.45, line=dict(width=0)),
            name="Студенты",
            hovertemplate=(
                f"{DISPLAY_NAMES.get(x_col, x_col)}: "
                + "%{x}<br>"
                + f"{DISPLAY_NAMES.get(y_col, y_col)}: "
                + "%{y}<extra></extra>"
            ),
        )
    )
    if len(pair) >= 2:
        slope, intercept = np.polyfit(pair[x_col], pair[y_col], 1)
        x_line = np.linspace(pair[x_col].min(), pair[x_col].max(), 100)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=slope * x_line + intercept,
                mode="lines",
                line=dict(color=CHART_BRICK, width=2),
                name="Тренд",
                hoverinfo="skip",
            )
        )

    fig.update_layout(**base_layout(title))
    fig.update_xaxes(title=DISPLAY_NAMES.get(x_col, x_col))
    fig.update_yaxes(title=DISPLAY_NAMES.get(y_col, y_col))
    return style_axes(fig)


def save_figures(df: pd.DataFrame, columns: dict[str, str | None]) -> list[dict]:
    usage_col = columns.get("usage_hours")
    addiction_col = columns.get("addiction_score")
    mental_col = columns.get("mental_health_score")
    sleep_col = columns.get("sleep_hours")
    study_col = columns.get("study_hours")
    activity_col = columns.get("physical_activity_hours")
    platform_col = columns.get("platform")
    stress_col = columns.get("stress_level")
    purpose_col = columns.get("purpose")
    key_numeric = [
        column
        for column in [usage_col, columns.get("daily_unlocks"), study_col, activity_col, sleep_col, mental_col]
        if column and column in df.columns
    ]

    charts: list[dict] = []

    if stress_col and stress_col in df.columns:
        title = "Распределение уровня стресса"
        charts.append(
            {
                "id": "stress_level_donut",
                "title": title,
                "kind": "distribution",
                "fig": build_donut(df[stress_col].value_counts(), title),
            }
        )

    if purpose_col and purpose_col in df.columns:
        title = "Цели использования соцсетей"
        charts.append(
            {
                "id": "purpose_of_use_donut",
                "title": title,
                "kind": "distribution",
                "fig": build_donut(df[purpose_col].value_counts().head(6), title),
            }
        )

    if len(key_numeric) >= 2:
        title = "Корреляции ключевых показателей"
        corr = df[key_numeric].corr(numeric_only=True)
        charts.append(
            {
                "id": "key_correlations_heatmap",
                "title": title,
                "kind": "correlation",
                "fig": build_heatmap(corr, title),
            }
        )

    if usage_col and usage_col in df.columns:
        title = "Ежедневное использование социальных сетей"
        charts.append(
            {
                "id": "usage_hours_distribution",
                "title": title,
                "kind": "distribution",
                "fig": build_histogram(df[usage_col], title, "Часов в день"),
            }
        )

    if platform_col and platform_col in df.columns:
        top_platforms = df[platform_col].value_counts().head(10)
        if not top_platforms.empty:
            title = "Самые популярные платформы"
            charts.append(
                {
                    "id": "platform_distribution",
                    "title": title,
                    "kind": "distribution",
                    "fig": build_bar(top_platforms, title),
                }
            )

    scatter_pairs = [
        (usage_col, addiction_col, "Время в соцсетях и показатель зависимости", "usage_vs_addiction"),
        (sleep_col, mental_col, "Сон и показатель ментального здоровья", "sleep_vs_mental_health"),
        (usage_col, mental_col, "Время в соцсетях и показатель ментального здоровья", "usage_vs_mental_health"),
        (usage_col, study_col, "Время в соцсетях и время учебы", "usage_vs_study_hours"),
        (
            activity_col,
            mental_col,
            "Физическая активность и показатель ментального здоровья",
            "activity_vs_mental_health",
        ),
    ]
    for x_col, y_col, title, chart_id in scatter_pairs:
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            charts.append(
                {
                    "id": chart_id,
                    "title": title,
                    "kind": "relationship",
                    "fig": build_scatter(df, x_col, y_col, title),
                }
            )

    return charts


def format_number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}".replace(",", " ")


def metric_card(label: str, value: str, note: str) -> str:
    match = re.search(r"[\d\s]+(?:\.\d+)?", value)
    target = match.group(0).replace(" ", "") if match else ""
    suffix = value[match.end() :] if match else ""
    return f"""
        <article class="metric">
            <span>{escape(label)}</span>
            <strong data-target="{escape(target)}" data-suffix="{escape(suffix)}">{escape(value)}</strong>
            <small>{escape(note)}</small>
        </article>
    """


def distribution_table(df: pd.DataFrame, column: str, title: str, limit: int = 6) -> str:
    counts = df[column].value_counts().head(limit).reset_index()
    counts.columns = [title, "Количество"]
    return counts.to_html(index=False, classes="data-table", border=0)


def build_html_dashboard(
    df: pd.DataFrame,
    csv_path: Path,
    columns: dict[str, str | None],
    figure_outputs: list[dict],
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "dashboard.html"

    usage_col = columns.get("usage_hours")
    mental_col = columns.get("mental_health_score")
    sleep_col = columns.get("sleep_hours")
    study_col = columns.get("study_hours")
    activity_col = columns.get("physical_activity_hours")
    daily_unlocks_col = columns.get("daily_unlocks")
    platform_col = columns.get("platform")
    stress_col = columns.get("stress_level")
    purpose_col = columns.get("purpose")

    metrics = [metric_card("Наблюдений", f"{len(df):,}".replace(",", " "), "строк в датасете")]
    if usage_col and usage_col in df.columns:
        metrics.append(metric_card("Соцсети", f"{format_number(df[usage_col].mean())} ч", "в среднем в день"))
    if daily_unlocks_col and daily_unlocks_col in df.columns:
        metrics.append(metric_card("Разблокировки", f"{format_number(df[daily_unlocks_col].mean(), 1)}", "в среднем в день"))
    if sleep_col and sleep_col in df.columns:
        metrics.append(metric_card("Сон", f"{format_number(df[sleep_col].mean())} ч", "в среднем в сутки"))
    if study_col and study_col in df.columns:
        metrics.append(metric_card("Учеба", f"{format_number(df[study_col].mean())} ч", "в среднем в день"))
    if mental_col and mental_col in df.columns:
        metrics.append(metric_card("Ментальное здоровье", format_number(df[mental_col].mean()), "средний показатель"))

    insights: list[tuple[str, str]] = []
    if platform_col and platform_col in df.columns:
        top_platform = df[platform_col].mode(dropna=True).iloc[0]
        top_count = int(df[platform_col].value_counts().iloc[0])
        insights.append(
            (
                f"Самая популярная платформа - {top_platform}",
                f"Ее указали {top_count} студентов. Остальные платформы можно сравнить в таблице распределений ниже.",
            )
        )
    if stress_col and stress_col in df.columns:
        stress_counts = df[stress_col].value_counts()
        high_stress = int(stress_counts.get("High", 0) + stress_counts.get("Very High", 0))
        high_stress_share = high_stress / len(df) * 100
        insights.append(
            (
                f"Высокий стресс у {high_stress_share:.1f}% студентов",
                "В эту группу входят ответы High и Very High, поэтому это один из главных сигналов для интерпретации датасета.",
            )
        )
        top_stress = stress_counts.index[0]
        top_stress_share = stress_counts.iloc[0] / len(df) * 100
        insights.append(
            (
                f"Самая частая категория стресса - {top_stress}",
                f"На нее приходится {top_stress_share:.1f}% выборки. Это помогает понять общий фон ответов студентов.",
            )
        )
    if usage_col and sleep_col and mental_col and usage_col in df.columns and sleep_col in df.columns and mental_col in df.columns:
        usage_sleep_corr = df[[usage_col, sleep_col]].corr(numeric_only=True).iloc[0, 1]
        usage_mental_corr = df[[usage_col, mental_col]].corr(numeric_only=True).iloc[0, 1]
        insights.append(
            (
                "Соцсети заметно связаны со сном и ментальным здоровьем",
                "Чем больше времени в соцсетях, тем ниже сон "
                f"({usage_sleep_corr:.3f}) и показатель ментального здоровья ({usage_mental_corr:.3f}). "
                "Это корреляция в данных, а не доказательство причины.",
            )
        )
    if usage_col and study_col and usage_col in df.columns and study_col in df.columns:
        usage_study_corr = df[[usage_col, study_col]].corr(numeric_only=True).iloc[0, 1]
        insights.append(
            (
                "Больше соцсетей - меньше учебного времени",
                f"Корреляция между временем в соцсетях и временем учебы равна {usage_study_corr:.3f}.",
            )
        )
    if usage_col and usage_col in df.columns:
        median_usage = df[usage_col].median()
        heavy_usage_share = (df[usage_col] >= 6).mean() * 100
        insights.append(
            (
                f"Медианное время в соцсетях - {median_usage:.1f} часа",
                f"При этом 6+ часов в день набирают {heavy_usage_share:.1f}% студентов.",
            )
        )
    key_cols = [
        column
        for column in [usage_col, daily_unlocks_col, study_col, activity_col, sleep_col, mental_col]
        if column and column in df.columns
    ]
    if len(key_cols) >= 2:
        corr = df[key_cols].corr(numeric_only=True)
        pairs = []
        for left_index, left in enumerate(corr.columns):
            for right in corr.columns[left_index + 1 :]:
                pairs.append((abs(corr.loc[left, right]), left, right, corr.loc[left, right]))
        if pairs:
            _, left, right, value = max(pairs)
            insights.append(
                (
                    "Самая сильная числовая связь",
                    f"{DISPLAY_NAMES.get(left, left)} и {DISPLAY_NAMES.get(right, right)}: корреляция {value:.3f}.",
                )
            )

    table_sections = []
    for column, title in [
        (platform_col, "Платформа"),
        (stress_col, "Уровень стресса"),
        (purpose_col, "Цель использования"),
    ]:
        if column and column in df.columns:
            table_sections.append(
                f"""
                <section class="panel">
                    <h2>{escape(title)}</h2>
                    {distribution_table(df, column, title)}
                </section>
                """
            )

    figure_cards = []
    plotly_config = {"displayModeBar": False, "responsive": True}
    include_plotlyjs: bool | str = "cdn"
    for chart in figure_outputs:
        chart_html = chart["fig"].to_html(
            full_html=False,
            include_plotlyjs=include_plotlyjs,
            config=plotly_config,
            div_id=chart["id"],
        )
        include_plotlyjs = False
        figure_cards.append(
            f"""
            <article class="chart-card" data-kind="{chart['kind']}">
                <h3>{escape(chart['title'])}</h3>
                {chart_html}
            </article>
            """
        )

    html = f"""<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Дашборд: соцсети и ментальное здоровье студентов</title>
    <style>
        :root {{
            --bg: #f6f4ef;
            --ink: #17211d;
            --muted: #66736d;
            --panel: #ffffff;
            --line: #d9ded7;
            --accent: #197278;
            --accent-2: #c44536;
            --accent-3: #283d3b;
            --soft: #e8f2ef;
            --warm: #efe6d4;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            background: var(--bg);
            color: var(--ink);
            font-family: Avenir Next, Avenir, Optima, Candara, "Segoe UI", sans-serif;
            line-height: 1.5;
        }}
        header {{
            position: relative;
            overflow: hidden;
            background: #17211d;
            color: #fff;
            padding: 48px 24px 54px;
        }}
        header::after {{
            content: "";
            position: absolute;
            right: max(24px, calc((100vw - 1180px) / 2));
            top: 24px;
            width: 144px;
            height: 144px;
            border-radius: 50%;
            border: 18px solid rgba(155, 209, 202, .2);
            border-top-color: #9bd1ca;
            animation: spin 18s linear infinite;
        }}
        .wrap {{
            position: relative;
            z-index: 1;
            width: min(1180px, calc(100% - 32px));
            margin: 0 auto;
        }}
        .eyebrow {{
            color: #9bd1ca;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
        }}
        h1 {{
            max-width: 900px;
            margin: 0;
            font-size: clamp(34px, 5vw, 64px);
            line-height: 1.02;
            letter-spacing: 0;
        }}
        main {{ padding: 24px 0 48px; }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 12px;
            margin-top: -28px;
        }}
        .metric {{
            min-height: 132px;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 14px 34px rgba(23, 33, 29, .08);
            transform: translateY(0);
            transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
        }}
        .metric:hover {{
            transform: translateY(-4px);
            border-color: rgba(25, 114, 120, .35);
            box-shadow: 0 20px 42px rgba(23, 33, 29, .13);
        }}
        .metric span, .metric small {{
            display: block;
            color: var(--muted);
            font-size: 13px;
        }}
        .metric strong {{
            display: block;
            margin: 10px 0 6px;
            color: var(--accent-3);
            font-size: 30px;
            line-height: 1;
        }}
        .section-title {{
            margin: 32px 0 14px;
            font-size: 24px;
        }}
        .insights {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }}
        .insight, .panel, .chart-card {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
        }}
        .insight {{
            border-left: 5px solid var(--accent);
            min-height: 116px;
            transition: transform .2s ease, background .2s ease;
        }}
        .insight:hover {{
            transform: translateY(-2px);
            background: #fffdf8;
        }}
        .insight-toggle {{
            all: unset;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            width: 100%;
            padding: 18px;
            cursor: pointer;
            color: var(--accent-3);
            font: inherit;
            font-weight: 750;
        }}
        .insight-icon {{
            flex: none;
            color: var(--accent);
            font-size: 20px;
            line-height: 1;
            transition: transform .25s ease;
        }}
        .insight-toggle[aria-expanded="true"] .insight-icon {{
            transform: rotate(45deg);
        }}
        .insight-panel {{
            display: grid;
            grid-template-rows: 0fr;
            transition: grid-template-rows .25s ease;
        }}
        .insight-panel.is-open {{
            grid-template-rows: 1fr;
        }}
        .insight-panel-inner {{
            overflow: hidden;
        }}
        .insight-panel-inner p {{
            margin: 0 18px 18px;
            color: var(--muted);
        }}
        .chart-controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: -4px 0 14px;
        }}
        .chart-controls button {{
            border: 1px solid var(--line);
            border-radius: 999px;
            background: var(--panel);
            color: var(--accent-3);
            cursor: pointer;
            padding: 8px 14px;
            font: inherit;
            font-size: 14px;
            transition: background .18s ease, color .18s ease, border-color .18s ease;
        }}
        .chart-controls button.active,
        .chart-controls button:hover {{
            background: var(--accent);
            border-color: var(--accent);
            color: #fff;
        }}
        .charts {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }}
        .chart-card {{
            padding: 14px;
            opacity: 1;
            transform: scale(1);
            transition: opacity .2s ease, transform .2s ease, box-shadow .2s ease;
        }}
        .chart-card:hover {{
            box-shadow: 0 14px 28px rgba(23, 33, 29, .09);
        }}
        .chart-card.is-hidden {{
            display: none;
        }}
        .chart-card h3, .panel h2 {{
            margin: 0 0 12px;
            font-size: 17px;
        }}
        .chart-card .js-plotly-plot, .chart-card .plotly-graph-div {{
            width: 100% !important;
            border-radius: 6px;
        }}
        .tables {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
        }}
        .panel {{ padding: 16px; overflow: auto; }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .data-table th {{
            text-align: left;
            color: var(--accent-3);
            background: var(--soft);
        }}
        .data-table th, .data-table td {{
            padding: 9px 8px;
            border-bottom: 1px solid var(--line);
        }}
        footer {{
            color: var(--muted);
            padding-top: 10px;
            font-size: 13px;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        @media (max-width: 980px) {{
            .metrics {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
            .insights, .tables {{ grid-template-columns: 1fr; }}
        }}
        @media (max-width: 760px) {{
            header {{ padding-top: 32px; }}
            .metrics, .charts {{ grid-template-columns: 1fr; }}
            .metrics {{ margin-top: 16px; }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="wrap">
            <h1>Социальные сети и ментальное здоровье студентов</h1>
        </div>
    </header>
    <main class="wrap">
        <section class="metrics">
            {"".join(metrics)}
        </section>

        <h2 class="section-title">Ключевые наблюдения</h2>
        <section class="insights">
            {"".join(
                f'''<article class="insight">
                    <button class="insight-toggle" type="button" aria-expanded="false">
                        <span>{escape(title)}</span>
                        <span class="insight-icon" aria-hidden="true">+</span>
                    </button>
                    <div class="insight-panel">
                        <div class="insight-panel-inner"><p>{escape(detail)}</p></div>
                    </div>
                </article>'''
                for title, detail in insights
            )}
        </section>

        <h2 class="section-title">Графики</h2>
        <nav class="chart-controls" aria-label="Фильтр графиков">
            <button class="active" type="button" data-filter="all">Все</button>
            <button type="button" data-filter="distribution">Распределения</button>
            <button type="button" data-filter="relationship">Связи</button>
            <button type="button" data-filter="correlation">Корреляции</button>
        </nav>
        <section class="charts">
            {"".join(figure_cards)}
        </section>

        <h2 class="section-title">Распределения</h2>
        <section class="tables">
            {"".join(table_sections)}
        </section>

        <footer>Источник данных: {escape(str(csv_path.relative_to(PROJECT_ROOT)))}.</footer>
    </main>
    <script>
        const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

        if (!prefersReducedMotion) {{
            document.querySelectorAll(".metric strong[data-target]").forEach((node) => {{
                const raw = node.dataset.target;
                const target = Number(raw);
                if (!Number.isFinite(target)) return;

                const suffix = node.dataset.suffix || "";
                const decimals = raw.includes(".") ? raw.split(".")[1].length : 0;
                const start = performance.now();
                const duration = 900;

                function tick(now) {{
                    const progress = Math.min((now - start) / duration, 1);
                    const eased = 1 - Math.pow(1 - progress, 3);
                    const value = target * eased;
                    node.textContent = value.toLocaleString("ru-RU", {{
                        minimumFractionDigits: decimals,
                        maximumFractionDigits: decimals,
                    }}) + suffix;
                    if (progress < 1) requestAnimationFrame(tick);
                }}

                requestAnimationFrame(tick);
            }});
        }}

        document.querySelectorAll(".insight-toggle").forEach((button) => {{
            button.addEventListener("click", () => {{
                const expanded = button.getAttribute("aria-expanded") === "true";
                button.setAttribute("aria-expanded", String(!expanded));
                button.nextElementSibling.classList.toggle("is-open", !expanded);
            }});
        }});

        document.querySelectorAll(".chart-controls button").forEach((button) => {{
            button.addEventListener("click", () => {{
                const filter = button.dataset.filter;
                document.querySelectorAll(".chart-controls button").forEach((item) => item.classList.remove("active"));
                button.classList.add("active");
                document.querySelectorAll(".chart-card").forEach((card) => {{
                    card.classList.toggle("is-hidden", filter !== "all" && card.dataset.kind !== filter);
                }});
            }});
        }});
    </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path

def run_analysis(input_path: Path | None = None) -> Path:
    csv_path = find_csv(input_path)
    raw_df = read_dataset(csv_path)
    columns = resolve_columns(raw_df.columns)
    df = clean_dataset(raw_df, columns)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_path = PROCESSED_DIR / "cleaned_student_social_media.csv"
    df.to_csv(cleaned_path, index=False)

    tables = save_summary_tables(df, columns)
    figures = save_figures(df, columns)
    dashboard_path = build_html_dashboard(df, csv_path, columns, figures)

    print(f"Загружен файл: {csv_path}")
    print(f"Очищенный датасет: {cleaned_path}")
    print(f"Дашборд: {dashboard_path}")

    return dashboard_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Анализ данных о соцсетях и ментальном здоровье студентов.")
    parser.add_argument("--input", type=Path, default=None, help="Путь к CSV-файлу или папке с CSV-файлами.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(args.input)


if __name__ == "__main__":
    main()
