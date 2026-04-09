from __future__ import annotations

import math

import pandas as pd


def _strict_score(value: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.01

    # Guard against NaN/inf so every grader output is always in (0, 1).
    if not math.isfinite(score):
        return 0.01

    return round(min(max(score, 0.01), 0.99), 4)


def grade_task1(df: pd.DataFrame) -> float:
    score = 0.0

    if df.duplicated().sum() == 0:
        score += 0.25

    if "email" in df.columns and df["email"].isna().sum() == 0:
        score += 0.25

    if "age" in df.columns and df["age"].isna().sum() == 0:
        score += 0.25

    valid_countries = {"United States", "United Kingdom", "Canada", "Australia"}
    if "country" in df.columns:
        non_null = df["country"].dropna()
        if len(non_null) == 0:
            pass
        elif set(non_null.unique()).issubset(valid_countries):
            score += 0.25
        else:
            valid_n = non_null.isin(valid_countries).sum()
            score += 0.25 * (valid_n / len(non_null))

    return _strict_score(score)


def grade_task2(df: pd.DataFrame) -> float:
    score = 0.0
    n = len(df)
    if n == 0:
        return 0.01

    if "date" in df.columns:
        pattern = r"^\d{4}-\d{2}-\d{2}$"
        valid = df["date"].astype(str).str.match(pattern).sum()
        score += 0.25 * (valid / n)

    if "price" in df.columns:
        numeric = pd.to_numeric(df["price"], errors="coerce")
        non_null = numeric.notna().sum()
        score += 0.25 * (non_null / n)

    valid_cats = {"Electronics", "Furniture"}
    if "category" in df.columns:
        non_null_cats = df["category"].dropna()
        if len(non_null_cats) > 0:
            valid_n = non_null_cats.isin(valid_cats).sum()
            score += 0.25 * (valid_n / len(non_null_cats))

    key_cols = [c for c in ["price", "category", "quantity"] if c in df.columns]
    if key_cols:
        total_cells = n * len(key_cols)
        missing = sum(int(df[c].isna().sum()) for c in key_cols)
        score += 0.25 * (1.0 - missing / total_cells)

    return _strict_score(score)


def grade_task3(df: pd.DataFrame) -> float:
    score = 0.0
    n = len(df)
    if n == 0:
        return 0.01

    if "user_id" in df.columns:
        dup = df["user_id"].duplicated().sum()
        if dup == 0:
            score += 0.34
        else:
            score += 0.34 * (1.0 - dup / n)

    if "session_duration" in df.columns:
        max_dur = df["session_duration"].dropna().max() if n > 0 else 0
        if max_dur <= 1000:
            score += 0.33
        elif max_dur <= 5000:
            score += 0.15

    if "bounce_rate" in df.columns:
        valid_br = ((df["bounce_rate"] >= 0) & (df["bounce_rate"] <= 1)).sum()
        score += 0.165 * (valid_br / n)

    if "page_views" in df.columns and df["page_views"].isna().sum() == 0:
        score += 0.165

    return _strict_score(score)


def grade_task(task_id: str, df: pd.DataFrame) -> float:
    fn = TASK_GRADERS.get(task_id)
    if fn is None:
        return 0.01
    return fn(df)


def grade_task1_easy(df: pd.DataFrame) -> float:
    return grade_task1(df)


def grade_task2_medium(df: pd.DataFrame) -> float:
    return grade_task2(df)


def grade_task3_hard(df: pd.DataFrame) -> float:
    return grade_task3(df)


TASK_GRADERS = {
    "task1_easy": grade_task1_easy,
    "task2_medium": grade_task2_medium,
    "task3_hard": grade_task3_hard,
}
