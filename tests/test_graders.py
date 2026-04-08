import pandas as pd

from env.graders import grade_task, grade_task1, grade_task2, grade_task3
from env.tasks import get_task


def test_grade_task1_dirty_is_low():
    cfg = get_task("task1_easy")
    score = grade_task1(cfg["dirty_df"])
    assert 0.0 < score <= 0.5


def test_grade_task1_perfect_is_bounded():
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol"],
            "email": ["a@x.com", "b@x.com", "c@x.com"],
            "country": ["United States", "United Kingdom", "Australia"],
            "age": [28.0, 35.0, 42.0],
        }
    )
    score = grade_task1(df)
    assert 0.99 == score


def test_grade_task1_partial():
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob"],
            "email": ["a@x.com", "b@x.com"],
            "country": ["USA", "UK"],
            "age": [28.0, 35.0],
        }
    )
    score = grade_task1(df)
    assert 0.4 < score < 0.99


def test_grade_task2_score_range():
    cfg = get_task("task2_medium")
    score = grade_task2(cfg["dirty_df"])
    assert 0.0 < score < 1.0


def test_grade_task3_score_range():
    cfg = get_task("task3_hard")
    score = grade_task3(cfg["dirty_df"])
    assert 0.0 < score < 1.0


def test_grade_task_dispatcher():
    for tid in ["task1_easy", "task2_medium", "task3_hard"]:
        cfg = get_task(tid)
        s = grade_task(tid, cfg["dirty_df"])
        assert 0.0 < s < 1.0


def test_grader_not_constant():
    cfg = get_task("task1_easy")
    dirty_score = grade_task1(cfg["dirty_df"])
    clean_df = pd.DataFrame(
        {
            "name": ["Alice", "Bob"],
            "email": ["a@x.com", "b@x.com"],
            "country": ["United States", "Australia"],
            "age": [28.0, 35.0],
        }
    )
    clean_score = grade_task1(clean_df)
    assert clean_score != dirty_score
