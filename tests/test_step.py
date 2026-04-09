from env.environment import DataCleaningEnv
from env.models import Action


def test_remove_duplicates_gives_positive_reward():
    env = DataCleaningEnv()
    env.reset(task_id="task1_easy")
    result = env.step(Action(type="remove_duplicates"))
    assert result.reward > 0
    assert not result.done


def test_fill_missing_median():
    env = DataCleaningEnv()
    env.reset(task_id="task1_easy")
    result = env.step(Action(type="fill_missing", column="age", strategy="median"))
    assert result.reward >= 0
    assert env.current_df["age"].isna().sum() == 0


def test_invalid_action_penalised():
    env = DataCleaningEnv()
    env.reset(task_id="task1_easy")
    result = env.step(Action(type="fill_missing", column="nonexistent_col", strategy="mean"))
    assert result.reward < 0
    assert result.info["error"] is not None


def test_submit_ends_episode():
    env = DataCleaningEnv()
    env.reset(task_id="task1_easy")
    result = env.step(Action(type="submit"))
    assert result.done
    assert result.info["final_score"] >= 0.0


def test_step_after_done_is_no_op():
    env = DataCleaningEnv()
    env.reset(task_id="task1_easy")
    env.step(Action(type="submit"))
    result = env.step(Action(type="remove_duplicates"))
    assert result.done
    assert 0.0 < result.reward < 1.0
    assert result.reward == result.info["final_score"]


def test_convert_type_datetime():
    env = DataCleaningEnv()
    env.reset(task_id="task2_medium")
    result = env.step(Action(type="convert_type", column="date", target_type="datetime"))
    assert result.reward > 0
    sample = env.current_df["date"].dropna().iloc[0]
    import re

    assert re.match(r"\d{4}-\d{2}-\d{2}", str(sample))


def test_clip_outliers():
    env = DataCleaningEnv()
    env.reset(task_id="task3_hard")
    result = env.step(Action(type="clip_outliers", column="session_duration", lower=0.0, upper=1000.0))
    assert result.reward > 0
    assert env.current_df["session_duration"].max() <= 1000.0
