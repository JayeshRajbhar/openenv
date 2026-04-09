from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .graders import grade_task
from .models import Action, Observation, StepResult, TablePreview
from .rewards import compute_reward
from .tasks import TASK_IDS, get_task


class DataCleaningEnv:
    MAX_STEPS: int = 20
    MIN_EPISODE_SCORE: float = 0.01
    MAX_EPISODE_SCORE: float = 0.99

    def __init__(self) -> None:
        self.task_id: Optional[str] = None
        self._task_config: Optional[dict] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.current_df: Optional[pd.DataFrame] = None
        self.step_count: int = 0
        self.cleaning_log: list = []
        self.action_history: list = []
        self.raw_cumulative_reward: float = 0.0
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.final_score: float = 0.01

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id is None:
            task_id = TASK_IDS[0]
        self.task_id = task_id
        self._task_config = get_task(task_id)
        self.original_df = self._task_config["dirty_df"].copy()
        self.current_df = self._task_config["dirty_df"].copy()
        self.step_count = 0
        self.cleaning_log = []
        self.action_history = []
        self.raw_cumulative_reward = 0.0
        self.cumulative_reward = 0.0
        self.done = False
        self.final_score = 0.01
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        if self.done:
            return StepResult(
                observation=self._build_observation(),
                reward=self.final_score,
                done=True,
                info={
                    "error": "Episode already finished",
                    "cumulative_reward": self.cumulative_reward,
                    "raw_cumulative_reward": self.raw_cumulative_reward,
                    "final_score": self.final_score,
                    "step": self.step_count,
                },
            )

        error: Optional[str] = None
        reward: float = 0.0

        if action.type == "submit":
            self.final_score = grade_task(self.task_id, self.current_df)
            reward = self.final_score
            self.cleaning_log.append(f"[SUBMIT] Final grade: {self.final_score:.4f}")
            self.done = True
        else:
            try:
                reward, log_msg = self._apply_action(action)
                self.cleaning_log.append(log_msg)
            except Exception as exc:
                error = str(exc)
                reward = -0.10
                self.cleaning_log.append(f"[ERROR] {error}")

        self.step_count += 1
        self.raw_cumulative_reward = round(self.raw_cumulative_reward + reward, 4)
        self.cumulative_reward = self._clamp_episode_score(self.raw_cumulative_reward)
        self.action_history.append(action.model_dump())

        if not self.done and self.step_count >= self.MAX_STEPS:
            self.final_score = grade_task(self.task_id, self.current_df)
            self.done = True

        return StepResult(
            observation=self._build_observation(),
            reward=round(reward, 4),
            done=self.done,
            info={
                "error": error,
                "cumulative_reward": self.cumulative_reward,
                "raw_cumulative_reward": self.raw_cumulative_reward,
                "final_score": self.final_score,
                "step": self.step_count,
            },
        )

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "raw_cumulative_reward": self.raw_cumulative_reward,
            "final_score": self.final_score,
            "done": self.done,
            "cleaning_log": self.cleaning_log,
            "action_history": self.action_history,
            "current_data": self._df_records_with_none(self.current_df) if self.current_df is not None else [],
        }

    @classmethod
    def _clamp_episode_score(cls, value: float) -> float:
        return round(min(max(value, cls.MIN_EPISODE_SCORE), cls.MAX_EPISODE_SCORE), 4)

    def _apply_action(self, action: Action) -> Tuple[float, str]:
        df = self.current_df

        if action.type == "fill_missing":
            col = self._require_column(action.column, df)
            missing_before = int(df[col].isna().sum())
            if missing_before == 0:
                return -0.05, f"[WARN] No missing values in '{col}' — wasted step"

            if action.strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif action.strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif action.strategy == "mode":
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            elif action.strategy == "constant":
                df[col] = df[col].fillna(action.value)
            else:
                raise ValueError(f"Unknown fill strategy '{action.strategy}'")

            reward = compute_reward("fill_missing", {"filled": missing_before})
            return reward, f"Filled {missing_before} missing values in '{col}' via {action.strategy}"

        if action.type == "standardize_values":
            col = self._require_column(action.column, df)
            if not action.mapping:
                raise ValueError("'mapping' dict is required for standardize_values")
            replaced = int(df[col].isin(action.mapping.keys()).sum())
            df[col] = df[col].apply(lambda x: action.mapping.get(str(x), x) if pd.notna(x) else x)
            reward = compute_reward("standardize_values", {"replaced": replaced})
            return reward, f"Standardised {replaced} values in '{col}'"

        if action.type == "remove_duplicates":
            before = len(df)
            self.current_df = df.drop_duplicates().reset_index(drop=True)
            removed = before - len(self.current_df)
            if removed == 0:
                return -0.05, "[WARN] No exact duplicates found — wasted step"
            reward = compute_reward("remove_duplicates", {"removed": removed})
            return reward, f"Removed {removed} duplicate row(s)"

        if action.type == "remove_row":
            if action.row_id is None:
                raise ValueError("'row_id' is required for remove_row")
            if action.row_id not in df.index:
                raise ValueError(f"Row index {action.row_id} not found (valid range 0–{len(df)-1})")
            self.current_df = df.drop(index=action.row_id).reset_index(drop=True)
            reward = compute_reward("remove_row", {})
            return reward, f"Removed row at index {action.row_id}"

        if action.type == "convert_type":
            col = self._require_column(action.column, df)
            tgt = action.target_type

            if tgt == "float":
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[$,\s]", "", regex=True)
                    .replace("nan", np.nan)
                    .replace("None", np.nan)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif tgt == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif tgt == "str":
                df[col] = df[col].astype(str)
            elif tgt == "datetime":
                parsed = pd.to_datetime(df[col], errors="coerce")
                df[col] = parsed.dt.strftime("%Y-%m-%d")
            else:
                raise ValueError(f"Unknown target_type '{tgt}'")

            reward = compute_reward("convert_type", {})
            return reward, f"Converted column '{col}' → {tgt}"

        if action.type == "clip_outliers":
            col = self._require_column(action.column, df)
            if action.lower is None and action.upper is None:
                raise ValueError("At least one of 'lower' or 'upper' must be set")

            series = pd.to_numeric(df[col], errors="coerce")
            clipped = 0
            if action.lower is not None:
                clipped += int((series < action.lower).sum())
            if action.upper is not None:
                clipped += int((series > action.upper).sum())

            df[col] = series.clip(lower=action.lower, upper=action.upper)
            reward = compute_reward("clip_outliers", {"clipped": clipped})
            return reward, f"Clipped '{col}' to [{action.lower}, {action.upper}] ({clipped} value(s) affected)"

        raise ValueError(f"Unknown action type '{action.type}'")

    @staticmethod
    def _require_column(col: Optional[str], df: pd.DataFrame) -> str:
        if not col:
            raise ValueError("'column' field is required for this action")
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
        return col

    @staticmethod
    def _df_records_with_none(df: pd.DataFrame) -> list[dict]:
        safe_df = df.astype(object).where(pd.notna(df), None)
        return safe_df.to_dict(orient="records")

    def _build_observation(self) -> Observation:
        df = self.current_df
        issues: list = []

        if df is not None:
            for col in df.columns:
                miss = int(df[col].isna().sum())
                if miss > 0:
                    issues.append(f"Column '{col}' has {miss} missing value(s)")
            dup = int(df.duplicated().sum())
            if dup > 0:
                issues.append(f"{dup} exact duplicate row(s) detected")

            head = df.head(10).copy()
            head.insert(0, "_row_id", head.index.tolist())
            preview_rows = self._df_records_with_none(head)
            schema_info = {c: str(df[c].dtype) for c in df.columns}
            shape = list(df.shape)
        else:
            preview_rows, schema_info, shape = [], {}, [0, 0]

        preview = TablePreview(
            columns=["_row_id"] + (list(df.columns) if df is not None else []),
            rows=preview_rows,
            shape=shape,
        )

        return Observation(
            task_id=self.task_id or "",
            task_description=(self._task_config["description"] if self._task_config else ""),
            table_preview=preview,
            schema_info=schema_info,
            valid_actions=[
                "fill_missing",
                "standardize_values",
                "remove_duplicates",
                "remove_row",
                "convert_type",
                "clip_outliers",
                "submit",
            ],
            step=self.step_count,
            max_steps=self.MAX_STEPS,
            cleaning_log=self.cleaning_log[-6:],
            issues_detected=issues,
        )
