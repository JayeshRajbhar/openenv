from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from env.environment import DataCleaningEnv
from env.models import Action

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_ID = os.getenv("TASK_ID", "task1_easy")
MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))
ENV_NAME = "data-cleaning-benchmark"

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are a data cleaning agent. Analyse the observation and choose ONE cleaning action.

Available action types and required fields:
  fill_missing       -> column (str), strategy (mean|median|mode|constant), value (if constant)
  standardize_values -> column (str), mapping (dict old->new)
  remove_duplicates  -> (no extra fields)
  remove_row         -> row_id (int from _row_id column in preview)
  convert_type       -> column (str), target_type (float|int|str|datetime)
  clip_outliers      -> column (str), lower (float|null), upper (float|null)
  submit             -> (no extra fields; use when dataset is clean)

Rules:
- Respond with a SINGLE valid JSON object and NOTHING else.
- No markdown fences, no explanation.
- When no issues remain, always respond with: {"type": "submit"}

Examples:
{"type": "remove_duplicates"}
{"type": "fill_missing", "column": "age", "strategy": "median"}
{"type": "standardize_values", "column": "country", "mapping": {"USA": "United States", "US": "United States", "UK": "United Kingdom", "CAN": "Canada", "australia": "Australia", "AUS": "Australia"}}
{"type": "convert_type", "column": "date", "target_type": "datetime"}
{"type": "convert_type", "column": "price", "target_type": "float"}
{"type": "clip_outliers", "column": "session_duration", "lower": 0.0, "upper": 1000.0}
{"type": "submit"}
"""


def get_action(obs_dict: dict, history: list[dict]) -> dict:
    user_msg = {
        "role": "user",
        "content": (
            "Current observation:\n" + json.dumps(obs_dict, indent=2, default=str) + "\n\nNext action (JSON only):"
        ),
    }
    history.append(user_msg)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        max_tokens=256,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": raw})

    clean = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"type": "submit"}


def run_inference() -> None:
    env = DataCleaningEnv()
    rewards: list[float] = []
    history: list[dict] = []
    step = 0
    done = False
    success = False

    print(f"[START] task={TASK_ID} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        obs = env.reset(task_id=TASK_ID)

        while not done and step < MAX_STEPS:
            try:
                action_dict = get_action(obs.model_dump(), history)
                action = Action(**action_dict)
            except Exception:
                action_dict = {"type": "submit"}
                action = Action(type="submit")

            result = env.step(action)
            obs = result.observation
            done = result.done
            reward = result.reward
            error = result.info.get("error")

            rewards.append(reward)
            step += 1

            action_str = json.dumps(action_dict, separators=(",", ":"), default=str)
            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward:.2f} done={'true' if done else 'false'} "
                f"error={error if error else 'null'}",
                flush=True,
            )

        if not done:
            result = env.step(Action(type="submit"))
            rewards.append(result.reward)
            step += 1
            print(
                f"[STEP] step={step} action={{\"type\":\"submit\"}} "
                f"reward={result.reward:.2f} done=true error={result.info.get('error') or 'null'}",
                flush=True,
            )

        success = bool(env.final_score >= 0.5)
    except Exception:
        success = False
    finally:
        try:
            if hasattr(env, "close"):
                env.close()
        except Exception:
            pass

        rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step} score={env.final_score:.2f} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    run_inference()
