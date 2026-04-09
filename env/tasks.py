from __future__ import annotations

from typing import Any, Dict

import pandas as pd


TASK1_DIRTY = [
    {"name": "Alice Johnson", "email": "alice@email.com", "country": "USA", "age": 28.0},
    {"name": "Bob Smith", "email": "bob@email.com", "country": "United States", "age": None},
    {"name": "Carol White", "email": "carol@email.com", "country": "UK", "age": 35.0},
    {"name": "Alice Johnson", "email": "alice@email.com", "country": "USA", "age": 28.0},
    {"name": "Dave Brown", "email": None, "country": "US", "age": 42.0},
    {"name": "Eve Davis", "email": "eve@email.com", "country": "United Kingdom", "age": 31.0},
    {"name": "Frank Miller", "email": "frank@email.com", "country": "Canada", "age": None},
    {"name": "Grace Wilson", "email": "grace@email.com", "country": "CAN", "age": 25.0},
    {"name": "Henry Moore", "email": "henry@email.com", "country": "australia", "age": 38.0},
    {"name": "Iris Taylor", "email": "iris@email.com", "country": "AUS", "age": 29.0},
]

TASK1_DESCRIPTION = (
    "Clean a customer dataset. Issues to fix:\n"
    "1) Remove exact duplicate rows\n"
    "2) Fill missing emails using constant 'unknown@email.com'\n"
    "3) Fill missing ages using median\n"
    "4) Standardize country names → 'United States', 'United Kingdom', 'Canada', 'Australia'"
)

TASK2_DIRTY = [
    {
        "order_id": 1,
        "date": "2023-01-15",
        "product": "Laptop",
        "category": "Electronics",
        "price": "$1200.00",
        "quantity": 2,
    },
    {
        "order_id": 2,
        "date": "02/20/2023",
        "product": "Chair",
        "category": "Furniture",
        "price": "$250.50",
        "quantity": 1,
    },
    {
        "order_id": 3,
        "date": "Mar 10, 2023",
        "product": "Headphones",
        "category": "Electronics",
        "price": "$89.99",
        "quantity": 3,
    },
    {
        "order_id": 4,
        "date": "2023-04-05",
        "product": "Desk",
        "category": "Furnitre",
        "price": "$450.00",
        "quantity": 1,
    },
    {
        "order_id": 5,
        "date": "05/12/2023",
        "product": "Monitor",
        "category": "Electronics",
        "price": "320.00",
        "quantity": 2,
    },
    {
        "order_id": 6,
        "date": "2023-06-18",
        "product": "Keyboard",
        "category": None,
        "price": "$75.00",
        "quantity": 5,
    },
    {
        "order_id": 7,
        "date": "July 22 2023",
        "product": "Mouse",
        "category": "Electronics",
        "price": "$35.00",
        "quantity": 4,
    },
    {
        "order_id": 8,
        "date": "2023-08-30",
        "product": "Bookshelf",
        "category": "Furniture",
        "price": None,
        "quantity": 1,
    },
    {
        "order_id": 9,
        "date": "09-14-2023",
        "product": "Webcam",
        "category": "ELECTRONICS",
        "price": "$65.00",
        "quantity": 2,
    },
    {
        "order_id": 10,
        "date": "2023-10-01",
        "product": "Lamp",
        "category": "Furniture",
        "price": "$45.00",
        "quantity": 3,
    },
    {
        "order_id": 11,
        "date": "11/15/2023",
        "product": "Tablet",
        "category": "Electronix",
        "price": "$599.00",
        "quantity": 1,
    },
    {
        "order_id": 12,
        "date": "2023-12-20",
        "product": "Sofa",
        "category": "Furniture",
        "price": "$1100.00",
        "quantity": 1,
    },
]

TASK2_DESCRIPTION = (
    "Clean an e-commerce orders dataset. Issues to fix:\n"
    "1) Normalise all dates to YYYY-MM-DD format using convert_type(date, datetime)\n"
    "2) Convert price column to float (strips $ signs automatically)\n"
    "3) Standardise category typos: 'Furnitre'→'Furniture', 'ELECTRONICS'→'Electronics', 'Electronix'→'Electronics'\n"
    "4) Fill missing price with median; fill or remove missing category rows"
)

TASK3_DIRTY = [
    {"user_id": "U001", "name": "Alice Johnson", "page_views": 45, "session_duration": 320, "bounce_rate": 0.25},
    {"user_id": "U001", "name": "Alice J.", "page_views": 45, "session_duration": 315, "bounce_rate": 0.25},
    {"user_id": "U002", "name": "Bob Smith", "page_views": 12, "session_duration": 85000, "bounce_rate": 0.80},
    {"user_id": "U003", "name": "Carol White", "page_views": 67, "session_duration": 450, "bounce_rate": 0.15},
    {"user_id": "U004", "name": "Dave Brown", "page_views": 23, "session_duration": 190, "bounce_rate": 0.55},
    {"user_id": "U005", "name": "Eve Davis", "page_views": 89, "session_duration": 95000, "bounce_rate": 0.10},
    {"user_id": "U003", "name": "Carol White", "page_views": 67, "session_duration": 450, "bounce_rate": 0.15},
    {"user_id": "U006", "name": "Frank Miller", "page_views": None, "session_duration": 280, "bounce_rate": 0.45},
    {"user_id": "U007", "name": "Grace Wilson", "page_views": 34, "session_duration": 360, "bounce_rate": 1.50},
    {"user_id": "U008", "name": "Henry Moore", "page_views": 56, "session_duration": 420, "bounce_rate": 0.35},
    {"user_id": "U009", "name": "Iris Taylor", "page_views": 78, "session_duration": 78000, "bounce_rate": 0.20},
    {"user_id": "U010", "name": "Jack Wilson", "page_views": 19, "session_duration": 150, "bounce_rate": 0.70},
]

TASK3_DESCRIPTION = (
    "Clean a web analytics dataset. Issues to fix:\n"
    "1) Remove duplicate user_ids (exact + near-duplicates — keep first occurrence)\n"
    "2) Clip session_duration outliers to max 1000 seconds\n"
    "3) Clip bounce_rate to valid range [0.0, 1.0]\n"
    "4) Fill missing page_views with median"
)

TASK_GRADER_ENTRYPOINTS = {
    "task1_easy": "env.graders:grade_task1_easy",
    "task2_medium": "env.graders:grade_task2_medium",
    "task3_hard": "env.graders:grade_task3_hard",
}


def get_task(task_id: str) -> Dict[str, Any]:
    registry = {
        "task1_easy": {
            "description": TASK1_DESCRIPTION,
            "dirty_df": pd.DataFrame(TASK1_DIRTY),
            "difficulty": "easy",
            "grader": TASK_GRADER_ENTRYPOINTS["task1_easy"],
            "grader_fn": TASK_GRADER_ENTRYPOINTS["task1_easy"],
            "grader_path": TASK_GRADER_ENTRYPOINTS["task1_easy"],
        },
        "task2_medium": {
            "description": TASK2_DESCRIPTION,
            "dirty_df": pd.DataFrame(TASK2_DIRTY),
            "difficulty": "medium",
            "grader": TASK_GRADER_ENTRYPOINTS["task2_medium"],
            "grader_fn": TASK_GRADER_ENTRYPOINTS["task2_medium"],
            "grader_path": TASK_GRADER_ENTRYPOINTS["task2_medium"],
        },
        "task3_hard": {
            "description": TASK3_DESCRIPTION,
            "dirty_df": pd.DataFrame(TASK3_DIRTY),
            "difficulty": "hard",
            "grader": TASK_GRADER_ENTRYPOINTS["task3_hard"],
            "grader_fn": TASK_GRADER_ENTRYPOINTS["task3_hard"],
            "grader_path": TASK_GRADER_ENTRYPOINTS["task3_hard"],
        },
    }
    if task_id not in registry:
        raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(registry)}")
    cfg = registry[task_id]
    cfg["dirty_df"] = cfg["dirty_df"].copy()
    return cfg


TASK_IDS = ["task1_easy", "task2_medium", "task3_hard"]


def list_tasks() -> list[dict[str, Any]]:
    return [
        {
            "id": "task1_easy",
            "difficulty": "easy",
            "max_steps": 20,
            "grader": TASK_GRADER_ENTRYPOINTS["task1_easy"],
            "grader_fn": TASK_GRADER_ENTRYPOINTS["task1_easy"],
            "grader_path": TASK_GRADER_ENTRYPOINTS["task1_easy"],
        },
        {
            "id": "task2_medium",
            "difficulty": "medium",
            "max_steps": 20,
            "grader": TASK_GRADER_ENTRYPOINTS["task2_medium"],
            "grader_fn": TASK_GRADER_ENTRYPOINTS["task2_medium"],
            "grader_path": TASK_GRADER_ENTRYPOINTS["task2_medium"],
        },
        {
            "id": "task3_hard",
            "difficulty": "hard",
            "max_steps": 20,
            "grader": TASK_GRADER_ENTRYPOINTS["task3_hard"],
            "grader_fn": TASK_GRADER_ENTRYPOINTS["task3_hard"],
            "grader_path": TASK_GRADER_ENTRYPOINTS["task3_hard"],
        },
    ]
