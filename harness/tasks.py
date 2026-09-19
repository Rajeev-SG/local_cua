"""Stage 2 tiny tasks with independent DOM verification.

T1 one-step: click a named visible control.
T2 input: create one named todo and commit it.
T3 short horizon: add two todos, complete one, select Active, end in expected state.

Verification reads the app's own state (todo.html exposes it on window.__state),
not the model's claim.
"""
from __future__ import annotations

from dataclasses import dataclass

from .browser import Executor, fixture_url


@dataclass
class TaskSpec:
    key: str
    instruction: str
    max_steps: int
    verify: str  # js expression -> bool


TASKS = {
    "T1": TaskSpec(
        "T1",
        "Click the 'Open settings' button.",
        max_steps=1,
        verify="() => document.getElementById('save-settings').style.display !== 'none'",
    ),
    "T2": TaskSpec(
        "T2",
        "Type 'Buy milk' into the new-todo field and commit it with the Add button.",
        max_steps=3,
        verify="() => window.__state().todos.length === 1 && "
               "window.__state().todos[0].title === 'Buy milk'",
    ),
    "T3": TaskSpec(
        "T3",
        "Add two todos: 'Buy milk' then 'Walk dog'. Then tick the checkbox for "
        "'Walk dog' to complete it. Then click the 'Active' filter. The list "
        "should then show only 'Buy milk'.",
        max_steps=8,
        verify="() => { const s = window.__state(); "
               "return s.todos.length === 2 && s.todos[0].title === 'Buy milk' && "
               "s.todos[0].done === false && s.todos[1].title === 'Walk dog' && "
               "s.todos[1].done === true && s.filter === 'active' && "
               "document.querySelectorAll('li.todo').length === 1; }",
    ),
}


def reset(ex: Executor) -> None:
    ex.goto(fixture_url("todo.html"))


def state(ex: Executor) -> dict:
    return ex.page.evaluate("() => window.__state()")


def check(ex: Executor, task: TaskSpec) -> bool:
    return bool(ex.page.evaluate(task.verify))
