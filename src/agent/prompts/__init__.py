"""Prompt templates and output schemas for agent nodes.

Public API
----------
Prompts:  PLAN_PROMPT, EXECUTOR_PROMPT, REFINE_PROMPT, SYNTHESIZE_PROMPT, AGGREGATE_PROMPT
Schemas:  PlanOutput, RefineDecision
"""

from agent.prompts.loader import load_prompt
from agent.prompts.schemas import Task, PlanOutput, RefineDecision, FileSelectionOutput

PLAN_PROMPT = load_prompt("plan")
EXECUTOR_PROMPT = load_prompt("executor")
REFINE_PROMPT = load_prompt("refine")
SYNTHESIZE_PROMPT = load_prompt("synthesize")
AGGREGATE_PROMPT = load_prompt("aggregate")
FILE_SELECTION_PROMPT = load_prompt("file_selection")

__all__ = [
    "PLAN_PROMPT",
    "EXECUTOR_PROMPT",
    "REFINE_PROMPT",
    "SYNTHESIZE_PROMPT",
    "AGGREGATE_PROMPT",
    "FILE_SELECTION_PROMPT",
    "Task",
    "PlanOutput",
    "RefineDecision",
    "FileSelectionOutput",
]
