"""Pydantic output schemas for structured LLM responses."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A single research task in the plan."""

    goal: str = Field(description="What to find out or verify (the objective, not tool instructions)")
    success_criteria: str = Field(description="Observable condition that means this task is done")
    abort_criteria: str = Field(description="When to stop and move on (e.g. after N failed attempts)")
    suggested_tools: Optional[List[str]] = Field(
        default=None,
        description="Recommended tools if the planner can infer them from the codebase profile",
    )
    context_hint: Optional[str] = Field(
        default=None,
        description="Relevant info extracted from the codebase profile (e.g. key file paths, module names)",
    )


class PlanOutput(BaseModel):
    """Output schema for the planner node."""

    tasks: List[Task] = Field(description="Ordered list of research tasks (3-5 recommended)")


class RefineDecision(BaseModel):
    """Output schema for the refinery node."""

    decision: Literal["CONTINUE", "FINISH"] = Field(
        description="Whether to continue research or finish"
    )
    reason: str = Field(description="Brief explanation for the decision")
