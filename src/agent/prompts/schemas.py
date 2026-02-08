"""Pydantic output schemas for structured LLM responses."""

from typing import List, Literal

from pydantic import BaseModel, Field


class PlanOutput(BaseModel):
    """Output schema for the planner node."""

    steps: List[str] = Field(description="Research steps to execute")


class RefineDecision(BaseModel):
    """Output schema for the refinery node."""

    decision: Literal["CONTINUE", "FINISH"] = Field(
        description="Whether to continue research or finish"
    )
    reason: str = Field(description="Brief explanation for the decision")
