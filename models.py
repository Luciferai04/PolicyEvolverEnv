# models.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Union
from enum import Enum
import uuid


class PolicyActionType(str, Enum):
    PROPOSE_CLARIFICATION = "propose_clarification"
    PROPOSE_NEW_RULE = "propose_new_rule"
    EVOLVE_POLICY = "evolve_policy"


class ProposeClarificationAction(BaseModel):
    """Easy task: identify an ambiguous policy term and clarify it."""
    action_type: Literal[PolicyActionType.PROPOSE_CLARIFICATION] = PolicyActionType.PROPOSE_CLARIFICATION
    ambiguous_term: str = Field(description="The exact ambiguous term found in policies")
    suggested_definition: str = Field(description="A specific, actionable definition")
    affected_policy_ids: List[str] = Field(default_factory=list, description="Policy IDs this affects")
    justification: str = Field(description="Why this term is ambiguous")
    think: Optional[str] = Field(default=None, description="Chain-of-thought reasoning (earns +0.1 bonus)")


class ProposeNewRuleAction(BaseModel):
    """Medium task: detect a policy gap and propose a new rule."""
    action_type: Literal[PolicyActionType.PROPOSE_NEW_RULE] = PolicyActionType.PROPOSE_NEW_RULE
    rule_domain: str = Field(description="Domain the rule covers, e.g. 'content_moderation'")
    new_rule: str = Field(description="The new rule text — must be clear and actionable")
    scope: List[str] = Field(description="List of scenario types this rule covers")
    integration_points: List[str] = Field(default_factory=list, description="How it connects to existing policies")
    justification: str = Field(description="Why a gap exists and why this rule fills it")
    think: Optional[str] = Field(default=None, description="Chain-of-thought reasoning (earns +0.1 bonus)")


class PolicyModification(BaseModel):
    policy_id: str
    change_type: Literal["enhance", "restrict", "add", "remove"]
    new_text: str
    reason: str


class EvolveProcessAction(BaseModel):
    """Hard task: holistically evolve the policy framework."""
    action_type: Literal[PolicyActionType.EVOLVE_POLICY] = PolicyActionType.EVOLVE_POLICY
    policy_modifications: List[PolicyModification] = Field(description="Specific changes to make")
    expected_outcomes: Dict[str, float] = Field(description="Metric name → expected delta (0.0–1.0)")
    rollback_conditions: List[str] = Field(default_factory=list, description="When to revert")
    justification: str = Field(description="Comprehensive reasoning")
    think: Optional[str] = Field(default=None, description="Chain-of-thought reasoning (earns +0.1 bonus)")


from pydantic import RootModel

class Action(RootModel):
    root: Union[ProposeClarificationAction, ProposeNewRuleAction, EvolveProcessAction] = Field(..., discriminator="action_type")


class Observation(BaseModel):
    """What the agent sees after reset() or step()."""
    task_id: str
    episode_id: str
    step_count: int
    data_corpus: List[Dict] = Field(description="Scenarios/posts/actions for the agent to analyze")
    current_policies: List[Dict] = Field(description="The existing policy set")
    policy_outcomes: Optional[List[Dict]] = Field(default=None, description="Historical outcome data (hard task)")
    system_metrics: Dict[str, float] = Field(default_factory=dict)
    identified_issues: List[Dict] = Field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    info: Dict = Field(default_factory=dict)


class State(BaseModel):
    """Episode metadata — returned by state() endpoint."""
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    step_count: int = 0
    max_steps: int = 5
    current_score: float = 0.0
    best_score: float = 0.0
    actions_taken: List[str] = Field(default_factory=list)
