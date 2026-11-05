from pydantic_ai import Agent, ModelSettings, PromptedOutput, UsageLimits
from pydantic import BaseModel, Field
from my_solution.config import BASE_INSTRUCTIONS
from pydantic_ai.models import infer_model
from my_solution.config import LLM_PROVIDER, MODEL
from my_solution.load_data import create_operations_data
from pathlib import Path
from my_solution.config import MAX_AGENT_STEPS, MAX_TOOL_CALLS, MAX_RETRIES
from typing import Literal


class AgentOutput(BaseModel):
    """Structured result returned by the operations assistant."""

    answer: str = Field(description="The final answer to the request created either by draft_ticket_update() or draft_escalation().")
    status: Literal["answered", "escalated", "abstained"] = Field(description="The status of the ticket after the agent's (your) action. Can be either 'answered' (if client-side response was generated), 'escalated' (if human review is requireed), or 'abstained' (if unable or unwilling to proceed).")
    source_ids: list[str] = Field(default_factory=list, description="List of source IDs used to answer the request (if any).")
    review_required: bool = Field(default=False, description="True if the ticket requires human review.")
    abstain_reason: str | None = Field(default=None, description="If the status is 'abstained', this field should contain a brief explanation of why the agent (you) is unable or unwilling to answer the request.")


def usage_limits() -> UsageLimits:
    """Return the usage limits for the operations assistant."""
    
    return UsageLimits(request_limit=MAX_AGENT_STEPS, tool_calls_limit=MAX_TOOL_CALLS)


def build_agent(project_root: Path = Path.cwd()) -> Agent[None, AgentOutput]:
    """Builds an agent that can function as an operations assistant."""

    model = infer_model(f"{LLM_PROVIDER}:{MODEL}")
    operations_data = create_operations_data(project_root)
    model_settings = ModelSettings(timeout=30, max_tokens=2048, max_retries=MAX_RETRIES)
    
    agent = Agent[None, AgentOutput](
        model=model,
        output_type=AgentOutput,
        deps_type=type(None),
        instructions=BASE_INSTRUCTIONS,
        model_settings=model_settings,
        retries=1,
    )
    
    agent.tool_plain(operations_data.get_ticket)
    agent.tool_plain(operations_data.get_customer_context)
    agent.tool_plain(operations_data.search_policy)
    agent.tool_plain(operations_data.check_availability)
    agent.tool_plain(operations_data.draft_ticket_update)
    agent.tool_plain(operations_data.draft_escalation)
    agent.tool_plain(operations_data.access_memory)
    
    return agent