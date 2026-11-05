from dataclasses import dataclass, field
from typing import Any
from pydantic_ai import AgentRunResult, ToolCallPart, ToolReturnPart
from my_solution.agent import AgentOutput



class AgentTrace(AgentOutput):
    """A record of the agent's decision-making process for a given case."""
    case_id: str
    ticket_id: str
    prompt: str
    tools: list
    
    
@dataclass(frozen=True)
class PlanStep:
    """One completed application-tool step in an agent trace."""

    tool: str
    reason: str
    arguments: dict[str, Any] = field(default_factory=dict)
    

def case_to_prompt(case: dict) -> str:
    """Converts a case dictionary to a prompt string."""
    
    prompt = f"ticket_id: {case['ticket_id']}\n"
    prompt += f"prompt: {case['prompt']}\n"
    
    return prompt


def trace_from_result(response: AgentRunResult[AgentOutput], case: dict) -> AgentTrace:
    """Record an enriched agent decision-making process for a given case."""

    tools = {
        "get_ticket",
        "get_customer_context",
        "search_policy",
        "check_availability",
        "draft_ticket_update",
        "draft_escalation"
    }
    messages = response.new_messages()
    calls = {
        part.tool_call_id: part.args_as_dict()
        for message in messages
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name in tools
    }
    steps = [
        PlanStep(
            part.tool_name, "Tool returned successfully.", calls[part.tool_call_id]
        )
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
        and part.tool_name in tools
        and part.outcome == "success"
    ]
    return AgentTrace(
        case_id = case['case_id'],
        ticket_id = case['ticket_id'],
        prompt = case['prompt'],
        tools = steps,
        answer = response.output.answer,
        status = response.output.status,
        source_ids = response.output.source_ids,
        review_required = response.output.review_required,
        abstain_reason = response.output.abstain_reason
    )