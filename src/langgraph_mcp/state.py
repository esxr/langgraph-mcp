"""State management for the router graph.

This module defines the state structures and reduction functions used in the
router graph.

"""

from dataclasses import dataclass, field
from typing import Annotated, Sequence, Optional
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

#########################  Router Builder State  ###############################


@dataclass(kw_only=True)
class BuilderState:
    status: str = field(
        default="refresh",
        metadata={
            "description": "The status of the builder state.",
        },
    )


#############################  Router State  ###################################


# Optional, the InputState is a restricted version of the State that is used to
# define a narrower interface to the outside world vs. what is maintained
# internally.
@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained internally.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages]
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    
        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )
    
    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""


# This is the primary state of your agent, where you can store any information


def add_queries(existing: Sequence[str], new: Sequence[str]) -> Sequence[str]:
    """Combine existing queries with new queries.

    Args:
        existing (Sequence[str]): The current list of queries in the state.
        new (Sequence[str]): The new queries to be added.

    Returns:
        Sequence[str]: A new list containing all queries from both input sequences.
    """
    return list(existing) + list(new)


class PlanItem(BaseModel):
    """A task to be executed by the agent."""

    expert: str = field(default="")
    task: str = field(default="")
    done: bool = field(default=False)


class PlanEvaluation(BaseModel):
    """The evaluation of the current plan."""

    decision: str = field(default="")
    plan: list[PlanItem] = field(default_factory=list)
    next_task: int = field(default=0)
    clarification: Optional[str] = field(default="")


@dataclass(kw_only=True)
class State(InputState):
    """The state of your graph / agent."""

    queries: Annotated[list[str], add_queries] = field(default_factory=list)
    """A list of search queries that the agent has generated."""

    retrieved_docs: list[Document] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""

    current_mcp_server: str = field(default="")

    current_tool: dict[str, str] = field(default_factory=dict)
    current_plan: list[PlanItem] = field(default_factory=list)
    current_task: int = field(default=0)