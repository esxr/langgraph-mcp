from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar
from langchain_core.runnables import RunnableConfig, ensure_config

from langgraph_mcp import prompts

@dataclass(kw_only=True)
class Configuration:
    """Configuration class for MCP routing operations."""

    embedding_model: Annotated[str, {"__template_metadata__": {"kind": "embeddings"}}] = field(
        default="openai/text-embedding-3-large",
        metadata={"description": "Embedding model to use."}
    )

    retriever_provider: Annotated[Literal["milvus"], {"__template_metadata__": {"kind": "retriever"}}] = field(
        default="milvus",
        metadata={"description": "Vector store provider to use for routing."}
    )

    mcp_server_config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "MCP server configurations."},
    )

    routing_query_system_prompt: str = field(
        default=prompts.ROUTING_QUERY_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for processing and refining routing queries."
        },
    )

    routing_query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The language model used for processing and refining routing queries. Should be in the form: provider/model-name."
        },
    )

    routing_response_system_prompt: str = field(
        default=prompts.ROUTING_RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating routing response."},
    )

    routing_response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The language model used for generating routing responses. Should be in the form: provider/model-name."
        },
    )

    mcp_orchestrator_system_prompt: str = field(
        default=prompts.MCP_ORCHESTRATOR_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for MCP server orchestration."},
    )

    mcp_orchestrator_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The language model used for MCP server orchestration. Should be in the form: provider/model-name."
        },
    )
    
    tool_refiner_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The language model used for rebinding the tool call. Should be in the form: provider/model-name."
        },
    )

    tool_refiner_prompt: str = field(
        default=prompts.TOOL_REFINER_PROMPT,
        metadata={"description": "The system prompt used for tool refining orchestration."},
    )
    
    planner_system_prompt: str = field(
        default=prompts.PLANNER_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for planning."},
    )

    planner_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The language model used for planning. Should be in the form: provider/model-name."
        },
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an Configuration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of Configuration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
    
    def get_mcp_server_descriptions(self) -> list[tuple[str, str]]:
        """Get a list of descriptions of the MCP servers in the specified configuration."""
        return [
            (server_name, server_config['description'])
            for server_name, server_config in self.mcp_server_config["mcpServers"].items()
        ]

T = TypeVar("T", bound=Configuration)