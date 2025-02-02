import json
from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import cast

from langgraph_mcp.configuration import Configuration
from langgraph_mcp import mcp_wrapper as mcp
from langgraph_mcp.retriever import make_retriever
from langgraph_mcp.state import InputState, State, PlanEvaluation
from langgraph_mcp.utils.utils import get_message_text, load_chat_model, format_docs

NOTHING_RELEVANT = "No MCP server with an appropriate tool to address current context"  # When available MCP servers seem to be irrelevant for the query
IDK_RESPONSE = "No appropriate tool available."  # Default response where the current MCP Server can't help
OTHER_SERVERS_MORE_RELEVANT = "Other servers are more relevant."  # Default response when other servers are more relevant than the currnet one
AMBIGUITY_PREFIX = (
    "Ambiguity:"  # Prefix to indicate ambiguity when asking the user for clarification
)
PROCEED = "SUCCESS: (Let's proceed to the next task!)"  # Default response when the current task and server don't match

##################  MCP Server Router: Sub-graph Components  ###################


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


class OrchestratorOutput(BaseModel):
    """
    The output of the orchestrator. Containes the actual output text, and the finished flag.
    If the finished flag is True, the output is complete and can be returned to the user.
    If the finished flag is False, something else must be done and the orchestrator should be called again.
    """

    text: str
    finished: bool


async def planner(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Orchestrates MCP server processing."""
    # Fetch mcp server config
    configuration = Configuration.from_runnable_config(config)
    mcp_servers = configuration.mcp_server_config["mcpServers"]

    # Prepare the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.planner_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.planner_model)
    current_plan_str = str(state.current_plan)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "experts": configuration.get_mcp_server_descriptions(),
            "current_plan": current_plan_str,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )

    # Bind tools to model and invoke
    response = await model.with_structured_output(PlanEvaluation).ainvoke(
        message_value, config
    )

    return {"current_plan": response.plan, "current_task": response.next_task}


def decide_orchestration_or_not(state: State) -> str:
    """Decide whether to route to MCP server processing or not"""
    # If the current plan has ended, return to user
    if state.current_plan == []:
        return END
    return "mcp_orchestrator"
    


##################  MCP Server Router: Sub-graph Components  ###################


async def mcp_orchestrator(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Orchestrates MCP server processing."""
    # Fetch the current MCP server from state
    server_name = state.current_plan[state.current_task].expert

    # Fetch mcp server config
    configuration = Configuration.from_runnable_config(config)
    mcp_servers = configuration.mcp_server_config["mcpServers"]
    server_config = mcp_servers[server_name]

    # Fetch tools from the MCP server
    tools = []
    args = (
        server_config["args"][1:]
        if server_config["args"][0] == "-y"
        else server_config["args"]
    )

    # TODO: refactor
    # Separate integration for openapi-mcp-server@1.1.0
    if args[0] == "openapi-mcp-server@1.1.0":
        openapi_path = args[1]

        # TODO: refactor
        # Get the openapi file as a json
        with open(openapi_path, "r") as file:
            openapi_spec = json.load(file)  # Converts JSON to a Python dictionary

        # convert the spec to openai tools
        tools = await mcp.apply(
            server_name,
            server_config,
            mcp.GetOpenAPITools(openapi_spec),
        )
    else:
        tools = await mcp.apply(server_name, server_config, mcp.GetTools())

    # Prepare the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.mcp_orchestrator_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.mcp_orchestrator_model)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "idk_response": IDK_RESPONSE,
            "current_plan": str([item.model_dump() for item in state.current_plan]),
            "current_task": str(state.current_plan[state.current_task].model_dump()),
            "current_server": server_name,
            "proceed_request": PROCEED,
            "ambiguity_prefix": AMBIGUITY_PREFIX,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )

    # Bind tools to model and invoke
    response = await model.bind_tools(tools).ainvoke(message_value, config)

    # TODO: use prompt to do this in the future
    # If response has multiple tool_calls, select just the first
    if response.__class__ == AIMessage and response.tool_calls:
        response.tool_calls = [response.tool_calls[0]]

    # If the model has an AI response with a tool_call, find the selected tool
    current_tool = None
    if args[0] == "openapi-mcp-server@1.1.0":
        if response.__class__ == AIMessage and response.tool_calls:
            current_tool = next(
                (
                    tool
                    for tool in tools
                    if tool["name"] == response.tool_calls[0].get("name")
                ),
                None,
            )

    # Handle IDK
    if response.content == IDK_RESPONSE:
        """model doesn't know how to proceed"""
        if state.messages[-1].__class__ != ToolMessage:
            """and this is not immediately after a tool call response"""
            # let's setup for routing again
            return

    return {"messages": [response], "current_tool": current_tool}


async def refine_tool_call(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the MCP server tool."""

    if state.current_tool == None:
        return

    # Fetch the current MCP server from state
    server_name = state.current_plan[state.current_task].expert

    # Fetch mcp server config
    configuration = Configuration.from_runnable_config(config)
    mcp_servers = configuration.mcp_server_config["mcpServers"]
    server_config = mcp_servers[server_name]

    # Get the tool info
    tool_info = state.current_tool.get("metadata", {}).get("tool_info", {})

    # Bind the tool call to the model
    # Prepare the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.tool_refiner_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.tool_refiner_model)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages[:-1],
            "tool_info": str(tool_info),
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )

    # get the last response id
    last_msg_id = state.messages[-1].id

    # Bind tools to model and invoke
    response = await model.bind_tools([state.current_tool]).ainvoke(
        message_value, config
    )
    response.id = last_msg_id

    return {"messages": [response], "current_tool": None}


async def mcp_tool_call(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the MCP server tool."""
    # Fetch the current MCP server from state
    server_name = state.current_plan[state.current_task].expert

    # Fetch mcp server config
    configuration = Configuration.from_runnable_config(config)
    mcp_servers = configuration.mcp_server_config["mcpServers"]
    server_config = mcp_servers[server_name]

    # Execute MCP server Tool
    tool_call = state.messages[-1].tool_calls[0]
    try:
        tool_output = await mcp.apply(
            server_name,
            server_config,
            mcp.RunTool(tool_call["name"], **tool_call["args"]),
        )
    except Exception as e:
        tool_output = f"Error: {e}"
    return {
        "messages": [ToolMessage(content=tool_output, tool_call_id=tool_call["id"])]
    }


def route_tools(state: State) -> str:
    """
    Route to the mcp_tool_call if last message has tool calls.
    Otherwise, route to the END.
    """
    last_message = state.messages[-1]

    if last_message.model_dump().get("tool_calls"):  # suggests tool calls
        return "refine_tool_call"
    if AMBIGUITY_PREFIX in last_message.content:
        return END
    # TODO: refactor if redundant
    if PROCEED in last_message.content:
        return "planner"

    return "planner"


#############################  Subgraph decider  ###############################
def decide_subgraph(state: State) -> str:
    """
    Route to MCP Server Router sub-graph if there is no state.current_mcp_server
    else route to MCP Server processing sub-graph.
    """
    if state.current_mcp_server:
        return "mcp_orchestrator"
    return "planner"


##################################  Wiring  ####################################


builder = StateGraph(State, input=InputState, config_schema=Configuration)
builder.add_node(planner)
builder.add_node(mcp_orchestrator)
builder.add_node(refine_tool_call)
builder.add_node(mcp_tool_call)

builder.add_edge(START, "planner")
builder.add_conditional_edges(
    "planner",
    decide_orchestration_or_not,
    {
        "mcp_orchestrator": "mcp_orchestrator",
        END: END,
    },
)
builder.add_conditional_edges(
    "mcp_orchestrator",
    route_tools,
    {
        "mcp_tool_call": "mcp_tool_call",
        "planner": "planner",
        "refine_tool_call": "refine_tool_call",
        END: END,
    },
)
builder.add_edge("refine_tool_call", "mcp_tool_call")
builder.add_edge("mcp_tool_call", "mcp_orchestrator")
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
graph.name = "AssistantGraph"
