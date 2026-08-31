"""LangGraph topology — ported from 2025/kepsoar/graph/build_graph.py.

Topology preserved:
    SCRIPT_GEN: script_gen → caution_eval → END
    REPORT_GEN: (caution_eval if is_script_changed) → report_gen → END

Differences from 2025:
- Uses the new `AgentState` TypedDict (telemetry/event windows instead of
  network 5-tuples).
- No webhook side effects. The harness emits actions and persists reports.
"""
from __future__ import annotations

try:
    from langgraph.graph import StateGraph, START, END  # type: ignore
except ImportError:  # pragma: no cover — keep import-safe in CI without langgraph
    StateGraph = START = END = None  # type: ignore

from .agents import caution_eval_agent, report_gen_agent, script_gen_agent
from .states import AgentState, OperationMode


def build():
    if StateGraph is None:
        raise RuntimeError("langgraph is required to build the agent graph")

    script_builder = StateGraph(AgentState, input=AgentState, output=AgentState)
    script_builder.add_node("script_gen", script_gen_agent)
    script_builder.add_node("eval_caution_level", caution_eval_agent)
    script_builder.add_edge(START, "script_gen")
    script_builder.add_edge("script_gen", "eval_caution_level")
    script_builder.add_edge("eval_caution_level", END)
    script_sub = script_builder.compile()

    report_builder = StateGraph(AgentState, input=AgentState, output=AgentState)
    report_builder.add_node("eval_caution_level", caution_eval_agent)
    report_builder.add_node("report_gen", report_gen_agent)
    report_builder.add_conditional_edges(
        START,
        lambda s: "eval" if s.get("is_script_changed", False) else "direct",
        path_map={"eval": "eval_caution_level", "direct": "report_gen"},
    )
    report_builder.add_edge("eval_caution_level", "report_gen")
    report_builder.add_edge("report_gen", END)
    report_sub = report_builder.compile()

    graph = StateGraph(AgentState, input=AgentState, output=AgentState)
    graph.add_node("script_path", script_sub)
    graph.add_node("report_path", report_sub)
    graph.add_conditional_edges(
        START,
        lambda s: s.get("mode", OperationMode.SCRIPT_GEN),
        path_map={
            OperationMode.SCRIPT_GEN: "script_path",
            OperationMode.REPORT_GEN: "report_path",
        },
    )
    graph.add_edge("script_path", END)
    graph.add_edge("report_path", END)
    return graph.compile()
