"""Agent loop, guardrails and the claims copilot (offline: scripted model, fake tools)."""
import json
import logging
from types import SimpleNamespace

from lighthouse_poc.agents.claims_copilot.agent import HandoffTool, build_copilot
from lighthouse_poc.agents.core.agent import Agent, AgentConfig, TurnContext
from lighthouse_poc.agents.core.guardrails import HUMAN_DECIDES, NonDecisionGuardrail, QuoteGuardrail
from lighthouse_poc.agents.core.tools import ToolResult, ToolSpec
from lighthouse_poc.config import load_profile
from lighthouse_poc.exceptions import ConfigurationError, ModelInvocationError
from lighthouse_poc.gateway import GatewayChatClient, NoAuth, ToolCall, ToolTurn
from lighthouse_poc.config import ModelRoute

LOG = logging.getLogger("test")
PASSAGE = "Flag gaps such as treating records that stop before the end of the elimination period."


class _Model:
    """Plays back scripted turns and records the messages it was sent."""

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def complete_with_tools(self, messages, tools, tags=None, tool_choice="auto"):
        self.seen.append((list(messages), [t["function"]["name"] for t in tools]))
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def _turn(text="", calls=()):
    return ToolTurn(text, [ToolCall(f"c{i}", n, json.dumps(a)) for i, (n, a) in enumerate(calls)], 10,
                    "served", 100, 20)


class _Tool:
    def __init__(self, name, result):
        self.spec = ToolSpec(name, name, {"type": "object", "properties": {}})
        self.result, self.calls = result, []

    def run(self, arguments):
        self.calls.append(arguments)
        return self.result


def _agent(turns, tools, guardrails=None, max_rounds=4):
    model = _Model(turns)
    agent = Agent(AgentConfig("t", "system", max_rounds=max_rounds), tools, lambda key: model, LOG,
                  guardrails=guardrails, price_of=lambda m, i, o: round(i * 1e-6 + o * 2e-6, 6))
    return agent, model


def test_loop_runs_tools_feeds_results_back_and_answers():
    case = _Tool("get_claim_case", ToolResult("get_claim_case", "ok", {"case_id": "A"}, "A: ok"))
    guide = _Tool("search_guidance", ToolResult("search_guidance", "ok", {"passages": [{"text": PASSAGE}]},
                                                "1 passage", data={"citations": [{"chunk_id": "k1", "doc_id": "kb-ltd", "text": PASSAGE}]}))
    agent, model = _agent([_turn(calls=[("get_claim_case", {"case_id": "A"}), ("search_guidance", {"query": "q"})]),
                           _turn(f'Missing records. "{PASSAGE}" [kb-ltd]')],
                          {"get_claim_case": case, "search_guidance": guide}, [QuoteGuardrail()])
    events = list(agent.run("Prepare A", model_key="claims-small"))
    types = [e.type for e in events]
    assert types == ["model_call", "model_call", "tool_call", "tool_result", "tool_call", "tool_result",
                     "model_call", "model_call", "guardrail", "answer", "usage", "done"]
    assert case.calls == [{"case_id": "A"}]
    second_round_messages = model.seen[1][0]
    assert second_round_messages[-1]["role"] == "tool" and second_round_messages[-3]["role"] == "assistant"
    answer = next(e for e in events if e.type == "answer").data
    assert answer["citations"][0]["doc_id"] == "kb-ltd"
    usage = next(e for e in events if e.type == "usage").data
    assert usage["model_calls"] == 2 and usage["tool_calls"] == 2 and usage["input_tokens"] == 200
    assert usage["cost_usd"] == round(200e-6 + 40 * 2e-6, 6)


def test_out_of_rounds_forces_an_answer_without_tools_and_unknown_tools_are_errors():
    agent, model = _agent([_turn(calls=[("nope", {})]), _turn("final")], {}, max_rounds=1)
    events = list(agent.run("q"))
    assert next(e for e in events if e.type == "tool_result").data["status"] == "error"
    assert model.seen[1][1] == []                                       # last round: no tools offered
    assert next(e for e in events if e.type == "answer").data["text"] == "final"


def test_model_failure_yields_error_then_usage_and_done():
    agent, _ = _agent([ModelInvocationError("down", "upstream")], {})
    events = list(agent.run("q"))
    assert [e.type for e in events][-3:] == ["error", "usage", "done"]
    assert events[-3].data["category"] == "upstream"


def test_non_decision_guardrail_removes_decision_sentences_only():
    guard = NonDecisionGuardrail()
    text, note = guard.check("Two documents are missing. The claim should be approved. Next, request the FCE.",
                             TurnContext())
    assert "approved" not in text and "Two documents are missing." in text and HUMAN_DECIDES in text
    assert note.startswith("removed 1")
    same, note = guard.check("I cannot decide whether the claim is covered; a human examiner decides.",
                             TurnContext())
    assert note is None and same.startswith("I cannot decide")


def test_quote_guardrail_flags_quotes_the_tools_did_not_return():
    ctx = TurnContext(tool_results=[ToolResult("search_guidance", "ok", {"passages": [{"text": PASSAGE}]}, "")])
    _, ok = QuoteGuardrail().check(f'"{PASSAGE}"', ctx)
    assert ok == "1 quote(s) verified against tool results"
    text, bad = QuoteGuardrail().check('"Pets are covered for broken legs up to 500 USD."', ctx)
    assert bad.startswith("1 of 1") and "could not be verified" in text


def test_handoff_tool_fails_closed_without_an_approved_prompt():
    case = _Tool("get_claim_case", ToolResult("get_claim_case", "ok",
                                              {"case_id": "CSET-LTD-002", "product_line": "ltd",
                                               "case_summary": "s", "documents": [
                                                   {"document_type": "a", "status": "received"},
                                                   {"document_type": "b", "status": "missing"}]}, "ok"))

    def no_champion():
        raise ConfigurationError("No approved version: alias @champion is not set.")
    result = HandoffTool(case, no_champion).run({"case_id": "CSET-LTD-002"})
    assert result.status == "error" and "@champion" in result.summary

    seen = {}
    assistant = SimpleNamespace(_prompt_version="5", predict=lambda *a: seen.update(args=a) or {
        "claim_id": "CSET-LTD-002", "benefit_determination": "Not determined"})
    result = HandoffTool(case, lambda: assistant).run({"case_id": "CSET-LTD-002"})
    assert result.status == "ok" and result.data["prompt_version"] == "5"
    assert seen["args"][1] == "long_term_disability" and seen["args"][3] == ["a"]
    assert "b" in seen["args"][2]                                     # missing documents in the narrative


def test_copilot_offers_all_tools_and_the_grounding_rules():
    tools = {n: _Tool(n, ToolResult(n, "ok", {}, "")) for n in ("get_claim_case", "search_guidance", "ask_portfolio")}
    agent = build_copilot(load_profile("local", env={}), LOG, tools=tools, model_factory=lambda k: None,
                          assistant_factory=lambda: None)
    assert set(agent.tools) == {"get_claim_case", "search_guidance", "ask_portfolio", "draft_handoff"}
    assert "I do not have this information in the knowledge base." in agent.config.system_prompt


def test_gateway_tool_turn_returns_calls_and_is_tagged():
    calls = []

    class _Completions:
        def create(self, **kw):
            calls.append(kw)
            fn = SimpleNamespace(name="get_claim_case", arguments='{"case_id": "A"}')
            message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="x", function=fn)])
            return SimpleNamespace(model="m", choices=[SimpleNamespace(message=message)],
                                   usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))
    client = GatewayChatClient("https://h", ModelRoute(alias="claims-small", upstream_model="system.ai.s"),
                               "unity_gateway", NoAuth(), LOG,
                               client=SimpleNamespace(chat=SimpleNamespace(completions=_Completions())))
    turn = client.complete_with_tools([{"role": "user", "content": "q"}], [{"type": "function"}])
    assert turn.tool_calls[0].name == "get_claim_case" and turn.prompt_tokens == 5
    assert calls[0]["tool_choice"] == "auto" and "Databricks-Ai-Gateway-Request-Tags" in calls[0]["extra_headers"]


def test_empty_model_replies_and_rate_limits_are_retried_once():
    agent, _ = _agent([ModelInvocationError("empty", "empty_response"), _turn("answer")], {})
    agent._sleep = lambda s: None
    events = list(agent.run("q"))
    assert any(e.type == "model_call" and e.data.get("status") == "retry" for e in events)
    assert next(e for e in events if e.type == "answer").data["text"] == "answer"
    agent, _ = _agent([ModelInvocationError("e", "empty_response"), ModelInvocationError("e", "empty_response")], {})
    agent._sleep = lambda s: None
    assert [e.type for e in agent.run("q")][-3:] == ["error", "usage", "done"]       # only one retry
    agent, _ = _agent([ModelInvocationError("auth", "auth")], {})
    assert not any(e.data.get("status") == "retry" for e in agent.run("q"))            # not retryable
