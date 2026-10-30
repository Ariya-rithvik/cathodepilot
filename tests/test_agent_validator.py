from cathodepilot.agent import validate_plan


def test_valid_plan_passes():
    ok, errors, clean = validate_plan({"operator_weights": {"strain": 1, "substitute": 0.5},
                                       "dopants": ["Al", "Fe"], "target_cells": [[3, 4]],
                                       "rationale": "fill sparse cells"}, (10, 10))
    assert ok and not errors
    assert clean["target_cells"] == [(3, 4)]


def test_invalid_plan_is_rejected_with_reasons():
    ok, errors, _ = validate_plan({"operator_weights": {"teleport": 1, "strain": -1},
                                   "dopants": ["U"], "target_cells": [[12, 0]]}, (10, 10))
    assert not ok
    joined = " ".join(errors)
    assert "teleport" in joined and "U" in joined and "outside" in joined


def test_zero_weights_rejected():
    ok, errors, _ = validate_plan({"operator_weights": {"strain": 0}}, (10, 10))
    assert not ok and any("sum" in e for e in errors)


def test_ask_human_tool_is_offered():
    from cathodepilot.agent import PLAN_TOOLS
    names = [t["function"]["name"] for t in PLAN_TOOLS]
    assert "ask_human" in names


def test_substitute_blocked_without_hull():
    plan = {"operator_weights": {"strain": 1, "substitute": 0.5}}
    assert validate_plan(plan, (10, 10), allow_substitute=True)[0]
    ok, errors, _ = validate_plan(plan, (10, 10), allow_substitute=False)
    assert not ok and any("hull" in e for e in errors)


def test_non_numeric_weight_is_an_error_not_a_crash():
    ok, errors, _ = validate_plan({"operator_weights": {"strain": "high"}}, (10, 10))
    assert not ok and any("not a number" in e for e in errors)


def test_choose_models_skips_embedding_and_guard_models(monkeypatch):
    from cathodepilot.agent import choose_models
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("FAST_MODEL", raising=False)
    ids = ["nvidia/llama-nemotron-embed-ultra", "nvidia/Nemotron-Safety-Guard", "nvidia/Nemotron-3-Ultra",
           "nvidia/Nemotron-3-Nano", "meta/llama"]
    assert choose_models(ids) == ("nvidia/Nemotron-3-Ultra", "nvidia/Nemotron-3-Nano")


def test_thinking_and_text_toolcalls_are_handled():
    from cathodepilot.agent import _tool_calls_from_text, strip_thinking
    assert strip_thinking("<think>hmm</think>answer") == "answer"
    calls = _tool_calls_from_text('<TOOLCALL>[{"name": "submit_plan", "arguments": {"rationale": "x"}}]</TOOLCALL>')
    assert calls[0]["function"]["name"] == "submit_plan"


class _FakeClient:
    """Counts what would really be sent."""
    def __init__(self, exc=None):
        from types import SimpleNamespace as NS
        self.sent, self.exc = 0, exc
        msg = NS(content="ok", tool_calls=None)
        reply = NS(choices=[NS(message=msg, finish_reason="stop")], usage=NS(prompt_tokens=1000000, completion_tokens=1000000))

        def create(**kw):
            self.sent += 1
            if self.exc:
                raise self.exc
            return reply
        self.chat = NS(completions=NS(create=create))


def _guarded(monkeypatch, limit):
    from pathlib import Path
    d = Path(__file__).parent / "_spend_cache"
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    monkeypatch.setenv("CATHODEPILOT_LLM_CACHE", str(d))
    monkeypatch.setenv("SPEND_LIMIT_USD", str(limit))
    monkeypatch.setenv("TOKEN_BUDGET", "999999999")
    monkeypatch.setenv("PRICE_IN_PER_M", "1")
    monkeypatch.setenv("PRICE_OUT_PER_M", "3")
    return d


def test_spend_limit_stops_calls_before_they_are_sent(monkeypatch):
    import shutil

    import pytest

    from cathodepilot import agent
    d = _guarded(monkeypatch, 5)
    try:
        c = _FakeClient()
        ask = lambda i: agent.chat(c, "t", model="m", max_tokens=100, messages=[{"role": "user", "content": str(i)}])  # noqa: E731
        ask(1)  # each reply bills $4
        assert c.sent == 1 and agent.usage_summary()["est_cost_usd"] == 4
        ask(2)
        with pytest.raises(agent.BudgetExceeded):
            ask(3)
        assert c.sent == 2, "nothing may be sent once the limit would be passed"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_one_huge_call_is_refused_on_its_worst_case(monkeypatch):
    import shutil

    import pytest

    from cathodepilot import agent
    d = _guarded(monkeypatch, 1)
    try:
        c = _FakeClient()
        with pytest.raises(agent.BudgetExceeded):
            agent.chat(c, "t", model="m", max_tokens=1000000, messages=[{"role": "user", "content": "x"}])
        assert c.sent == 0
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_lost_answer_still_counts_as_spent(monkeypatch):
    import shutil

    import pytest

    from cathodepilot import agent
    APITimeoutError = type("APITimeoutError", (Exception,), {})
    d = _guarded(monkeypatch, 5)
    try:
        c = _FakeClient(APITimeoutError("lost"))
        with pytest.raises(APITimeoutError):
            agent.chat(c, "t", model="m", max_tokens=1000, messages=[{"role": "user", "content": "x"}])
        assert agent.usage_summary()["completion_tokens"] == 1000
    finally:
        shutil.rmtree(d, ignore_errors=True)
