"""Verify the hackathon setup through the SAME code path the agent uses (cache + token ledger included).

Run:  .venv/Scripts/python check_setup.py
Cost: 2 short Token Factory calls the first time (a few hundred tokens), free afterwards (cached).
Exit code 0 only if the key works, a Nemotron chat model answers, and tool calling works.
"""
import json
import os
import sys

from cathodepilot import agent


def main() -> int:
    agent.load_env()
    if not os.environ.get("NEBIUS_API_KEY"):
        print("FAIL  NEBIUS_API_KEY missing. Copy .env.example to .env and paste your key.")
        return 1
    c = agent.client()
    try:
        ids = sorted(m.id for m in c.models.list().data)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  Token Factory /models: {type(e).__name__}: {e}. Check the key and NEBIUS_BASE_URL.")
        return 1
    nvidia = [i for i in ids if "nvidia" in i.lower() or "nemotron" in i.lower()]
    print(f"OK    Token Factory reachable, {len(ids)} models, {len(nvidia)} NVIDIA:")
    for m in nvidia:
        print(f"        {m}")
    ultra, nano = agent.choose_models(ids)
    if not ultra:
        print("FAIL  no NVIDIA chat model found; set AGENT_MODEL in .env to one of the ids above")
        return 1
    print(f"OK    agent model (AGENT_MODEL): {ultra}\n      fast model  (FAST_MODEL):  {nano}")

    ok = True
    r = agent.chat(c, "setup chat", model=nano, max_tokens=800,
                   messages=[{"role": "user", "content": "Reply with exactly: setup works"}])
    txt = agent.strip_thinking(r["content"])
    print(f"{'OK  ' if txt else 'FAIL'}  {nano} replied: {txt[:80]!r} (finish: {r['finish_reason']})")
    ok &= bool(txt)

    tool = agent._fn("get_cell_coverage", "Return archive coverage for a run", {"run_id": {"type": "string"}},
                     ["run_id"])
    r = agent.chat(c, "setup tools", model=ultra, max_tokens=2000, tools=[tool], temperature=0.2,
                   messages=[{"role": "user", "content": "Get the archive coverage for run 'moqd-oct-15'."}])
    calls = r["tool_calls"]
    good = bool(calls) and calls[0]["function"]["name"] == "get_cell_coverage" \
        and json.loads(calls[0]["function"]["arguments"]).get("run_id")
    print(f"{'OK  ' if good else 'FAIL'}  tool calling on {ultra}: "
          f"{calls[0]['function'] if calls else 'no tool call; reply was ' + repr(r['content'][:120])}")
    ok &= bool(good)

    u = agent.usage_summary()
    print(f"INFO  used so far: {u['calls']} live calls, {u['prompt_tokens'] + u['completion_tokens']} tokens, "
          f"about ${u['est_cost_usd']} of your ${u['spend_limit_usd']} limit (budget {u['budget_tokens']} tokens)")

    for name, var in (("Materials Project", "MP_API_KEY"), ("Tavily", "TAVILY_API_KEY")):
        print(f"{'SET ' if os.environ.get(var) else 'SKIP'}  {name} key {'present' if os.environ.get(var) else 'not set'}")
    if os.environ.get("CALLE_API_KEY"):
        print(f"SET   CALL-E key present, DRY_RUN={os.environ.get('DRY_RUN', '1')}, "
              f"supervisor ...{os.environ.get('SUPERVISOR_PHONE', '')[-4:]}")
    print("\nREADY: the agent can run." if ok else "\nNOT READY: fix the FAIL lines before spending more credits.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
