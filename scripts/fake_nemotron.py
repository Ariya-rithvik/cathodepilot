"""A strict, scripted stand-in for Token Factory, used to rehearse the agent without spending credits.

It speaks the OpenAI-compatible API (/v1/models, /v1/chat/completions) and is deliberately awkward:
  * rejects bad keys (401), assistant messages with extra fields, and tool results without a matching call id (400)
  * returns <think> text and a reasoning_content field, like reasoning models do
  * makes mistakes the agent must survive: a rejected plan, an unknown tool, bad arguments
  * FAKE_TEXT_TOOLCALLS=1 returns tool calls as <TOOLCALL> text instead of structured tool_calls
  * FAKE_FAIL_FIRST=n answers the first n chat requests with HTTP 500 (tests retries)
"""
from __future__ import annotations

import json
import os
import re
import uuid

from fastapi import FastAPI, Header, HTTPException, Request

KEY = "fake-key"
MODELS = ["nvidia/fake-Nemotron-Ultra", "nvidia/fake-Nemotron-Nano", "nvidia/fake-embedqa-e5", "meta/llama-x"]
app = FastAPI()
STATE = {"requests": 0, "fail_left": int(os.environ.get("FAKE_FAIL_FIRST", "0")), "log": []}


def _check(messages: list) -> None:
    ids = set()
    for m in messages:
        if m["role"] == "assistant":
            extra = set(m) - {"role", "content", "tool_calls"}
            if extra:
                raise HTTPException(400, f"assistant message has unsupported fields {sorted(extra)}")
            for tc in m.get("tool_calls") or []:
                ids.add(tc["id"])
        if m["role"] == "tool" and m.get("tool_call_id") not in ids:
            raise HTTPException(400, f"tool message with unknown tool_call_id {m.get('tool_call_id')}")


def _reply(content: str = "", calls: list | None = None) -> dict:
    calls = calls or []
    msg = {"role": "assistant", "content": "<think>checking the tool results</think>" + content,
           "reasoning_content": "internal reasoning that must not be sent back"}
    if calls and os.environ.get("FAKE_TEXT_TOOLCALLS") == "1":
        msg["content"] += "<TOOLCALL>" + json.dumps([{"name": n, "arguments": a} for n, a in calls]) + "</TOOLCALL>"
    elif calls:
        msg["tool_calls"] = [{"id": f"chatcmpl-tool-{uuid.uuid4().hex[:10]}", "type": "function",
                              "function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]
    return msg


@app.get("/v1/models")
def models(authorization: str = Header("")):
    if authorization != f"Bearer {KEY}":
        raise HTTPException(401, "invalid api key")
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in MODELS]}


@app.post("/v1/chat/completions")
async def chat(req: Request, authorization: str = Header("")):
    if authorization != f"Bearer {KEY}":
        raise HTTPException(401, "invalid api key")
    STATE["requests"] += 1
    if STATE["fail_left"] > 0:
        STATE["fail_left"] -= 1
        raise HTTPException(500, "fake upstream error")
    body = await req.json()
    if body["model"] not in MODELS or "embed" in body["model"]:
        raise HTTPException(404, f"model {body['model']} not found")
    msgs = body["messages"]
    _check(msgs)
    tools = {t["function"]["name"] for t in body.get("tools") or []}
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    STATE["log"].append({"tools": sorted(tools), "n_messages": len(msgs), "turn": len(tool_msgs)})

    if not tools and "RUN FACTS" in msgs[-1]["content"]:  # run report: one number from the facts, one invented
        facts = json.loads(msgs[-1]["content"].split("RUN FACTS\n", 1)[1].split("\n\nWhat happened", 1)[0])
        n = len(facts["incidents"])
        msg = _reply(f"FAKE SUMMARY. The run had {n} problem(s). Check the job time limit. It used 987 hours.")
    elif not tools:
        msg = _reply("setup works")
    elif "get_cell_coverage" in tools:
        msg = _reply(calls=[("get_cell_coverage", {"run_id": "moqd-oct-15"})])
    elif "integrity_checks" in tools:  # audit
        script = [[("run_overview", {}), ("integrity_checks", {})],
                  [("bounds_advice", {"descriptor": "density", "min": 0, "max": 5}), ("teleport", {})],
                  [("bounds_advice", {"descriptor": "density", "min": "zero"})]]
        step = sum(1 for m in msgs if m["role"] == "assistant")
        if step < len(script):
            msg = _reply(calls=script[step])
        else:
            res = [json.loads(m["content"]) for m in tool_msgs]
            reach = next((r.get("bins_reachable_with_observed_range") for r in res if isinstance(r, dict)
                          and "bins_reachable_with_observed_range" in r), "?")
            msg = _reply(f"FAKE AUDIT REPORT. integrity_checks returned findings; bounds_advice: {reach} of 10 "
                         "density bins reachable with bounds 0-5.")
    elif "propose_action" in tools:  # run watcher
        enum = next(t for t in body["tools"] if t["function"]["name"] == "propose_action")
        enum = enum["function"]["parameters"]["properties"]["action"]["enum"]
        if not tool_msgs:
            msg = _reply(calls=[("run_status", {}), ("tail_log", {"lines": 5})])
        else:
            st = json.loads(tool_msgs[0]["content"])
            act = "kill_stuck_child" if "kill_stuck_child" in enum else "notify_only"
            msg = _reply(calls=[("propose_action", {
                "action": act, "question": "The run is stuck and a job looks hung. May I stop that job?",
                "reason": f"run CPU {st.get('cpu_pct')}%, hung job {st.get('stuck_job_candidates')}"})])
    elif "submit_plan" in tools:  # planner
        state = json.loads(re.search(r"Current archive state:\n(\{.*\})\n", msgs[1]["content"], re.S).group(1))
        last = json.loads(tool_msgs[-1]["content"]) if tool_msgs else None
        asked = any('"approved"' in m["content"] for m in tool_msgs)
        if last is None:  # first try: a plan the validator must reject
            msg = _reply(calls=[("submit_plan", {"operator_weights": {"strain": 0.5, "substitute": 0.5,
                                                                      "teleport": 1}, "rationale": "bad first try"})])
        elif state["round"] == 0 and not asked:
            msg = _reply(calls=[("ask_human", {"question": "Raise rattle weight to 0.6?",
                                               "proposed_action": "rattle 0.6, strain 0.3, permute 0.1",
                                               "reason": f"{state['duplicates']} duplicates so far"})])
        else:
            occ = {tuple(o["cell"]) for o in state["occupied"]}
            near = sorted({(x + dx, y + dy) for x, y in occ for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                           if 0 <= x + dx < 10 and 0 <= y + dy < 10} - occ)[:4]
            msg = _reply(calls=[("submit_plan", {"operator_weights": {"strain": 0.5, "rattle": 0.3, "permute": 0.2},
                                                 "target_cells": [list(c) for c in near],
                                                 "rationale": f"round {state['round']}: {state['occupied_cells']} "
                                                              f"cells filled; aim at empty neighbours {near}"})])
    else:
        msg = _reply("no idea")
    pt = len(json.dumps(body)) // 4
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion", "model": body["model"],
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if "tool_calls" in msg else "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": 60, "total_tokens": pt + 60}}


@app.get("/stats")
def stats():
    return STATE
