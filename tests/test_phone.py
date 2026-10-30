from cathodepilot import phone


class FakeCallE(phone.CallE):
    def __init__(self, result, **kw):
        super().__init__("key", allowed="+919999999999", **kw)
        self.result, self.bodies = result, []

    def _req(self, path, body=None, headers=None):
        if body is not None:
            self.bodies.append((body, headers))
            return {"call_id": "c1"}
        return {"status": "completed", "structured_result": self.result}


def ask(client):
    sup = phone.phone_supervisor(poll_s=0, client=client)
    return sup(question="Widen density bounds?", proposed_action="density 2.0-5.5", reason="3 of 10 bins reachable")


def test_approval_comes_back_from_the_call(monkeypatch):
    monkeypatch.setenv("SUPERVISOR_PHONE", "+919999999999")
    c = FakeCallE({"heard": True, "approved": True, "said": "yes approve"})
    ans = ask(c)
    assert ans["approved"] and ans["call_id"] == "c1"
    body, headers = c.bodies[0]
    assert "Widen density bounds?" in body["task"] and headers["Idempotency-Key"].startswith("cathodepilot-")


def test_unheard_or_rejected_is_not_approved(monkeypatch):
    monkeypatch.setenv("SUPERVISOR_PHONE", "+919999999999")
    assert not ask(FakeCallE({"heard": False, "approved": True, "said": ""}))["approved"]
    assert not ask(FakeCallE({"heard": True, "approved": False, "said": "reject"}))["approved"]


def test_allowlist_and_cap(monkeypatch):
    monkeypatch.setenv("SUPERVISOR_PHONE", "+918888888888")
    assert ask(FakeCallE({}))["status"] == "skipped"
    monkeypatch.setenv("SUPERVISOR_PHONE", "+919999999999")
    c = FakeCallE({"heard": True, "approved": True, "said": "ok"}, max_calls=1)
    assert ask(c)["approved"] and ask(c)["status"] == "skipped"
