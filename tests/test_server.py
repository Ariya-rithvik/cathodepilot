from fastapi.testclient import TestClient

from app.server import app

c = TestClient(app)


def test_read_endpoints_work_on_real_data():
    for url in ("/api/overview", "/api/audit", "/api/bench", "/api/shortlist", "/api/agent/status"):
        assert c.get(url).status_code == 200, url
    assert c.get("/api/audit").json()["checks"]


def test_evidence_viewer_only_serves_data_files():
    assert c.get("/api/file", params={"path": "data/runs/moqd-oct15/4x48.got"}).status_code == 200
    assert c.get("/api/file", params={"path": ".env.example"}).status_code == 403
    assert c.get("/api/file", params={"path": "data/../.env.example"}).status_code == 403


def test_watch_page_and_notebook_edits(monkeypatch, tmp_path_factory=None):
    from pathlib import Path
    nb = Path(__file__).parent / "_nb_server.json"
    monkeypatch.setenv("CATHODEPILOT_NOTEBOOK", str(nb))
    try:
        assert c.get("/api/watch").status_code == 200
        assert c.post("/api/notebook", json={"kind": "never", "text": "Never restart mid batch"}).status_code == 200
        assert c.post("/api/notebook", json={"kind": "step", "text": "generation 35"}).json()["step"] == "generation 35"
        assert c.post("/api/notebook", json={"kind": "delete_everything", "text": "x"}).status_code == 400
        assert any(r["text"] == "Never restart mid batch" for r in c.get("/api/watch").json()["notebook"]["rules"])
    finally:
        nb.unlink(missing_ok=True)


def test_file_viewer_never_serves_secrets_or_llm_cache(monkeypatch):
    for bad in ("data/cache/li_reference.json", "data/memory/notebook.json", "../.env", ".env.example"):
        assert c.get("/api/file", params={"path": bad}).status_code == 403, bad


def test_public_demo_is_read_only(monkeypatch):
    import app.server as srv
    monkeypatch.setattr(srv, "PUBLIC", True)
    assert c.post("/api/agent/audit").status_code == 403
    assert c.post("/api/notebook", json={"kind": "note", "text": "x"}).status_code == 403
    w = c.get("/api/watch").json()
    assert w["read_only"] and "top_memory" not in w["system"]


def test_network_and_absolute_paths_are_refused_without_touching_the_disk():
    for bad in ("\\\\192.0.2.1\\share\\x", "//192.0.2.1/share/x", "C:/Windows/win.ini", "/etc/passwd"):
        r = c.get("/api/file", params={"path": bad}, timeout=3)
        assert r.status_code == 403, bad


def test_public_watch_is_an_allowlist(monkeypatch):
    import json
    from pathlib import Path

    import app.server as srv
    state = Path(srv.ROOT) / "data/memory/watch__public_test.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"name": "r", "events": [{"text": "started secret.exe"}], "prestart": [{"text": "password"}],
                                 "run": {"alive": True, "pid": 5, "children": [{"name": "gulp"}]}, "why": "private",
                                 "progress": {"kind": "log", "last_line": "secret line"},
                                 "issues": [{"signature": "stalled_idle", "summary": "secret summary"}]}))
    monkeypatch.setattr(srv, "PUBLIC", True)
    try:
        text = c.get("/api/watch").text
    finally:
        state.unlink()
    assert srv._outcome_word("recorded; phone: my flat is 12 Park Road") == "recorded"
    assert srv._outcome_word("not executed: backup failed (D:/secret)") == "not executed"
    for leak in ("secret", "private", "password", "pid", "top_memory"):
        assert leak not in text, leak
