"""Job mode of the Antigravity image tools.

antigravity_image and antigravity_image_swarm run their generation as a job and
answer `status: running` + a job_id once `wait_sec` passes, so a client that
abandons calls at ~60 s (Claude Code) can still collect the result through
antigravity_job_result. The fake runners here block on an Event the test
releases, so "slow" costs no wall-clock time.
"""

import asyncio
import os
import re
import threading

import pytest

import server
import swarm

# Through server: importing antigravity_tools before server is a circular import.
ag = server.antigravity_tools

_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 8


@pytest.fixture
def scratch_dir(tmp_path, monkeypatch):
    d = tmp_path / "scratch"
    d.mkdir()
    monkeypatch.setattr(server, "SCRATCH_DIR", d)
    return d


@pytest.fixture
def release():
    """An Event the fake runners wait on; always set at teardown so no job
    thread outlives its test."""
    ev = threading.Event()
    yield ev
    ev.set()
    with ag._IMAGE_JOBS_LOCK:
        jobs = list(ag._IMAGE_JOBS.values())
        ag._IMAGE_JOBS.clear()
    for job in jobs:
        job.done.wait(5)


def _job_id(text: str) -> str:
    m = re.search(r"^job_id: (\w+)$", text, re.M)
    assert m, f"no job_id in: {text!r}"
    return m.group(1)


def _slow_image_run(release, path, payload=_JPEG):
    def fake_run(prompt, ws, continue_conv, timeout_s):
        assert release.wait(10), "test never released the fake agy run"
        path.write_bytes(payload)
        return str(path)

    return fake_run


def test_slow_image_returns_job_id_then_job_result_collects_it(
    tmp_path, scratch_dir, monkeypatch, release
):
    target = tmp_path / "art.png"
    monkeypatch.setattr(server, "_run_agy", _slow_image_run(release, target))

    first = asyncio.run(
        server.antigravity_image(
            "a cat", output_path=str(target), workspace=str(tmp_path), wait_sec=0
        )
    )
    assert first.startswith("status: running\n")
    assert f"output_path: {target}" in first
    assert "antigravity_job_result" in first
    job_id = _job_id(first)

    # Still running: the poll answers running again with the same id.
    again = asyncio.run(server.antigravity_job_result(job_id, wait_sec=0))
    assert again.startswith("status: running\n")
    assert _job_id(again) == job_id

    release.set()
    done = asyncio.run(server.antigravity_job_result(job_id, wait_sec=10))
    # Exactly the format a fast antigravity_image returns, extension corrected.
    assert done == f"{tmp_path / 'art.jpg'}\nformat=JPEG  size={len(_JPEG)} bytes"
    assert os.path.isfile(tmp_path / "art.jpg")

    # A finished result can be collected again while it is retained.
    assert asyncio.run(server.antigravity_job_result(job_id, wait_sec=0)) == done


def test_image_waits_up_to_wait_sec_before_answering_running(
    tmp_path, scratch_dir, monkeypatch, release
):
    target = tmp_path / "art.png"
    monkeypatch.setattr(server, "_run_agy", _slow_image_run(release, target))
    monkeypatch.setattr(server, "_PROGRESS_NOTIFY_INTERVAL_S", 0.05)

    class Ctx:
        def __init__(self):
            self.calls = []

        async def report_progress(self, progress, total, message):
            self.calls.append((progress, total, message))

    ctx = Ctx()
    out = asyncio.run(
        server.antigravity_image(
            "a cat",
            output_path=str(target),
            workspace=str(tmp_path),
            timeout_s=240,
            wait_sec=0.3,
            ctx=ctx,
        )
    )
    assert out.startswith("status: running\n")
    # Progress kept flowing while the call waited, on the job's timeout scale.
    assert ctx.calls
    assert all(total == 240.0 and msg.startswith("agy running") for _, total, msg in ctx.calls)


def test_job_that_finishes_within_wait_returns_the_plain_result(
    tmp_path, scratch_dir, monkeypatch, release
):
    target = tmp_path / "art.png"
    release.set()  # the "slow" run is not slow at all
    monkeypatch.setattr(server, "_run_agy", _slow_image_run(release, target))
    out = asyncio.run(
        server.antigravity_image("a cat", output_path=str(target), workspace=str(tmp_path))
    )
    assert "status: running" not in out
    assert out.splitlines()[0] == str(tmp_path / "art.jpg")


def test_job_result_raises_the_jobs_error(tmp_path, scratch_dir, monkeypatch, release):
    target = tmp_path / "art.png"

    def slow_failure(prompt, ws, continue_conv, timeout_s):
        assert release.wait(10)
        raise RuntimeError("agy exited 1")

    monkeypatch.setattr(server, "_run_agy", slow_failure)
    first = asyncio.run(
        server.antigravity_image(
            "a cat", output_path=str(target), workspace=str(tmp_path), wait_sec=0
        )
    )
    job_id = _job_id(first)
    release.set()
    # The same error antigravity_image raises when nothing was produced.
    with pytest.raises(RuntimeError, match="no image file found.*agy also failed: agy exited 1"):
        asyncio.run(server.antigravity_job_result(job_id, wait_sec=10))


def test_job_result_unknown_job_id():
    with pytest.raises(RuntimeError, match="Unknown job_id 'nope'"):
        asyncio.run(server.antigravity_job_result("nope", wait_sec=0))


def test_finished_job_expires_after_retention(release):
    job = ag._start_job("antigravity_image", ["x.png"], 240, lambda: "done")
    assert job.done.wait(5)
    assert asyncio.run(server.antigravity_job_result(job.id, wait_sec=0)) == "done"

    job.finished -= ag._JOB_RETENTION_S + 1
    with pytest.raises(RuntimeError, match="Unknown job_id"):
        asyncio.run(server.antigravity_job_result(job.id, wait_sec=0))


def test_running_job_is_never_purged(release):
    job = ag._start_job("antigravity_image", ["x.png"], 240, lambda: release.wait(10) and "ok")
    job.started -= ag._JOB_RETENTION_S * 2  # running far longer than the retention
    ag._purge_expired_jobs()
    assert ag._get_job(job.id) is job


def test_image_swarm_job_flow(tmp_path, monkeypatch, release):
    paths = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    seen = {}

    def fake_swarm_image(prompts, output_paths, **kw):
        assert release.wait(10)
        seen.update(prompts=prompts, output_paths=output_paths, **kw)
        return ["result-a", "result-b"]

    monkeypatch.setattr(swarm, "swarm_image", fake_swarm_image)
    monkeypatch.setattr(swarm, "format_image_results", lambda results: " | ".join(results))

    first = asyncio.run(
        server.antigravity_image_swarm(
            ["a", "b"], output_paths=paths, workspaces=[str(tmp_path)], wait_sec=0
        )
    )
    assert first.startswith("status: running\n")
    assert "tool: antigravity_image_swarm" in first
    for p in paths:
        assert f"  {p}" in first
    job_id = _job_id(first)

    release.set()
    done = asyncio.run(server.antigravity_job_result(job_id, wait_sec=10))
    assert done == "result-a | result-b"
    assert seen["output_paths"] == paths
    assert seen["workspaces"] == str(tmp_path)  # 1-item list -> same dir for all


@pytest.mark.parametrize(
    "requested, expected",
    [
        (None, 45.0),
        (45, 45.0),
        (10, 10.0),
        (999, 50.0),
        (-3, 0.0),
        ("x", 45.0),
        (float("nan"), 45.0),
    ],
)
def test_wait_sec_is_clamped(requested, expected):
    assert ag._clamp_wait(requested) == expected


def test_job_result_tool_is_registered_and_read_only():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    ann = tools["antigravity_job_result"].annotations
    # MCP SDK v2 (fastmcp 4) renamed readOnlyHint to read_only_hint.
    read_only = getattr(ann, "read_only_hint", None)
    assert (read_only if read_only is not None else ann.readOnlyHint) is True
    for name in ("antigravity_image", "antigravity_image_swarm", "antigravity_job_result"):
        assert "wait_sec" in tools[name].parameters["properties"]
