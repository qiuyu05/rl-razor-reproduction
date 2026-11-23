"""scripts/launch.py scheduling safety (R41): dependencies, locks, budget guard, logs, signals. No Modal."""
import fcntl
import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from rlrazor.config import RunConfig, run_id

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("script_launch_t", ROOT / "scripts" / "launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


launch = _load()


class FakeProc:
    def __init__(self, cmd, stdout=None, stderr=None, cwd=None, pass_fds=()):
        self.cmd, self.stdout, self.pass_fds = cmd, stdout, pass_fds
        self.code, self.signals, self.killed, self.exit_on_sigint = None, [], False, True

    def poll(self):
        return self.code

    def send_signal(self, sig):
        self.signals.append(sig)
        if self.exit_on_sigint:
            self.code = -sig

    def wait(self, timeout=None):
        if self.code is None:
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        return self.code

    def kill(self):
        self.killed, self.code = True, -9


class Spawner:
    def __init__(self):
        self.procs = {}

    def __call__(self, cmd, **kw):
        p = FakeProc(cmd, **kw)
        self.procs[Path(cmd[2]).stem] = p
        return p


def job(name, *, init_from=None, status=None, method="sft"):
    cfg = RunConfig(arm="q0", method=method, lr=1e-5, init_from=init_from)
    st = status or {"index": None, "scored_steps": [], "base_row": {}, "general_pool": True}
    return launch.Job(f"configs/{name}.yaml", cfg, name, st)


def make(tmp_path, jobs, **kw):
    sp = kw.pop("spawn", None) or Spawner()
    lines = []
    lz = launch.Launcher(jobs, logs=tmp_path / "logs", locks=tmp_path / "locks", spawn=sp,
                         sleep=lambda s: None, out=lines.append, **kw)
    return lz, sp, lines


def finish(sp, name, code=0):
    sp.procs[name].code = code


def test_dependency_not_trained_is_deferred_and_never_started(tmp_path):
    lz, sp, _ = make(tmp_path, [job("child", init_from="grpo-x/step-400")], ready=lambda r: False)
    assert lz.run() == 1
    assert lz.jobs[0].state == "deferred" and "grpo-x" in lz.jobs[0].reason and not sp.procs


def test_dependency_in_same_launch_waits_for_its_source(tmp_path):
    trained = set()
    src, child = job("grpo-x"), job("child", init_from="grpo-x/step-400")
    lz, sp, _ = make(tmp_path, [src, child], ready=lambda r: r in trained)
    assert lz.step() is True
    assert set(sp.procs) == {"grpo-x"} and child.state == "queued"
    finish(sp, "grpo-x")
    trained.add("grpo-x")
    assert lz.step() is True and set(sp.procs) == {"grpo-x", "child"}
    finish(sp, "child")
    assert lz.step() is False and lz.summary() == 0


def test_trained_child_needs_no_source(tmp_path):
    st = {"index": {"ckpts": [[1, "x"], [2, "y"]]}, "scored_steps": [1]}
    lz, sp, _ = make(tmp_path, [job("child", init_from="gone/step-1", status=st)], ready=lambda r: False)
    lz.step()
    assert "child" in sp.procs  # only its unscored checkpoint is left: eval-only, no dependency


def test_done_already_is_not_started(tmp_path):
    st = {"index": {"ckpts": [[1, "x"]]}, "scored_steps": [1]}
    lz, sp, _ = make(tmp_path, [job("a", status=st)])
    assert lz.run() == 0 and lz.jobs[0].state == "done already" and not sp.procs


def test_concurrency_cap(tmp_path):
    lz, sp, _ = make(tmp_path, [job(n) for n in "abc"], max_parallel=2)
    lz.step()
    assert set(sp.procs) == {"a", "b"}
    finish(sp, "a")
    lz.step()
    assert set(sp.procs) == {"a", "b", "c"}


def test_lock_held_elsewhere_skips(tmp_path):
    (tmp_path / "locks").mkdir()
    fd = os.open(tmp_path / "locks" / "a.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another launch (or its orphaned child) is running "a"
    try:
        lz, sp, _ = make(tmp_path, [job("a"), job("b")])
        lz.step()
        assert lz.jobs[0].state == "skipped" and "already running" in lz.jobs[0].reason
        assert set(sp.procs) == {"b"}
    finally:
        os.close(fd)


def test_lock_survives_launcher_while_child_lives(tmp_path):
    """The child inherits the lock fd: if the launcher dies, a relaunch still sees the run as in flight."""
    real = []

    def spawn(cmd, stdout=None, stderr=None, cwd=None, pass_fds=()):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], pass_fds=pass_fds)
        real.append(p)
        return p

    lz, _, _ = make(tmp_path, [job("a")], spawn=spawn)
    try:
        lz.step()
        lz._release(lz.jobs[0])  # the launcher's copy is gone (as if it were SIGKILLed)
        again, sp2, _ = make(tmp_path, [job("a")])
        again.step()
        assert again.jobs[0].state == "skipped" and not sp2.procs
    finally:
        for p in real:
            p.kill()
            p.wait()
    third, sp3, _ = make(tmp_path, [job("a")])
    third.step()
    assert "a" in sp3.procs  # child gone -> lock free


def test_budget_guard_waits_then_reports_not_launched(tmp_path):
    lz, sp, lines = make(tmp_path, [job("a"), job("b")], budget=5.0, spent=1.0,
                         projector=lambda j: 3.0, actual_cost=lambda j: 3.5)
    lz.step()
    assert set(sp.procs) == {"a"} and lz.jobs[1].state == "queued"  # 1 + 3 + 3 > 5: waits while "a" runs
    finish(sp, "a")
    assert lz.step() is False
    assert lz.jobs[0].cost == 3.5 and lz.jobs[1].state == "not launched" and "budget" in lz.jobs[1].reason
    assert lz.summary() == 1 and "b" not in sp.procs


def test_budget_guard_uses_measured_cost_to_free_headroom(tmp_path):
    lz, sp, _ = make(tmp_path, [job("a"), job("b")], budget=5.0, spent=1.0,
                     projector=lambda j: 3.0, actual_cost=lambda j: 0.5)
    lz.step()
    finish(sp, "a")
    lz.step()
    assert "b" in sp.procs  # 1 + 0.5 + 3 <= 5


def test_failed_run_is_charged_at_least_its_projection(tmp_path):
    lz, sp, _ = make(tmp_path, [job("a")], budget=10.0, projector=lambda j: 3.0, actual_cost=lambda j: 1.0)
    lz.step()
    finish(sp, "a", code=1)
    lz.step()
    assert lz.jobs[0].state == "failed" and lz.jobs[0].cost == 3.0


def test_budget_without_cost_model_is_not_launched(tmp_path):
    def projector(j):
        raise ValueError("no cost model for method 'teacher_sft'")

    lz, sp, _ = make(tmp_path, [job("a")], budget=10.0, projector=projector)
    lz.run()
    assert lz.jobs[0].state == "not launched" and "no cost model" in lz.jobs[0].reason and not sp.procs


def test_logs_append_with_header(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "a.log").write_text("earlier attempt\n")
    lz, sp, _ = make(tmp_path, [job("a")])
    lz.step()
    sp.procs["a"].stdout.write("child output\n")
    finish(sp, "a")
    lz.step()
    text = (tmp_path / "logs" / "a.log").read_text()
    assert text.startswith("earlier attempt\n") and "===== launch.py" in text and "--require-prepared" in text
    assert text.index("===== launch.py") < text.index("child output")
    assert sp.procs["a"].pass_fds and sp.procs["a"].cmd[3] == "--require-prepared"


def test_stop_interrupts_children_and_kills_stragglers(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "GRACE_S", 0)
    lz, sp, _ = make(tmp_path, [job("a"), job("b"), job("c")], max_parallel=2)
    lz.step()
    sp.procs["b"].exit_on_sigint = False  # ignores SIGINT
    lz.stop(signal.SIGTERM)
    assert sp.procs["a"].signals == [signal.SIGINT] and sp.procs["b"].killed
    assert not lz.step()  # nothing new starts after a stop
    assert "c" not in sp.procs and lz.summary() == 1
    assert [j.state for j in lz.jobs] == ["failed", "failed", "queued"]
    assert not list((tmp_path / "locks").glob("*.lock")) or all(j.lock_fd is None for j in lz.jobs)


def test_source_run_id():
    assert launch.source_run_id(RunConfig(arm="q1", method="rl_distill_sft", lr=1e-5,
                                          init_from="grpo-a-1/step-400")) == "grpo-a-1"
    assert launch.source_run_id(RunConfig(arm="q0", method="sft", lr=1e-5, init_from="base")) is None
    assert run_id(RunConfig(arm="q0", method="sft", lr=1e-5))  # sanity: config ids still build


# ---------------------------------------------------------------- launcher accounting (R44)

def _hold_lock(tmp_path, name):
    (tmp_path / "locks").mkdir(exist_ok=True)
    fd = os.open(tmp_path / "locks" / f"{name}.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def test_skipped_running_elsewhere_stays_committed(tmp_path):
    """N1: after a crash + relaunch, orphans' remaining cost must still count against the budget."""
    fd = _hold_lock(tmp_path, "a")
    try:
        lz, sp, _ = make(tmp_path, [job("a"), job("b"), job("c")], budget=10.0, spent=1.0,
                         projector=lambda j: 4.0)
        lz.step()
        assert lz.jobs[0].state == "skipped" and lz.jobs[0].projected == 4.0
        assert set(sp.procs) == {"b"} and lz.jobs[2].state == "queued"  # 1 + 4 (orphan) + 4 + 4 > 10
        assert lz.committed() == pytest.approx(9.0)
        finish(sp, "b")
        assert lz.step() is False and lz.jobs[2].state == "not launched"
    finally:
        os.close(fd)


def test_budget_wait_does_not_keep_the_lock(tmp_path):
    lz, sp, _ = make(tmp_path, [job("a"), job("b")], budget=5.0, projector=lambda j: 3.0)
    lz.step()
    assert lz.jobs[1].state == "queued" and lz.jobs[1].lock_fd is None
    fd = _hold_lock(tmp_path, "b")  # the waiting job holds no lock
    os.close(fd)


def test_measurement_error_charges_projection_and_loop_continues(tmp_path):
    def boom(j):
        raise RuntimeError("modal: connection reset")

    lz, sp, lines = make(tmp_path, [job("a"), job("b")], max_parallel=1, budget=100.0,
                         projector=lambda j: 2.0, actual_cost=boom)
    lz.step()
    finish(sp, "a")
    lz.step()
    assert lz.jobs[0].state == "ok" and lz.jobs[0].cost == 2.0 and "b" in sp.procs
    assert any("charging the projection" in x for x in lines)


def test_dependency_check_error_waits_then_defers(tmp_path):
    def ready(dep):
        raise RuntimeError("modal down")

    lz, sp, _ = make(tmp_path, [job("a"), job("child", init_from="src/step-1")], ready=ready)
    assert lz.step() is True
    assert lz.jobs[1].state == "queued" and "failed" in lz.jobs[1].reason  # waits while "a" runs
    finish(sp, "a")
    assert lz.step() is False
    assert lz.jobs[1].state == "deferred" and "modal down" in lz.jobs[1].reason and "child" not in sp.procs


def test_loop_error_stops_children_and_prints_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "GRACE_S", 0)

    def sleep(s):
        raise RuntimeError("unexpected")

    sp = Spawner()
    lines = []
    lz = launch.Launcher([job("a"), job("b")], logs=tmp_path / "logs", locks=tmp_path / "locks", spawn=sp,
                         sleep=sleep, out=lines.append, max_parallel=1)
    assert lz.run() == 1
    assert sp.procs["a"].signals == [signal.SIGINT] and lz.jobs[0].state == "failed"
    assert any(x.startswith("summary") for x in lines) and any("scheduling loop failed" in x for x in lines)
    assert lz.jobs[1].state == "queued" and "b" not in sp.procs


def _teacher_jobs():
    data = launch.Job("configs/td.yaml", RunConfig(arm="q0", method="teacher_data", lr=0.0), "teacher-data",
                      {"index": None, "scored_steps": []})
    sft = job("tsft", method="teacher_sft")
    return data, sft


def test_teacher_sft_waits_for_teacher_data_job(tmp_path):
    """M1: the teacher data set is its own budgeted job; teacher_sft configs depend on it."""
    ready = set()
    data, sft = _teacher_jobs()
    lz, sp, _ = make(tmp_path, [data, sft], budget=50.0, projector=lambda j: 1.5, ready=lambda d: d in ready)
    lz.step()
    assert set(sp.procs) == {"td"} and sft.state == "queued" and "teacher_data" in sft.reason
    assert lz.committed() == pytest.approx(1.5)
    finish(sp, "td")
    ready.add(launch.TEACHER_DATA)
    lz.step()
    assert "tsft" in sp.procs


def test_teacher_sft_deferred_without_teacher_data(tmp_path):
    _, sft = _teacher_jobs()
    lz, sp, _ = make(tmp_path, [sft], ready=lambda d: False)
    lz.run()
    assert sft.state == "deferred" and "teacher_data" in sft.reason and not sp.procs


def test_teacher_data_done_already(tmp_path):
    data, _ = _teacher_jobs()
    lz, sp, _ = make(tmp_path, [data], ready=lambda d: d == launch.TEACHER_DATA)
    assert lz.run() == 0 and data.state == "done already" and not sp.procs


def test_teacher_data_cost_model():
    spec = importlib.util.spec_from_file_location("script_cost_t", ROOT / "scripts" / "cost_report.py")
    cr = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cr
    spec.loader.exec_module(cr)
    u = cr.Units(eval_s=700.0, sft_step_s=1.0, grpo_step_s=10.0, sources={})
    p = cr.Prices()
    usd, est = cr.run_cost(RunConfig(arm="q0", method="teacher_data", lr=0.0), u, p)
    assert est and usd == pytest.approx(p.train_usd(cr.teacher_data_seconds(p, "qwen2.5-7b-instruct")))  # 0 evals
    assert cr.run_cost(RunConfig(arm="q0", method="teacher_sft", lr=1e-5), u, p)[1] is True


def test_wall_clock_timeout_kills_and_charges(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "GRACE_S", 0)
    clock = [0.0]
    lz, sp, _ = make(tmp_path, [job("a")], budget=50.0, projector=lambda j: 2.0, actual_cost=lambda j: 0.5,
                     wall_projector=lambda j: 100.0, timeout_factor=3.0, min_timeout_s=0.0, now=lambda: clock[0])
    lz.step()
    assert lz.jobs[0].deadline == pytest.approx(300.0)
    sp.procs["a"].exit_on_sigint = False
    clock[0] = 299.0
    lz.step()
    assert lz.jobs[0].state == "running"
    clock[0] = 301.0
    lz.step()
    j = lz.jobs[0]
    assert j.state == "failed" and j.reason.startswith("timeout") and sp.procs["a"].killed
    assert j.cost == 2.0 and j.lock_fd is None  # max(measured 0.5, projected 2.0)


def test_timeout_floor(tmp_path):
    lz, sp, _ = make(tmp_path, [job("a")], wall_projector=lambda j: 10.0, min_timeout_s=3600.0,
                     now=lambda: 0.0)
    lz.step()
    assert lz.jobs[0].deadline == pytest.approx(3600.0)


def test_spawn_failure_releases_lock_and_is_listed(tmp_path):
    def spawn(cmd, **kw):
        raise OSError("fork failed")

    lz, _, lines = make(tmp_path, [job("a")], spawn=spawn)
    assert lz.run() == 1
    assert lz.jobs[0].state == "failed" and "spawn failed" in lz.jobs[0].reason and lz.jobs[0].lock_fd is None
    os.close(_hold_lock(tmp_path, "a"))  # lock is free again
    assert any("failed" in x and "a" in x for x in lines if x.startswith("  "))


# ---- R50: staggered spawns and Modal app-creation rate limit

RATE_LIMIT = ("modal.exception.ResourceExhaustedError: App create rate limit exceeded. Please wait and retry.\n")


def _write(sp, name, text):
    sp.procs[name].stdout.write(text)
    sp.procs[name].stdout.flush()


def test_stagger_starts_one_run_per_interval(tmp_path):
    clock = [0.0]
    slept = []
    lz, sp, _ = make(tmp_path, [job("a"), job("b")], stagger_s=20.0, now=lambda: clock[0])
    lz.sleep = slept.append
    assert lz.step() is True and set(sp.procs) == {"a"}
    clock[0] = 19.0
    lz.step()
    assert set(sp.procs) == {"a"} and lz.jobs[1].state == "queued"
    assert lz.next_sleep() == pytest.approx(1.0)  # wakes when the stagger interval ends, not after a full poll
    clock[0] = 20.0
    lz.step()
    assert set(sp.procs) == {"a", "b"}


def test_rate_limited_start_is_requeued_without_charge(tmp_path):
    clock = [0.0]
    lz, sp, lines = make(tmp_path, [job("a")], budget=10.0, spent=1.0, projector=lambda j: 3.0,
                         actual_cost=lambda j: None, now=lambda: clock[0], rate_limit_backoff_s=60.0)
    lz.step()
    _write(sp, "a", "run_id a\n" + RATE_LIMIT)
    finish(sp, "a", code=1)
    first = sp.procs["a"]
    assert lz.step() is True  # still something to wait for: the retry
    a = lz.jobs[0]
    assert a.state == "queued" and a.attempts == 1 and a.cost is None
    assert lz.committed() == pytest.approx(1.0)  # no phantom projection
    clock[0] = 59.0
    lz.step()
    assert sp.procs["a"] is first  # not before the backoff
    clock[0] = 61.0
    lz.step()
    assert sp.procs["a"] is not first and a.state == "running"
    finish(sp, "a", code=0)
    lz.step()
    assert a.state == "ok" and any("rate limit" in x for x in lines)


def test_rate_limit_after_app_created_is_a_real_failure(tmp_path):
    lz, sp, _ = make(tmp_path, [job("a")], budget=10.0, spent=1.0, projector=lambda j: 3.0,
                     actual_cost=lambda j: None)
    lz.step()
    _write(sp, "a", "└── ✓ Created Function train.\n" + RATE_LIMIT)
    finish(sp, "a", code=1)
    lz.step()
    assert lz.jobs[0].state == "failed" and lz.jobs[0].cost == pytest.approx(3.0)


def test_rate_limit_gives_up_after_max_retries_without_charge(tmp_path):
    clock = [0.0]
    lz, sp, _ = make(tmp_path, [job("a")], budget=10.0, spent=1.0, projector=lambda j: 3.0,
                     actual_cost=lambda j: None, now=lambda: clock[0], rate_limit_retries=2,
                     rate_limit_backoff_s=1.0)
    a = lz.jobs[0]
    for _ in range(10):
        lz.step()
        if a.state == "running":
            _write(sp, "a", RATE_LIMIT)
            finish(sp, "a", code=1)
        clock[0] += 100.0
        if a.state == "failed":
            break
    assert a.state == "failed" and a.cost == 0.0 and "rate limit" in a.reason
