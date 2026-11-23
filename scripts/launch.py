"""Launch several configs as parallel `run.py` processes; safe to leave running unattended (rulings R36, R40, R41).

    uv run python scripts/launch.py <config.yaml>... [--max-parallel 6] [--budget USD --spent USD]
                                    [--smoke] [--logs DIR] [--prepare-only] [--poll 30]

1. One CPU status look-up per config. Then, only if missing, the shared state is built ONCE, sequentially: the general
   (Tulu) pool (CPU) when any config needs it, and the base row + base KL cache (eval GPU) per smoke flag in use (R36).
2. Configs are started IN THE ORDER GIVEN, at most --max-parallel at a time, each as
   `scripts/run.py <config> --require-prepared` logging to <logs>/<run_id>.log (append mode; a header per launch).
   Before a config starts:
   * nothing to do (trained and every checkpoint scored)      -> "done already", never started;
   * a dependency is missing -- the init_from source run (ckpts/<run_id>/ckpts.json) or, for a teacher_sft config,
     the teacher data set (TEACHER_DATA_DONE, written by the method "teacher_data" job, which is budgeted like any
     run) -> waits while its producer is queued/running in this launch, otherwise "deferred": nothing starts;
     re-run launch.py later to pick it up;
   * another launch on this machine (or an orphaned child) holds its lock -> "skipped (already running)", and
     its projected cost stays COMMITTED, since --spent from billing does not include its unfinished work;
   * --budget: baseline --spent + cost of runs finished in this launch (measured from their results rows and
     train_times, priced as scripts/cost_report.py does; a failed run: max(measured, projected)) + projected cost
     of runs in flight + this run's projection must be <= --budget. Otherwise it waits while runs are in flight
     (their measured cost may free headroom) and is finally reported "not launched (budget)". With --budget, a
     config whose method has no cost model is "not launched (no cost model)".
   Every started run gets a wall-clock timeout of max(--min-timeout-h, --timeout-factor x projected wall time);
   past it the child is interrupted/killed, marked failed and charged max(measured, projected). A failed spawn
   releases the lock and is listed as failed. Modal errors while polling (dependency checks, cost measurement)
   never crash the loop: a failed check waits / defers, a failed measurement charges the projection, and any
   other loop error stops every child and still prints the summary.
3. Locks: outputs/locks/<run_id>.lock (flock), inherited by the run.py child, so the lock lives as long as either
   the launcher or the child. Locks are per machine: launch each grid from ONE machine.
   Spawns are staggered (--stagger, default 20 s between starts) and a child that dies on Modal's "App create rate
   limit exceeded" BEFORE its app was created (no "Created Function" in this attempt's log, so nothing ran) is
   requeued with backoff instead of failed, and is never charged; after --rate-limit-retries it is failed at $0 (R50).
4. Stopping: SIGINT/SIGTERM/SIGHUP to the launcher sends SIGINT to every run.py child (they exit their Modal app,
   which stops its containers), waits up to 60 s, then kills them. If the launcher itself is SIGKILLed, the
   children keep running and keep their locks: a re-run of launch.py reports them "skipped (already running)"
   instead of starting a second copy; stop them with `modal app stop` / kill if needed.
Exit code: 0 if every started run exited 0 and nothing was deferred/not launched, else 1 (all listed at the end).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # modal_app.py at the repo root
sys.path.insert(0, str(ROOT / "scripts"))

from rlrazor.config import RunConfig, config_to_yaml, load_config, run_id  # noqa: E402
from rlrazor.pipeline import needs_general_pool  # noqa: E402

GRACE_S = 60


def prepare_plan(cfgs, statuses: list[dict]) -> dict:
    """Shared work still missing: {"pool": bool, "base": sorted smoke flags whose base row is missing}."""
    pool = any(needs_general_pool(c) and not st["general_pool"] for c, st in zip(cfgs, statuses))
    base = sorted({c.smoke for c, st in zip(cfgs, statuses) if st["base_row"] is None})
    return {"pool": pool, "base": base}


def source_run_id(cfg: RunConfig) -> str | None:
    """init_from is "<source run_id>/step-<n>" or "base"/None (R30)."""
    return None if cfg.init_from in (None, "base") else str(cfg.init_from).split("/")[0]


# The teacher data job (method "teacher_data") must write this Volume file when its data set is complete;
# every "teacher_sft" config depends on it (R44). Other dependencies are init_from sources (ckpts/<run_id>/ckpts.json).
TEACHER_DATA = "teacher_data"
TEACHER_DATA_DONE = "cache/teacher_data/done.json"


def dependencies(cfg: RunConfig) -> list[str]:
    """What must exist before this config trains: its init_from source run_id and, for teacher_sft, TEACHER_DATA."""
    deps = [d for d in [source_run_id(cfg)] if d]
    if cfg.method == "teacher_sft":
        deps.append(TEACHER_DATA)
    return deps


@dataclasses.dataclass
class Job:
    path: str
    cfg: RunConfig
    rid: str
    status: dict  # run_status at launch time
    state: str = "queued"  # queued | running | ok | failed | done already | deferred | skipped | not launched
    reason: str = ""
    projected: float = 0.0
    cost: float | None = None  # charged against the budget once finished
    proc: object = None
    lock_fd: int | None = None
    log: object = None
    deadline: float | None = None  # monotonic time after which the child is killed (wall-clock timeout)
    attempts: int = 0  # starts lost to Modal's app-creation rate limit (R50)
    not_before: float | None = None  # monotonic time before which a rate-limited job is not restarted
    log_start: int = 0  # byte offset of this attempt in its log

    @property
    def trained(self) -> bool:
        return self.status.get("index") is not None

    @property
    def complete(self) -> bool:
        idx = self.status.get("index")
        return idx is not None and len(self.status.get("scored_steps", [])) == len(idx["ckpts"])

    def produces(self, dep: str) -> bool:
        return self.rid == dep or (dep == TEACHER_DATA and self.cfg.method == TEACHER_DATA)


class Launcher:
    """The scheduling loop, with every Modal / process interaction injected (tests use fakes).

    ready(dep) -> bool: is a dependency (a run_id or TEACHER_DATA) available on the Volume.
    projector(job) -> $ and wall_projector(job) -> seconds: projections (cost_report unit costs).
    actual_cost(job) -> $ | None: measured cost of a finished run. Any exception from these is contained.
    """

    def __init__(self, jobs: list[Job], *, logs: Path, locks: Path, max_parallel: int = 6,
                 budget: float | None = None, spent: float = 0.0, smoke: bool = False,
                 projector: Callable[[Job], float] | None = None,
                 wall_projector: Callable[[Job], float] | None = None,
                 timeout_factor: float = 3.0, min_timeout_s: float = 3600.0,
                 ready: Callable[[str], bool] = lambda dep: False,
                 actual_cost: Callable[[Job], float | None] = lambda job: None,
                 spawn: Callable = subprocess.Popen, sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], float] = time.monotonic, poll_s: float = 30.0,
                 out: Callable[[str], None] = print, stagger_s: float = 0.0, rate_limit_retries: int = 5,
                 rate_limit_backoff_s: float = 60.0):
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        self.jobs, self.logs, self.locks = jobs, Path(logs), Path(locks)
        self.max_parallel, self.budget, self.spent, self.smoke = max_parallel, budget, spent, smoke
        self.projector, self.wall_projector = projector, wall_projector
        self.timeout_factor, self.min_timeout_s = timeout_factor, min_timeout_s
        self.ready, self.actual_cost = ready, actual_cost
        self.spawn, self.sleep, self.now, self.poll_s, self.out = spawn, sleep, now, poll_s, out
        self.stopping = False
        self.stagger_s, self.rate_limit_retries, self.rate_limit_backoff_s = (stagger_s, rate_limit_retries,
                                                                              rate_limit_backoff_s)
        self.last_start: float | None = None

    # ---- bookkeeping
    def running(self) -> list[Job]:
        return [j for j in self.jobs if j.state == "running"]

    def committed(self) -> float:
        """Baseline + finished runs (measured or projected) + runs in flight (projected) + runs that another
        launch / an orphaned child is running (projected: --spent from billing does not include their rest)."""
        return (self.spent + sum(j.cost or 0.0 for j in self.jobs if j.state in ("ok", "failed"))
                + sum(j.projected for j in self.jobs if j.state in ("running", "skipped")))

    # ---- locks and logs
    def _lock(self, job: Job) -> int | None:
        self.locks.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.locks / f"{job.rid}.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        return fd

    def _release(self, job: Job) -> None:
        if job.lock_fd is not None:
            os.close(job.lock_fd)  # the lock itself is freed once the child's inherited copy is closed too
            job.lock_fd = None
        if job.log is not None:
            job.log.close()
            job.log = None

    def _timeout_s(self, job: Job) -> float | None:
        if self.wall_projector is None:
            return None
        try:
            return max(self.min_timeout_s, self.timeout_factor * self.wall_projector(job))
        except Exception as e:  # noqa: BLE001  no projection -> no timeout, but say so
            self.out(f"{job.rid}: no wall-time projection ({e}); no timeout")
            return None

    def _start(self, job: Job) -> bool:
        cmd = [sys.executable, str(ROOT / "scripts" / "run.py"), job.path, "--require-prepared"]
        if self.smoke:
            cmd.append("--smoke")
        timeout = self._timeout_s(job)
        try:
            self.logs.mkdir(parents=True, exist_ok=True)
            job.log = open(self.logs / f"{job.rid}.log", "a")
            stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            job.log.write(f"\n===== launch.py {stamp} pid {os.getpid()} projected ${job.projected:.2f} timeout "
                          f"{'none' if timeout is None else f'{timeout:.0f} s'}: {shlex.join(cmd)}\n")
            job.log.flush()
            job.log_start = job.log.tell()
            job.proc = self.spawn(cmd, stdout=job.log, stderr=subprocess.STDOUT, cwd=ROOT,
                                  pass_fds=(job.lock_fd,))
        except Exception as e:  # noqa: BLE001  never crash the loop or keep the lock on a failed spawn
            self._release(job)
            job.state, job.reason, job.cost = "failed", f"spawn failed: {e!r}", 0.0
            self.out(f"{job.rid}: spawn failed: {e!r}")
            return False
        job.deadline = None if timeout is None else self.now() + timeout
        job.state = "running"
        self.last_start = self.now()
        self.out(f"started {job.rid} (projected ${job.projected:.2f}, committed ${self.committed():.2f})")
        return True

    def _deps_missing(self, job: Job) -> list[str] | None:
        """Unmet dependencies, or None if a check failed (Modal error): then the job waits / is deferred."""
        if job.trained and job.cfg.method != TEACHER_DATA:
            return []  # already trained: only evals left, no dependency
        missing = []
        for dep in dependencies(job.cfg):
            try:
                ok = bool(self.ready(dep))
            except Exception as e:  # noqa: BLE001
                job.reason = f"dependency check for {dep} failed: {e!r}"
                return None
            if not ok:
                missing.append(dep)
        return missing

    # ---- one decision for a queued job: True if it was started
    def _consider(self, job: Job) -> bool:
        if job.complete:
            job.state, job.reason = "done already", "trained and every checkpoint scored"
            return False
        if job.cfg.method == TEACHER_DATA:
            try:
                if self.ready(TEACHER_DATA):
                    job.state, job.reason = "done already", f"{TEACHER_DATA_DONE} exists"
                    return False
            except Exception as e:  # noqa: BLE001
                job.reason = f"teacher data check failed: {e!r}"
                return False  # stays queued; deferred if nothing else is running
        missing = self._deps_missing(job)
        if missing is None:
            return False  # check failed: wait (deferred at the end if nothing is in flight)
        for dep in missing:
            if any(j is not job and j.produces(dep) and j.state in ("queued", "running") for j in self.jobs):
                job.reason = f"waiting for {dep}"
                return False  # its source is part of this launch: wait for it
            job.state, job.reason = "deferred", f"{dep} not available; re-run launch.py later"
            return False
        if self.budget is not None:
            if self.projector is None:
                raise RuntimeError("--budget needs a projector")
            try:
                job.projected = self.projector(job)
            except Exception as e:  # noqa: BLE001  cost_report.NoCostModel or a broken projection
                job.state, job.reason = "not launched", f"no cost model ({e})"
                return False
        fd = self._lock(job)
        if fd is None:  # charged at its projection in committed() (N1)
            job.state, job.reason = "skipped", "already running (lock held by another launch/child)"
            return False
        if self.budget is not None and self.committed() + job.projected > self.budget + 1e-9:
            os.close(fd)
            if not self.running():
                job.state = "not launched"
                job.reason = f"budget: committed ${self.committed():.2f} + ${job.projected:.2f} > ${self.budget:.2f}"
            return False  # else wait: finishing runs may free headroom
        job.lock_fd = fd
        return self._start(job)

    def _measure(self, job: Job) -> float | None:
        try:
            return self.actual_cost(job)
        except Exception as e:  # noqa: BLE001  (N2) charge the projection instead
            self.out(f"{job.rid}: measuring its cost failed ({e!r}); charging the projection")
            return None

    def _finish(self, job: Job, code: int, reason: str) -> None:
        job.state = "ok" if code == 0 and not reason.startswith("timeout") else "failed"
        job.reason = reason
        measured = self._measure(job)
        if job.state == "ok":
            job.cost = job.projected if measured is None else measured
        else:  # a failed or killed run may have spent more than it recorded
            job.cost = max(measured or 0.0, job.projected)
        self._release(job)
        self.out(f"finished {job.rid}: {reason}, charged ${job.cost:.2f}, committed ${self.committed():.2f}")

    RATE_LIMIT_MARK = "App create rate limit exceeded"
    APP_CREATED_MARK = "Created Function"

    def _rate_limited(self, job: Job) -> bool:
        """This attempt died on Modal's app-creation rate limit before its app existed (so no container ran)."""
        try:
            with open(self.logs / f"{job.rid}.log", "rb") as f:
                f.seek(job.log_start)
                text = f.read().decode("utf-8", "replace")
        except OSError:
            return False
        return self.RATE_LIMIT_MARK in text and self.APP_CREATED_MARK not in text

    def _requeue(self, job: Job) -> None:
        self._release(job)
        job.attempts += 1
        job.proc, job.deadline = None, None
        if job.attempts > self.rate_limit_retries:
            job.state, job.cost = "failed", 0.0
            job.reason = f"Modal app-creation rate limit on {job.attempts} attempts; nothing ran (charged $0)"
        else:
            job.state, job.cost = "queued", None
            job.not_before = self.now() + self.rate_limit_backoff_s * job.attempts
            job.reason = f"Modal app-creation rate limit (attempt {job.attempts}); retrying"
        self.out(f"{job.rid}: {job.reason}; committed ${self.committed():.2f}")

    def _kill(self, job: Job, grace: float) -> None:
        try:
            job.proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            job.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            job.proc.kill()
            job.proc.wait()

    def _reap(self) -> None:
        for job in self.running():
            code = job.proc.poll()
            if code is None and job.deadline is not None and self.now() > job.deadline:
                self.out(f"{job.rid}: wall-clock timeout, stopping it")
                self._kill(job, GRACE_S)
                self._finish(job, job.proc.poll() if job.proc.poll() is not None else -9,
                             "timeout (wall-clock limit exceeded; killed)")
            elif code is not None:
                if code != 0 and self._rate_limited(job):
                    self._requeue(job)
                else:
                    self._finish(job, code, f"exit {code}")

    def step(self) -> bool:
        """One scheduling pass. Returns False when nothing is left to wait for."""
        self._reap()
        if self.stopping:
            return False
        for job in [j for j in self.jobs if j.state == "queued"]:
            if len(self.running()) >= self.max_parallel or self._stagger_wait() > 0:
                break
            if job.not_before is not None and self.now() < job.not_before:
                continue  # rate-limit backoff
            self._consider(job)
        if self.running() or self._time_blocked():
            return True
        # Nothing in flight and nothing could start: the remaining queued jobs can never start in this launch.
        for job in [j for j in self.jobs if j.state == "queued"]:
            job.state = "deferred" if dependencies(job.cfg) or job.reason else "not launched"
            job.reason = job.reason or "could not start"
        return False

    def _stagger_wait(self) -> float:
        if self.stagger_s <= 0 or self.last_start is None:
            return 0.0
        return max(0.0, self.last_start + self.stagger_s - self.now())

    def _time_blocked(self) -> bool:
        """A queued job waits only on time (stagger interval or rate-limit backoff)."""
        queued = [j for j in self.jobs if j.state == "queued"]
        return bool(queued) and (self._stagger_wait() > 0 or any(
            j.not_before is not None and self.now() < j.not_before for j in queued))

    def next_sleep(self) -> float:
        """Seconds until the next pass: the poll interval, or sooner when a stagger/backoff wait ends."""
        waits = [self.poll_s]
        queued = [j for j in self.jobs if j.state == "queued"]
        if queued and self._stagger_wait() > 0:
            waits.append(self._stagger_wait())
        waits += [j.not_before - self.now() for j in queued if j.not_before is not None and j.not_before > self.now()]
        return max(0.0, min(waits))

    def stop(self, signum=None, frame=None) -> None:
        """Signal handler (and error path): stop launching, interrupt every child, kill stragglers after GRACE_S."""
        if self.stopping:
            return
        self.stopping = True
        self.out(f"launch.py: stopping ({'signal ' + str(signum) if signum else 'error'}): "
                 f"interrupting {len(self.running())} run(s)")
        for job in self.running():
            try:
                job.proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        deadline = time.time() + GRACE_S
        for job in self.running():
            try:
                job.proc.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                job.proc.kill()
                job.proc.wait()
        self._reap()

    def run(self) -> int:
        error = None
        try:
            while self.step():
                self.sleep(self.next_sleep())
        except BaseException as e:  # noqa: BLE001  (N2) never leave children spending unobserved
            error = e
            self.out(f"launch.py: scheduling loop failed: {e!r}")
        finally:
            if error is not None or self.running():
                self.stop()
            code = self.summary()
        return 1 if error is not None else code

    def summary(self) -> int:
        self.out(f"summary (committed ${self.committed():.2f}"
                 + (f" of budget ${self.budget:.2f})" if self.budget is not None else ")"))
        for job in self.jobs:
            self.out(f"  {job.state:13s} {job.rid:44s} {job.reason}")
        bad = [j for j in self.jobs if j.state not in ("ok", "done already")]
        return 1 if (bad or self.stopping) else 0


# ---------------------------------------------------------------- Modal wiring

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--smoke", action="store_true", help="force smoke: true for every config")
    ap.add_argument("--logs", type=Path, default=ROOT / "outputs" / "logs",
                    help="per-run log directory (default: outputs/logs/, git-ignored)")
    ap.add_argument("--locks", type=Path, default=ROOT / "outputs" / "locks", help="per-run lock files")
    ap.add_argument("--max-parallel", type=int, default=6, help="run.py processes at once (default 6)")
    ap.add_argument("--budget", type=float, help="USD ceiling for spent + committed (see above); no guard if unset")
    ap.add_argument("--spent", type=float, default=0.0,
                    help="USD already spent before this launch (from `modal billing report`)")
    ap.add_argument("--poll", type=float, default=30.0, help="seconds between scheduling passes")
    ap.add_argument("--stagger", type=float, default=20.0,
                    help="seconds between run.py spawns (Modal app-creation rate limit; default 20)")
    ap.add_argument("--rate-limit-retries", type=int, default=5,
                    help="restarts of a run lost to the app-creation rate limit before it is failed at $0")
    ap.add_argument("--timeout-factor", type=float, default=3.0,
                    help="kill a run after this many times its projected wall time (default 3)")
    ap.add_argument("--min-timeout-h", type=float, default=1.0, help="floor of the per-run timeout (default 1 h)")
    ap.add_argument("--prepare-only", action="store_true", help="build the shared state, launch nothing")
    a = ap.parse_args(argv)

    cfgs = [load_config(p) for p in a.configs]
    if a.smoke:
        cfgs = [dataclasses.replace(c, smoke=True) for c in cfgs]
    rids = [run_id(c) for c in cfgs]
    if len(set(rids)) != len(rids):
        raise SystemExit(f"duplicate run_ids: {rids}")

    import modal

    import cost_report as cr
    from modal_app import (EVAL_GPU, EVAL_MAX_CONTAINERS, TEACHER_GPU, app, base_eval, build_general_pool,
                           export_results, read_files, run_status)

    with modal.enable_output(), app.run():
        statuses = [run_status.remote(config_to_yaml(c)) for c in cfgs]
        plan = prepare_plan(cfgs, statuses)
        print(f"shared state to build: {plan}")
        if plan["pool"]:
            print(f"general pool: {build_general_pool.remote()} rows")
        for smoke in plan["base"]:
            row = base_eval.remote(smoke=smoke)
            print(f"base row (smoke={smoke}): gsm8k_acc {row['gsm8k_acc']:.4f}")
        if a.prepare_only:
            return 0
        statuses = [run_status.remote(config_to_yaml(c)) for c in cfgs]
        jobs = [Job(p, c, r, st) for p, c, r, st in zip(a.configs, cfgs, rids, statuses)]

        prices, units = cr.Prices.from_env(), None
        try:  # unit costs from the measured rows (pilot): needed by --budget and by the wall-clock timeouts
            export_results.remote()
            files = read_files.remote(["results.jsonl", "train_times.jsonl"])
            parse = lambda t: [json.loads(x) for x in t.splitlines() if x.strip()]  # noqa: E731
            units = cr.measure(parse(files.get("results.jsonl", "")), parse(files.get("train_times.jsonl", "")))
        except (Exception, SystemExit) as e:  # noqa: BLE001
            if a.budget is not None:
                raise SystemExit(f"--budget needs measured unit costs: {e}")
            print(f"no unit costs ({e}): no wall-clock timeouts")
        if a.budget is not None:
            print(f"budget guard: ${a.budget:.2f}, baseline spent ${a.spent:.2f}; units {units}")

        def projector(job: Job) -> float:
            idx = job.status.get("index")
            n_evals = None if idx is None else len(idx["ckpts"]) - len(job.status.get("scored_steps", []))
            return cr.run_cost(job.cfg, units, prices, train=idx is None, n_evals=n_evals)[0]

        def wall_projector(job: Job) -> float:
            """Projected wall seconds: training container, then the evals one after another (modal_app caps
            eval_ckpt at EVAL_MAX_CONTAINERS per run, default 1, to stay under the workspace GPU limit)."""
            idx = job.status.get("index")
            c = dataclasses.asdict(job.cfg)
            train = 0.0
            if idx is None:
                train = prices.container_s + cr.train_seconds(
                    c["method"], units, prices, epochs=c["epochs"], grpo_steps=c["grpo_steps"],
                    rounds=c["rounds"], mix_f=c["mix_f"])[0]
            if c["method"] == TEACHER_DATA:
                return train
            if c["wise_alpha"]:
                train = prices.container_s + prices.wise_cpu_s if idx is None else 0.0
            n = (1 if c["smoke"] or c["wise_alpha"] else c["rounds"] if c["method"] == "rft_iter" else c["n_ckpts"])
            if idx is not None:
                n = len(idx["ckpts"]) - len(job.status.get("scored_steps", []))
            waves = -(-max(n, 0) // max(1, EVAL_MAX_CONTAINERS))
            return train + waves * (prices.container_s + units.eval_s)

        def ready(dep: str) -> bool:
            path = TEACHER_DATA_DONE if dep == TEACHER_DATA else f"ckpts/{dep}/ckpts.json"
            return bool(read_files.remote([path]))

        def actual_cost(job: Job) -> float | None:
            st = run_status.remote(config_to_yaml(job.cfg))
            new_steps = [s for s in st["scored_steps"] if s not in job.status.get("scored_steps", [])]
            paths = [f"results/{job.rid}__step{s}.json" for s in new_steps] + [f"cache/{job.rid}/train_time.json"]
            files = read_files.remote(paths)
            usd, seen = 0.0, False
            # The teacher data job runs on TEACHER_GPU (default: the eval GPU class), so price it at that rate.
            train_usd = (prices.eval_usd if job.cfg.method == TEACHER_DATA and TEACHER_GPU == EVAL_GPU
                         else prices.train_usd)
            for path, text in files.items():
                if path.startswith("results/"):
                    usd, seen = usd + prices.eval_usd(json.loads(text)["gpu_seconds"]), True
                elif not job.trained:
                    usd, seen = usd + train_usd(json.loads(text)["wall_seconds"]), True
            return usd if seen else None

        launcher = Launcher(jobs, logs=a.logs, locks=a.locks, max_parallel=a.max_parallel, budget=a.budget,
                            spent=a.spent, smoke=a.smoke, projector=projector if units else None,
                            wall_projector=wall_projector if units else None, timeout_factor=a.timeout_factor,
                            min_timeout_s=a.min_timeout_h * 3600, ready=ready, actual_cost=actual_cost,
                            poll_s=a.poll, stagger_s=a.stagger, rate_limit_retries=a.rate_limit_retries)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, launcher.stop)
        return launcher.run()


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if not isinstance(e.code, int) and e.code is not None:
            print(e.code, file=sys.stderr)
    except BaseException:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
