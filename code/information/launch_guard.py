"""Exclusive launcher for long experiments: one instance per tag.

Why: on 2026-10-19 a watchdog-launched control and a manual launch of the same tag ran
together (see docs/INCIDENT_duplicate_control_2026-10-19.md), using 96% GPU and leaving
82 MB of system RAM. This guard makes that impossible:

  1. refuse if a live process whose command line contains the tag already exists;
  2. take an atomic lock directory (results/.locks/<tag>.lock); refuse if it exists;
  3. refuse if free RAM is below --min-ram-gb;
  4. create the log directory before starting, tee the child's output, and release the
     lock on exit (including SIGINT/SIGTERM).

Usage:
    python code/information/launch_guard.py --tag SH_ctrl_blocksplit_stock_s0 \
        --log results/subject_holdout/control_run.log -- <command...>
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
LOCKS = ROOT / "results" / ".locks"


def _ancestors():
    """Guard + all ancestors: the shell command string that launched us may contain the
    tag, and it must not count as a duplicate instance."""
    pids = {os.getpid()}
    try:
        for p in psutil.Process(os.getpid()).parents():
            pids.add(p.pid)
    except psutil.Error:
        pass
    return pids


def live_tag_processes(tag):
    skip = _ancestors()
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if tag in cmd and p.pid not in skip:
            out.append((p.pid, cmd[:200]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--min-ram-gb", type=float, default=1.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command after --")

    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    LOCKS.mkdir(parents=True, exist_ok=True)
    lock = LOCKS / f"{a.tag}.lock"

    # 1) process preflight (the lock is the primary guard; this catches stale/orphan runs)
    procs = live_tag_processes(a.tag)
    if procs:
        print(f"REFUSED: tag '{a.tag}' already has a live process:", file=sys.stderr)
        for pid, c in procs:
            print(f"  PID {pid}: {c}", file=sys.stderr)
        print("If this is stale, verify the PID is gone, then remove the lock manually.",
              file=sys.stderr)
        sys.exit(2)

    # 2) atomic lock
    try:
        lock.mkdir()
    except FileExistsError:
        owner = (lock / "pid")
        pid = owner.read_text().strip() if owner.exists() else "unknown"
        print(f"REFUSED: lock exists ({lock}, owner PID {pid}).", file=sys.stderr)
        print("Check nvidia-smi / the process list before removing it.", file=sys.stderr)
        sys.exit(3)
    (lock / "pid").write_text(str(os.getpid()), encoding="utf-8")

    # 3) RAM preflight
    avail_gb = psutil.virtual_memory().available / 1e9
    if avail_gb < a.min_ram_gb:
        print(f"REFUSED: only {avail_gb:.2f} GB RAM available "
              f"(< {a.min_ram_gb:.1f} GB).", file=sys.stderr)
        shutil.rmtree(lock, ignore_errors=True)
        sys.exit(4)

    child = None

    def release(sig=None, frame=None):
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
        shutil.rmtree(lock, ignore_errors=True)
        if sig is not None:
            sys.exit(128 + sig)

    signal.signal(signal.SIGINT, lambda s, f: release(s, f))
    signal.signal(signal.SIGTERM, lambda s, f: release(s, f))

    print(f"guard: tag={a.tag} pid={os.getpid()} lock={lock} "
          f"ram_available={avail_gb:.1f}GB log={log}", flush=True)
    print("guard: command = " + " ".join(cmd), flush=True)
    t0 = time.time()
    try:
        with open(log, "w", encoding="utf-8", errors="replace") as lf:
            child = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in child.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
                lf.flush()
            rc = child.wait()
    finally:
        release()
    print(f"guard: tag={a.tag} finished rc={rc} in {time.time()-t0:.0f}s", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
