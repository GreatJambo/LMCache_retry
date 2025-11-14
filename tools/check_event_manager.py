#!/usr/bin/env python3
"""Lightweight, dependency-free checks for EventManager.

Run: python tools/check_event_manager.py

Avoids repo test fixtures to sidestep GPU/C-ops imports.
"""
# Standard
import asyncio
import sys
import threading

# First Party
from pathlib import Path

# Ensure repo root is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
if str(REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(REPO_ROOT))

from lmcache.v1.event_manager import EventManager, EventStatus, EventType


def check_add_update_pop_success():
    mgr = EventManager()
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        fut = loop.create_future()
        mgr.add_event(EventType.LOADING, "e1", fut)

        fut.set_result({"ok": True})
        mgr.update_event_status(EventType.LOADING, "e1", EventStatus.DONE)
        got = mgr.pop_event(EventType.LOADING, "e1")
        assert got is fut and got.result() == {"ok": True}
        assert mgr.get_event_status(EventType.LOADING, "e1") == EventStatus.NOT_FOUND
    finally:
        loop.close()


def check_get_status_not_found():
    mgr = EventManager()
    assert mgr.get_event_status(EventType.LOADING, "missing") == EventStatus.NOT_FOUND


def check_update_missing_raises():
    mgr = EventManager()
    try:
        mgr.update_event_status(EventType.LOADING, "missing", EventStatus.DONE)
    except KeyError:
        return
    raise AssertionError("update_event_status should raise on missing id")


def check_thread_safety_add_many():
    mgr = EventManager()
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        n = 32
        futs = [loop.create_future() for _ in range(n)]

        def worker(i: int):
            mgr.add_event(EventType.LOADING, f"id-{i}", futs[i])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i, f in enumerate(futs):
            f.set_result(i)
            mgr.update_event_status(EventType.LOADING, f"id-{i}", EventStatus.DONE)

        results = []
        for i in range(n):
            got = mgr.pop_event(EventType.LOADING, f"id-{i}")
            results.append(got.result())
        assert results == list(range(n))
    finally:
        loop.close()


def main():
    checks = [
        ("add/update/pop", check_add_update_pop_success),
        ("status NOT_FOUND", check_get_status_not_found),
        ("update missing raises", check_update_missing_raises),
        ("thread-safety add many", check_thread_safety_add_many),
    ]
    failed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"[OK] {name}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
