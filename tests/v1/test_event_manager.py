# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.event_manager import (
    EventManager,
    EventStatus,
    EventType,
)


def test_add_update_pop_success():
    mgr = EventManager()
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        fut = loop.create_future()

        mgr.add_event(EventType.LOADING, "e1", fut)

        # Complete the future and mark DONE
        fut.set_result({"ok": True})
        mgr.update_event_status(EventType.LOADING, "e1", EventStatus.DONE)

        got = mgr.pop_event(EventType.LOADING, "e1")
        assert got is fut
        assert got.result() == {"ok": True}

        # After pop, the event should be gone
        assert (
            mgr.get_event_status(EventType.LOADING, "e1") == EventStatus.NOT_FOUND
        )
    finally:
        loop.close()


def test_get_event_status_not_found():
    mgr = EventManager()
    assert mgr.get_event_status(EventType.LOADING, "missing") == EventStatus.NOT_FOUND


def test_update_event_status_missing_raises():
    mgr = EventManager()
    with pytest.raises(KeyError):
        mgr.update_event_status(EventType.LOADING, "missing", EventStatus.DONE)


def test_thread_safety_add_many():
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

        # Resolve and mark DONE
        for i, f in enumerate(futs):
            f.set_result(i)
            mgr.update_event_status(EventType.LOADING, f"id-{i}", EventStatus.DONE)

        # Pop and verify all
        results = []
        for i in range(n):
            got = mgr.pop_event(EventType.LOADING, f"id-{i}")
            results.append(got.result())

        assert results == list(range(n))
    finally:
        loop.close()
