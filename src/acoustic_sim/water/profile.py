"""Time per stage of the harbor picture (``water/stream.py``), for the compute budget.

``PROFILE.stage(name)`` is a context manager around a stage of the chain.  Off (the default) it does nothing.  On
(:meth:`Profiler.start`), it waits for the GPU before and after the stage, so each stage's time is its own and not the
queued work of the one before; it also records the GPU memory pool's high water mark and, with ``transfers=True``,
the time and bytes of every host to device and device to host copy made through CuPy.  The waits change how the GPU
overlaps work, so a profiled run is a little slower than an unprofiled one: the profile says where the time goes, the
chain's own ``compute_per_second`` says how much there is.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class Profiler:
    def __init__(self) -> None:
        self.enabled = False
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.mem_high_bytes = 0
        self.transfer = {"h2d_s": 0.0, "h2d_bytes": 0, "h2d_calls": 0, "d2h_s": 0.0, "d2h_bytes": 0, "d2h_calls": 0}
        self._sync = None
        self._pool = None
        self._patched = None
        self._depth = 0
        self._extra: list = []

    def start(self, transfers: bool = False) -> None:
        """Clear the counters and switch the profile on."""
        self.__init__()
        self.enabled = True
        try:
            import cupy as cp

            cp.cuda.runtime.getDeviceCount()
            self._sync = cp.cuda.Device().synchronize
            self._pool = cp.get_default_memory_pool()
            if transfers:
                self._patch(cp)
        except Exception:                                    # no GPU: wall time of the host alone
            self._sync = None

    def stop(self) -> dict:
        """Switch the profile off (restoring CuPy's copies) and return what it measured."""
        if self._patched is not None:
            import cupy as cp

            cp.asarray, cp.asnumpy = self._patched[:2]
            if self._patched[2] is not None:
                cp.ndarray.get = self._patched[2]
            for obj, name, fn in self._extra:
                setattr(obj, name, fn)
            self._patched = None
        self.enabled = False
        return self.report()

    def report(self) -> dict:
        return {"seconds": dict(self.seconds), "calls": dict(self.calls), "gpu_memory_high_bytes": self.mem_high_bytes,
                "transfer": dict(self.transfer)}

    def _wait(self) -> None:
        if self._sync is not None:
            self._sync()

    def _mem(self) -> None:
        if self._pool is not None:
            self.mem_high_bytes = max(self.mem_high_bytes, int(self._pool.total_bytes()))

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        self._wait()
        t0 = time.perf_counter()
        self._depth += 1
        try:
            yield
        finally:
            self._wait()
            self._depth -= 1
            self.seconds[name] += time.perf_counter() - t0
            self.calls[name] += 1
            self._mem()

    def begin(self, name: str) -> None:
        """Start the stage *name* (``end`` closes it): the same as ``stage`` without a block."""
        if self.enabled:
            self._wait()
            self._open = getattr(self, "_open", {})
            self._open[name] = time.perf_counter()

    def end(self, name: str) -> None:
        if self.enabled and name in getattr(self, "_open", {}):
            self._wait()
            self.seconds[name] += time.perf_counter() - self._open.pop(name)
            self.calls[name] += 1
            self._mem()

    def _patch(self, cp) -> None:
        import numpy as np

        asarray, asnumpy, get = cp.asarray, cp.asnumpy, cp.ndarray.get
        prof = self

        def timed(kind, fn, nbytes):
            def wrapper(*a, **k):
                prof._wait()
                t0 = time.perf_counter()
                out = fn(*a, **k)
                prof._wait()
                prof.transfer[f"{kind}_s"] += time.perf_counter() - t0
                prof.transfer[f"{kind}_bytes"] += nbytes(a, out)
                prof.transfer[f"{kind}_calls"] += 1
                return out
            return wrapper

        def h2d(a, k=None, **kw):
            if isinstance(a, np.ndarray):
                return timed("h2d", asarray, lambda args, out: int(getattr(out, "nbytes", 0)))(a, k, **kw) if k is not None \
                    else timed("h2d", asarray, lambda args, out: int(getattr(out, "nbytes", 0)))(a, **kw)
            return asarray(a, k, **kw) if k is not None else asarray(a, **kw)

        def d2h(a, *args, **kw):
            if isinstance(a, cp.ndarray):
                return timed("d2h", asnumpy, lambda args_, out: int(getattr(out, "nbytes", 0)))(a, *args, **kw)
            return asnumpy(a, *args, **kw)

        def get_(self_, *args, **kw):
            return timed("d2h", get, lambda args_, out: int(getattr(out, "nbytes", 0)))(self_, *args, **kw)

        cp.asarray, cp.asnumpy = h2d, d2h
        from acoustic_sim.water import bistatic, device

        cls = next(c for c in vars(device).values() if isinstance(c, type) and "asnumpy" in vars(c))
        self._extra = [(cls, "asnumpy", cls.asnumpy), (bistatic, "_asnumpy", bistatic._asnumpy)]
        cls.asnumpy = lambda self_, a: d2h(a) if isinstance(a, cp.ndarray) else np.asarray(a)
        bistatic._asnumpy = lambda a: d2h(a) if isinstance(a, cp.ndarray) else np.asarray(a)
        try:
            cp.ndarray.get = get_
        except (TypeError, AttributeError):                  # a compiled type: its .get() copies are not counted
            get = None
        self._patched = (asarray, asnumpy, get)


def reference_ms(repeats: int = 5) -> float:
    """The time [ms] of a fixed workload on this machine's CPU (the median of *repeats* runs of a complex FFT of 2^18
    points and a 256 x 256 matrix product): the unit the per frame budgets of the tiny scenario are scaled by, so a
    slower machine gets a proportionally larger budget."""
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.standard_normal(2 ** 18) + 1j * rng.standard_normal(2 ** 18)
    a = rng.standard_normal((256, 256))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.fft.ifft(np.fft.fft(x))
        a @ a
        times.append(1e3 * (time.perf_counter() - t0))
    return float(np.median(times))


def over_budget(report: dict, frames: int, budget: dict, scale: float = 1.0, floor_ms: float = 1.0) -> dict:
    """The stages of *report* (``Profiler.report``) whose time per frame exceeds their budget: *budget* maps a stage to
    its milliseconds per frame, each scaled by *scale* and at least *floor_ms*.  A stage without a budget is not
    checked; a budgeted stage that did not run is not an excess."""
    out = {}
    for name, ms in budget.items():
        got = 1e3 * report["seconds"].get(name, 0.0) / max(frames, 1)
        lim = max(ms * scale, floor_ms)
        if got > lim:
            out[name] = {"ms_per_frame": got, "budget_ms": lim}
    return out


PROFILE = Profiler()
