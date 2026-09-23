"""JSONL tracer for standalone generation — off unless ``SYNEVAD_PROFILE`` is set.

Enable with ``--profile`` (or ``SYNEVAD_PROFILE=1`` and ``SYNEVAD_PROFILE_DIR=...``).
Each process writes ``<role>-pid<pid>.jsonl``: one object per span, with ``name``,
``t`` and ``dur_s``, so a run is summarised with whatever reads JSONL.

Optional env:

- ``SYNEVAD_PROFILE_CUDA=1`` — ``cuda.synchronize()`` around spans / attention (accurate
  GPU time, slower).
- ``SYNEVAD_PROFILE_TORCH=1`` — Chrome trace of the first ``edit_image`` in each gen
  worker via ``torch.profiler``.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_KIND_BLOCK = "block"

_role = "unknown"
_gpu: int | None = None
_fh = None
_lock = threading.Lock()
_smi_stop: threading.Event | None = None
_attn_n = 0
_attn_s = 0.0
_attn_cast_s = 0.0
_attn_sdpa_s = 0.0
_attn_chunked = 0
_torch_used = False


def enabled() -> bool:
    raw = os.environ.get("SYNEVAD_PROFILE", "").strip().lower()
    if raw in ("1", "true", "yes", "jsonl", "on"):
        return True
    return bool(os.environ.get("SYNEVAD_PROFILE_DIR", "").strip())


def cuda_sync_enabled() -> bool:
    return os.environ.get("SYNEVAD_PROFILE_CUDA", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def torch_trace_enabled() -> bool:
    return os.environ.get("SYNEVAD_PROFILE_TORCH", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def profile_dir() -> Path | None:
    raw = os.environ.get("SYNEVAD_PROFILE_DIR", "").strip()
    return Path(raw) if raw else None


def configure(role: str, gpu: int | None = None) -> None:
    """Set process identity used in every event. Call once at worker entry."""
    global _role, _gpu
    _role = str(role)
    _gpu = gpu


def activate(directory: Path, *, cuda: bool = False, torch_trace: bool = False) -> Path:
    """Turn profiling on in this process (and children that inherit the env)."""
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    os.environ["SYNEVAD_PROFILE"] = "1"
    os.environ["SYNEVAD_PROFILE_DIR"] = str(dest)
    if cuda:
        os.environ["SYNEVAD_PROFILE_CUDA"] = "1"
    if torch_trace:
        os.environ["SYNEVAD_PROFILE_TORCH"] = "1"
    print(f"[profile] writing {dest}", flush=True)
    return dest


def _maybe_cuda_sync() -> None:
    if not cuda_sync_enabled():
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _file():
    global _fh
    if _fh is not None:
        return _fh
    dest = profile_dir()
    if dest is None:
        return None
    dest.mkdir(parents=True, exist_ok=True)
    gpu = "na" if _gpu is None else str(_gpu)
    path = dest / f"{_role}-pid{os.getpid()}-gpu{gpu}.jsonl"
    _fh = path.open("a", encoding="utf-8")
    return _fh


def emit(name: str, *, dur_ms: float, kind: str = "span", **extra: Any) -> None:
    if not enabled():
        return
    rec = {
        "ts": time.time(),
        "dur_ms": round(float(dur_ms), 3),
        "name": name,
        "kind": kind,
        "role": _role,
        "gpu": _gpu,
        "pid": os.getpid(),
        **{k: v for k, v in extra.items() if v is not None},
    }
    line = json.dumps(rec, default=str)
    with _lock:
        fh = _file()
        if fh is None:
            return
        fh.write(line + "\n")
        fh.flush()
    if kind == _KIND_BLOCK and dur_ms >= 1000:
        print(f"[profile] BLOCK {name} {dur_ms / 1000:.2f}s pid={os.getpid()} {extra}", flush=True)


@contextmanager
def span(name: str, *, kind: str = "span", **extra: Any) -> Iterator[None]:
    if not enabled():
        yield
        return
    _maybe_cuda_sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _maybe_cuda_sync()
        emit(name, dur_ms=(time.perf_counter() - t0) * 1000.0, kind=kind, **extra)


def attn_add(seconds: float, *, chunked: bool = False, part: str = "sdpa") -> None:
    """Accumulate attention wall time; flushed into each ``dit_step`` event.

    ``part`` is ``cast`` (permute + dtype convert) or ``sdpa`` (the kernel).
    """
    global _attn_n, _attn_s, _attn_cast_s, _attn_sdpa_s, _attn_chunked
    if not enabled():
        return
    sec = float(seconds)
    _attn_s += sec
    if part == "cast":
        _attn_cast_s += sec
        return
    _attn_n += 1
    _attn_sdpa_s += sec
    if chunked:
        _attn_chunked += 1


def attn_flush() -> dict[str, Any]:
    global _attn_n, _attn_s, _attn_cast_s, _attn_sdpa_s
    snap = {
        "attn_n": _attn_n,
        "attn_ms": round(_attn_s * 1000.0, 3),
        "attn_cast_ms": round(_attn_cast_s * 1000.0, 3),
        "attn_sdpa_ms": round(_attn_sdpa_s * 1000.0, 3),
        "attn_chunked": _attn_chunked,
    }
    _attn_n = 0
    _attn_s = 0.0
    _attn_cast_s = 0.0
    _attn_sdpa_s = 0.0
    return snap


def cuda_mem() -> dict[str, Any] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "cuda_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        }
    except Exception:
        return None


def maybe_torch_profile():
    """Context manager: torch.profiler Chrome trace on the first use per process, else no-op."""
    global _torch_used

    @contextmanager
    def _noop():
        yield None

    if not enabled() or not torch_trace_enabled() or _torch_used:
        return _noop()
    dest = profile_dir()
    if dest is None:
        return _noop()

    @contextmanager
    def _prof():
        global _torch_used
        import torch

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        dest.mkdir(parents=True, exist_ok=True)
        trace = dest / f"torch-{_role}-pid{os.getpid()}.json"
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            yield prof
            try:
                import torch as _t

                if _t.cuda.is_available():
                    _t.cuda.synchronize()
            except Exception:
                pass
        prof.export_chrome_trace(str(trace))
        avg_path = dest / f"torch-{_role}-pid{os.getpid()}-averages.txt"
        try:
            table = prof.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
                row_limit=80,
            )
        except Exception:
            table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=80)
        avg_path.write_text(table, encoding="utf-8")
        _torch_used = True
        print(f"[profile] torch chrome trace -> {trace}", flush=True)
        print(f"[profile] torch op averages -> {avg_path}", flush=True)
        _print_copy_ops(table)

    return _prof()


def _print_copy_ops(table: str) -> None:
    """Echo aten to/copy rows so dtype conversions are visible without Perfetto."""
    keys = ("aten::to", "aten::_to_copy", "aten::copy_", "aten::copy", "aten::contiguous")
    hits = [ln for ln in table.splitlines() if any(k in ln for k in keys)]
    if not hits:
        print("[profile] torch: no aten::to / copy ops in the top-80 table", flush=True)
        return
    print("[profile] torch dtype/copy ops (from key averages):", flush=True)
    for ln in hits[:30]:
        print(f"  {ln.rstrip()}", flush=True)


def start_smi_sampler(interval_s: float = 1.0) -> None:
    """Background ``nvidia-smi`` samples so GPU util overlap is visible."""
    global _smi_stop
    if not enabled() or _smi_stop is not None:
        return
    dest = profile_dir()
    if dest is None:
        return
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "nvidia-smi.jsonl"
    _smi_stop = threading.Event()
    stop = _smi_stop

    def _loop() -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        while not stop.is_set():
            try:
                out = subprocess.check_output(cmd, text=True, timeout=5)
            except Exception as exc:
                rec = {"ts": time.time(), "error": str(exc)}
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                return
            ts = time.time()
            rows = []
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue
                rows.append(
                    {
                        "index": int(parts[0]),
                        "uuid": parts[1],
                        "name": parts[2],
                        "util_gpu": _num(parts[3]),
                        "util_mem": _num(parts[4]),
                        "mem_used_mb": _num(parts[5]),
                        "mem_total_mb": _num(parts[6]),
                    }
                )
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": ts, "gpus": rows}) + "\n")
            stop.wait(interval_s)

    threading.Thread(target=_loop, name="synevad-smi", daemon=True).start()


def stop_smi_sampler() -> None:
    global _smi_stop
    if _smi_stop is not None:
        _smi_stop.set()
        _smi_stop = None


def reset_for_tests() -> None:
    """Drop process-local tracer state (unit tests only)."""
    global _fh, _role, _gpu, _attn_n, _attn_s, _attn_cast_s, _attn_sdpa_s, _attn_chunked, _torch_used
    if _fh is not None:
        try:
            _fh.close()
        except Exception:
            pass
        _fh = None
    _role = "unknown"
    _gpu = None
    _attn_n = 0
    _attn_s = 0.0
    _attn_cast_s = 0.0
    _attn_sdpa_s = 0.0
    _attn_chunked = 0
    _torch_used = False
    stop_smi_sampler()


def _num(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def load_events(directory: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        if path.name == "nvidia-smi.jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    events.sort(key=lambda e: e.get("ts", 0))
    return events


def load_smi(directory: Path) -> list[dict[str, Any]]:
    path = Path(directory) / "nvidia-smi.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_chrome_trace(events: list[dict[str, Any]], dest: Path) -> None:
    """Perfetto/chrome://tracing file; ``ts``/``dur`` are microseconds."""
    trace_events = []
    pids = sorted({int(e["pid"]) for e in events if "pid" in e})
    pid_meta = {pid: i + 1 for i, pid in enumerate(pids)}
    seen_meta: set[int] = set()
    for ev in events:
        pid = int(ev.get("pid", 0))
        local = pid_meta.get(pid, 1)
        if local not in seen_meta:
            label = f"{ev.get('role', '?')} pid={pid} gpu={ev.get('gpu')}"
            trace_events.append(
                {"name": "process_name", "ph": "M", "pid": local, "args": {"name": label}}
            )
            seen_meta.add(local)
        dur_us = max(1, int(float(ev.get("dur_ms", 0)) * 1000))
        ts_us = int(float(ev.get("ts", 0)) * 1e6) - dur_us
        trace_events.append(
            {
                "name": ev.get("name", "?"),
                "cat": ev.get("kind", "span"),
                "ph": "X",
                "ts": ts_us,
                "dur": dur_us,
                "pid": local,
                "tid": 1,
                "args": {k: v for k, v in ev.items() if k not in {"name", "kind", "ts", "dur_ms"}},
            }
        )
    dest.write_text(json.dumps({"traceEvents": trace_events}), encoding="utf-8")


def _fmt_s(ms: float) -> str:
    return f"{ms / 1000.0:8.2f}s"


def summarize(directory: Path) -> str:
    """Human-readable report: totals, waits/blocks, gen overlap, GPU util."""
    directory = Path(directory)
    events = load_events(directory)
    lines: list[str] = [f"# synevad profile  {directory.resolve()}", ""]
    if not events:
        lines.append("No span events. Was SYNEVAD_PROFILE set before workers spawned?")
        return "\n".join(lines) + "\n"

    by_name: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        by_name[str(ev.get("name"))].append(float(ev.get("dur_ms", 0)))
        by_kind[str(ev.get("kind", "span"))].append(ev)

    lines.append("## spans  (count, total, mean, max)")
    rows = sorted(by_name.items(), key=lambda kv: -sum(kv[1]))
    for name, durs in rows:
        lines.append(
            f"  {name:28s}  n={len(durs):4d}  total={_fmt_s(sum(durs))}  "
            f"mean={_fmt_s(sum(durs) / len(durs))}  max={_fmt_s(max(durs))}"
        )
    lines.append("")

    blocks = [e for e in events if e.get("kind") == _KIND_BLOCK]
    lines.append("## blocking / idle")
    if not blocks:
        lines.append("  (no wait/idle spans recorded)")
    else:
        for ev in sorted(blocks, key=lambda e: -float(e.get("dur_ms", 0)))[:20]:
            lines.append(
                f"  {float(ev['dur_ms']) / 1000:7.2f}s  {ev.get('name')}  "
                f"role={ev.get('role')} pid={ev.get('pid')} gpu={ev.get('gpu')}"
            )
        wait_job = sum(e["dur_ms"] for e in blocks if e.get("name") == "wait_for_job")
        wait_raw = sum(e["dur_ms"] for e in blocks if e.get("name") == "wait_for_raw")
        lines.append(
            f"  totals: wait_for_job={wait_job / 1000:.2f}s  wait_for_raw={wait_raw / 1000:.2f}s"
        )
        if wait_job > 2000:
            lines.append(
                "  note: gen workers spent >2s in wait_for_job — starved of queue work, not GPU."
            )
        if wait_raw > 5000:
            lines.append(
                "  note: mask worker idle waiting for crop_edited/ (FLUX still loading/running)."
            )
    lines.append("")

    gen_edits = [e for e in events if e.get("name") == "generate_one" and e.get("role") == "gen"]
    lines.append("## generation overlap")
    if len(gen_edits) < 2:
        lines.append(f"  {len(gen_edits)} generate_one span(s) — need ≥2 to measure overlap")
    else:
        intervals = []
        for e in gen_edits:
            end = float(e["ts"])
            start = end - float(e["dur_ms"]) / 1000.0
            intervals.append((start, end, e.get("gpu"), e.get("pid")))
        wall0 = min(s for s, _, _, _ in intervals)
        wall1 = max(t for _, t, _, _ in intervals)
        wall = wall1 - wall0
        busy = sum(t - s for s, t, _, _ in intervals)
        gpus = {g for _, _, g, _ in intervals}
        n_gpu = max(1, len(gpus))
        lines.append(f"  generate_one wall={wall:.1f}s  sum={busy:.1f}s  workers={n_gpu}")
        lines.append(f"  parallel efficiency={busy / wall / n_gpu:.2f}  (1.00 = fully overlapped)")
        if busy / wall / n_gpu < 0.7 and n_gpu > 1:
            lines.append("  note: gen workers are NOT fully overlapped — likely load serialisation or a shared lock.")
    lines.append("")

    steps = [e for e in events if e.get("name") == "dit_step"]
    if steps:
        attn = [float(e.get("attn_ms") or 0) for e in steps]
        casts = [float(e.get("attn_cast_ms") or 0) for e in steps]
        sdpas = [float(e.get("attn_sdpa_ms") or 0) for e in steps]
        durs = [float(e["dur_ms"]) for e in steps]
        chunked = sum(int(e.get("attn_chunked") or 0) for e in steps)
        lines.append("## diffusion steps")
        lines.append(
            f"  n={len(steps)}  mean={sum(durs) / len(durs) / 1000:.2f}s  "
            f"attn mean={sum(attn) / max(1, len(attn)) / 1000:.2f}s  "
            f"cast mean={sum(casts) / max(1, len(casts)) / 1000:.2f}s  "
            f"sdpa mean={sum(sdpas) / max(1, len(sdpas)) / 1000:.2f}s  "
            f"attn_chunked_calls={chunked}"
        )
        if chunked:
            lines.append("  note: chunked_sdpa ran — mem-efficient kernel missed at least once.")
        torch_avgs = sorted(Path(directory).glob("torch-*-averages.txt"))
        if torch_avgs:
            lines.append("  torch profiler averages:")
            for path in torch_avgs:
                lines.append(f"    {path.name}")
        lines.append("")

    smi = load_smi(directory)
    lines.append("## nvidia-smi")
    if not smi:
        lines.append("  (no nvidia-smi.jsonl)")
    else:
        util: dict[int, list[float]] = defaultdict(list)
        for snap in smi:
            for gpu in snap.get("gpus") or []:
                u = gpu.get("util_gpu")
                if u is not None:
                    util[int(gpu["index"])].append(float(u))
        for idx in sorted(util):
            vals = util[idx]
            busy_pct = 100.0 * sum(1 for v in vals if v >= 10) / len(vals)
            lines.append(
                f"  gpu {idx:2d}  samples={len(vals):4d}  "
                f"mean util={sum(vals) / len(vals):5.1f}%  "
                f"max={max(vals):5.1f}%  busy(≥10%)={busy_pct:5.1f}%"
            )
        # Overlap: fraction of samples where ≥2 GPUs are busy
        overlap_n = 0
        for snap in smi:
            gpus = snap.get("gpus") or []
            n_busy = sum(1 for g in gpus if (g.get("util_gpu") or 0) >= 10)
            if n_busy >= 2:
                overlap_n += 1
        if smi:
            lines.append(
                f"  samples with ≥2 GPUs busy: {overlap_n}/{len(smi)} "
                f"({100.0 * overlap_n / len(smi):.0f}%)"
            )
            if overlap_n / len(smi) < 0.3 and len(util) > 1:
                lines.append("  note: GPUs rarely busy together — work is serialised or only one card is computing.")
    lines.append("")
    chrome = directory / "chrome_trace.json"
    write_chrome_trace(events, chrome)
    lines.append(f"chrome/perfetto trace: {chrome}")
    return "\n".join(lines) + "\n"
