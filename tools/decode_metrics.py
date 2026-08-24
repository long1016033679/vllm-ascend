#!/usr/bin/env python3
"""Measure steady-state Decode TPS/TPOT from existing vLLM metrics.

No inference hot-path changes are required. The tool only polls Decoder /metrics
and never inserts logging/synchronization into DSA, MTP, ACLGraph or model code.

State machine:
  WAIT_DECODE -> WARMUP -> SEEK_STEADY -> MEASURE -> FINAL

Decode-only gate:
  running == --expected-running
  waiting == 0
  delta(prompt_tokens_total) == 0
  delta(generation_tokens_total) > 0

The first lower-concurrency tail sample is never included in the final result.
TPS counts actual generated output tokens, so MTP rejected draft tokens are not
counted. Steady TPOT is request-occupancy time / generated output tokens; with
fixed concurrency C this is C / TPS.

For PD deployments, scrape Decoder endpoints only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import signal
import statistics
import sys
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

GEN = "vllm:generation_tokens_total"
PROMPT = "vllm:prompt_tokens_total"
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
KV = "vllm:kv_cache_usage_perc"
REQUIRED = (GEN, PROMPT, RUNNING, WAITING)
TARGETS = set(REQUIRED + (KV,))
METRIC_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|Inf|-Inf)$"
)
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"])*)"')


@dataclass(frozen=True)
class Snapshot:
    mono: float
    wall: float
    generation: float
    prompt: float
    running: int
    waiting: int
    kv_avg: float
    engine_generation: dict[str, float]
    engine_running: dict[str, float]


@dataclass(frozen=True)
class Window:
    dt: float
    dgen: float
    dprompt: float
    running: int
    waiting: int
    tps: float
    tpot_ms: float
    kv_avg: float


@dataclass
class Aggregate:
    seconds: float = 0.0
    tokens: float = 0.0
    occupancy_s: float = 0.0
    tps_windows: list[float] = field(default_factory=list)
    tpot_windows: list[float] = field(default_factory=list)

    def add(self, w: Window) -> None:
        self.seconds += w.dt
        self.tokens += w.dgen
        self.occupancy_s += w.running * w.dt
        self.tps_windows.append(w.tps)
        self.tpot_windows.append(w.tpot_ms)

    @property
    def samples(self) -> int:
        return len(self.tps_windows)

    @property
    def tps(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else math.nan

    @property
    def tpot_ms(self) -> float:
        return 1000.0 * self.occupancy_s / self.tokens if self.tokens > 0 else math.nan


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/metrics") else endpoint + "/metrics"


def labels(raw: str | None) -> dict[str, str]:
    return dict(LABEL_RE.findall(raw or ""))


def parse_metrics(
    text: str, model_name: str | None
) -> tuple[dict[str, float], dict[str, int], dict[str, float], dict[str, float]]:
    values = {name: 0.0 for name in TARGETS}
    counts = {name: 0 for name in TARGETS}
    engine_gen: dict[str, float] = {}
    engine_run: dict[str, float] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        if name not in TARGETS:
            continue
        metric_labels = labels(raw_labels)
        if model_name is not None and metric_labels.get("model_name") != model_name:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        values[name] += value
        counts[name] += 1
        engine = metric_labels.get("engine")
        if engine is not None and name == GEN:
            engine_gen[engine] = engine_gen.get(engine, 0.0) + value
        elif engine is not None and name == RUNNING:
            engine_run[engine] = engine_run.get(engine, 0.0) + value

    missing = [name for name in REQUIRED if counts[name] == 0]
    if missing:
        raise RuntimeError("missing required metrics: " + ", ".join(missing))
    return values, counts, engine_gen, engine_run


def fetch(endpoint: str, timeout: float, model_name: str | None):
    req = urllib.request.Request(
        endpoint, headers={"Accept": "text/plain; version=0.0.4"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    try:
        return parse_metrics(text, model_name)
    except RuntimeError as exc:
        raise RuntimeError(f"{endpoint}: {exc}") from exc


def snapshot(endpoints: list[str], timeout: float, model_name: str | None) -> Snapshot:
    begin = time.monotonic()
    if len(endpoints) == 1:
        samples = [fetch(endpoints[0], timeout, model_name)]
    else:
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            samples = list(
                pool.map(lambda e: fetch(e, timeout, model_name), endpoints)
            )
    end = time.monotonic()

    totals = {name: 0.0 for name in TARGETS}
    counts = {name: 0 for name in TARGETS}
    engine_gen: dict[str, float] = {}
    engine_run: dict[str, float] = {}
    for endpoint, (vals, cnts, e_gen, e_run) in zip(
        endpoints, samples, strict=True
    ):
        for name in TARGETS:
            totals[name] += vals[name]
            counts[name] += cnts[name]
        for engine, value in e_gen.items():
            engine_gen[f"{endpoint}|{engine}"] = value
        for engine, value in e_run.items():
            engine_run[f"{endpoint}|{engine}"] = value

    kv_avg = totals[KV] / counts[KV] if counts[KV] else math.nan
    return Snapshot(
        mono=(begin + end) / 2.0,
        wall=time.time(),
        generation=totals[GEN],
        prompt=totals[PROMPT],
        running=int(round(totals[RUNNING])),
        waiting=int(round(totals[WAITING])),
        kv_avg=kv_avg,
        engine_generation=engine_gen,
        engine_running=engine_run,
    )


def delta(old: Snapshot, new: Snapshot) -> Window | None:
    dt = new.mono - old.mono
    dgen = new.generation - old.generation
    dprompt = new.prompt - old.prompt
    if dt <= 0 or dgen < -1e-9 or dprompt < -1e-9:
        return None
    tps = dgen / dt
    tpot_ms = (
        1000.0 * new.running * dt / dgen
        if dgen > 0 and new.running > 0
        else math.nan
    )
    return Window(
        dt, dgen, dprompt, new.running, new.waiting, tps, tpot_ms, new.kv_avg
    )


def decode_only(w: Window, expected: int) -> bool:
    return (
        w.running == expected
        and w.waiting == 0
        and abs(w.dprompt) < 1e-9
        and w.dgen > 0
    )


def cv(values: list[float]) -> float:
    if not values:
        return math.inf
    mean = statistics.fmean(values)
    if mean <= 0:
        return math.inf
    return statistics.pstdev(values) / mean if len(values) > 1 else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def fmt(value: float, digits: int = 3) -> str:
    return "N/A" if math.isnan(value) else f"{value:.{digits}f}"


def run_self_test() -> None:
    text = """\
vllm:generation_tokens_total{engine="0",model_name="glm5-1"} 120
vllm:generation_tokens_total{engine="1",model_name="glm5-1"} 80
vllm:prompt_tokens_total{engine="0",model_name="glm5-1"} 64000
vllm:prompt_tokens_total{engine="1",model_name="glm5-1"} 64000
vllm:num_requests_running{engine="0",model_name="glm5-1"} 1
vllm:num_requests_running{engine="1",model_name="glm5-1"} 1
vllm:num_requests_waiting{engine="0",model_name="glm5-1"} 0
vllm:num_requests_waiting{engine="1",model_name="glm5-1"} 0
vllm:kv_cache_usage_perc{engine="0",model_name="glm5-1"} 0.2
vllm:kv_cache_usage_perc{engine="1",model_name="glm5-1"} 0.3
"""
    vals, counts, _, _ = parse_metrics(text, "glm5-1")
    assert vals[GEN] == 200 and vals[PROMPT] == 128000
    assert vals[RUNNING] == 2 and vals[WAITING] == 0 and counts[KV] == 2
    old = Snapshot(10, 0, 100, 128000, 2, 0, 0.25, {}, {})
    new = Snapshot(20, 0, 300, 128000, 2, 0, 0.25, {}, {})
    w = delta(old, new)
    assert w is not None and decode_only(w, 2)
    assert w.tps == 20 and abs(w.tpot_ms - 100) < 1e-9
    changed = Snapshot(30, 0, 400, 128001, 2, 0, 0.25, {}, {})
    w2 = delta(new, changed)
    assert w2 is not None and not decode_only(w2, 2)
    reset = Snapshot(40, 0, 1, 1, 2, 0, 0.25, {}, {})
    assert delta(changed, reset) is None
    assert cv([100, 101, 99, 100]) < 0.01
    print("self-test: PASS")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure full-concurrency steady Decode TPS/TPOT from vLLM /metrics."
    )
    p.add_argument(
        "--endpoint",
        action="append",
        default=[],
        help="Decoder URL or /metrics URL; repeat for independent D endpoints.",
    )
    p.add_argument(
        "--expected-running",
        type=int,
        help="Total steady Decode request concurrency; not card count.",
    )
    p.add_argument("--model-name", default=None, help="Optional model_name label filter.")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=3.0)
    p.add_argument("--arm-samples", type=int, default=3)
    p.add_argument("--warmup-seconds", type=float, default=10.0)
    p.add_argument("--steady-samples", type=int, default=5)
    p.add_argument(
        "--steady-cv",
        type=float,
        default=0.03,
        help="TPS CV threshold; default 0.03 = 3%%.",
    )
    p.add_argument(
        "--measure-seconds",
        type=float,
        default=60.0,
        help="0 means until tail/Ctrl-C.",
    )
    p.add_argument("--csv", default="decode_steady_metrics.csv")
    p.add_argument("--summary-json", default="decode_steady_summary.json")
    p.add_argument("--per-engine", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def validate(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    if not args.endpoint:
        raise SystemExit("at least one --endpoint is required")
    if not args.expected_running or args.expected_running <= 0:
        raise SystemExit("--expected-running must be > 0")
    if args.interval <= 0 or args.timeout <= 0 or args.arm_samples <= 0:
        raise SystemExit("--interval, --timeout and --arm-samples must be > 0")
    if args.warmup_seconds < 0 or args.steady_cv < 0 or args.measure_seconds < 0:
        raise SystemExit("warmup/steady-cv/measure values must be >= 0")
    if args.steady_samples < 2:
        raise SystemExit("--steady-samples must be >= 2")


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parser().parse_args()
    validate(args)
    if args.self_test:
        run_self_test()
        return 0

    endpoints = list(dict.fromkeys(normalize_endpoint(e) for e in args.endpoint))
    csv_path, summary_path = Path(args.csv), Path(args.summary_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def stop_handler(signum: int, frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    try:
        previous = snapshot(endpoints, args.timeout, args.model_name)
    except Exception as exc:
        print(f"[DECODE_MONITOR_ERROR] initial snapshot failed: {exc}", file=sys.stderr)
        return 2

    phase = "WAIT_DECODE"
    armed = 0
    decode_start = 0.0
    candidates: deque[float] = deque(maxlen=args.steady_samples)
    total = Aggregate()
    end_reason = "unknown"

    print(
        "[DECODE_MONITOR_INIT] "
        f"expected_running={args.expected_running} interval={args.interval}s "
        f"warmup={args.warmup_seconds}s steady_samples={args.steady_samples} "
        f"steady_cv={args.steady_cv} measure={args.measure_seconds}s "
        f"endpoints={','.join(endpoints)}"
    )

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "timestamp",
                "dt_s",
                "running",
                "generation_tokens",
                "tps",
                "tpot_ms",
                "kv_usage_pct",
            ]
        )

        while not stop:
            time.sleep(args.interval)
            try:
                current = snapshot(endpoints, args.timeout, args.model_name)
            except Exception as exc:
                print(f"[DECODE_MONITOR_WARN] snapshot failed: {exc}", file=sys.stderr)
                continue
            w = delta(previous, current)
            if w is None:
                print(
                    "[DECODE_MONITOR_WARN] counter reset; restart gate",
                    file=sys.stderr,
                )
                phase, armed, candidates = (
                    "WAIT_DECODE",
                    0,
                    deque(maxlen=args.steady_samples),
                )
                previous = current
                continue

            is_decode = decode_only(w, args.expected_running)
            ts = datetime.fromtimestamp(current.wall).astimezone().isoformat(
                timespec="seconds"
            )
            kv_pct = 100.0 * w.kv_avg if not math.isnan(w.kv_avg) else math.nan

            if phase == "WAIT_DECODE":
                armed = armed + 1 if is_decode else 0
                print(
                    f"[DECODE_GATE] armed={armed}/{args.arm_samples} "
                    f"running={w.running} waiting={w.waiting} "
                    f"d_prompt={int(round(w.dprompt))} d_gen={int(round(w.dgen))} "
                    f"TPS={w.tps:.3f} kv={fmt(kv_pct, 1)}%"
                )
                if armed >= args.arm_samples:
                    phase, decode_start = "WARMUP", current.mono
                    print(
                        f"[DECODE_START] running={w.running} "
                        f"warmup={args.warmup_seconds}s timestamp={ts}"
                    )

            elif phase == "WARMUP":
                if not is_decode:
                    print("[DECODE_WARMUP_RESET] decode-only condition lost")
                    phase, armed = "WAIT_DECODE", 0
                elif current.mono - decode_start >= args.warmup_seconds:
                    phase, candidates = (
                        "SEEK_STEADY",
                        deque(maxlen=args.steady_samples),
                    )
                    print("[DECODE_WARMUP_DONE] seeking steady TPS")

            elif phase == "SEEK_STEADY":
                if w.running < args.expected_running:
                    end_reason = "tail_before_steady"
                    break
                if not is_decode:
                    candidates.clear()
                    print("[DECODE_STEADY_ARM_RESET] decode-only condition lost")
                else:
                    candidates.append(w.tps)
                    rolling_cv = cv(list(candidates))
                    print(
                        f"[DECODE_STEADY_ARM] "
                        f"samples={len(candidates)}/{args.steady_samples} "
                        f"TPS={w.tps:.3f} CV={fmt(rolling_cv, 4)}"
                    )
                    if (
                        len(candidates) == args.steady_samples
                        and rolling_cv <= args.steady_cv
                    ):
                        phase, total = "MEASURE", Aggregate()
                        print(
                            f"[DECODE_STEADY_BEGIN] running={w.running} "
                            f"rolling_CV={rolling_cv:.4f} timestamp={ts}"
                        )

            elif phase == "MEASURE":
                if w.running < args.expected_running:
                    end_reason = "tail_running_dropped"
                    print(
                        f"[DECODE_TAIL] running={w.running} "
                        f"expected={args.expected_running}"
                    )
                    break
                if not is_decode:
                    print(
                        f"[DECODE_STEADY_SKIP] running={w.running} "
                        f"waiting={w.waiting} d_prompt={int(round(w.dprompt))} "
                        f"d_gen={int(round(w.dgen))}"
                    )
                else:
                    total.add(w)
                    writer.writerow(
                        [
                            ts,
                            f"{w.dt:.6f}",
                            w.running,
                            int(round(w.dgen)),
                            f"{w.tps:.6f}",
                            f"{w.tpot_ms:.6f}",
                            fmt(kv_pct, 6),
                        ]
                    )
                    fp.flush()
                    extra = ""
                    if args.per_engine:
                        items = []
                        keys = sorted(
                            set(previous.engine_generation)
                            & set(current.engine_generation)
                            & set(current.engine_running)
                        )
                        for key in keys:
                            dgen = (
                                current.engine_generation[key]
                                - previous.engine_generation[key]
                            )
                            if dgen >= 0:
                                engine = key.rsplit("|", 1)[-1]
                                items.append(
                                    f"e{engine}:r{int(round(current.engine_running[key]))}:"
                                    f"tps{dgen / w.dt:.2f}"
                                )
                        if items:
                            extra = " engines=[" + " ".join(items) + "]"
                    print(
                        f"[DECODE_STEADY] window={total.samples} dt={w.dt:.3f}s "
                        f"running={w.running} gen_tokens={int(round(w.dgen))} "
                        f"TPS={w.tps:.3f} TPOT_ms={w.tpot_ms:.3f} "
                        f"kv={fmt(kv_pct, 1)}%{extra}"
                    )
                    if (
                        args.measure_seconds > 0
                        and total.seconds >= args.measure_seconds
                    ):
                        end_reason = "measure_seconds_reached"
                        break

            previous = current

    if stop and end_reason == "unknown":
        end_reason = "signal"

    if total.samples:
        final_cv = cv(total.tps_windows)
        summary = {
            "status": "ok",
            "end_reason": end_reason,
            "expected_running": args.expected_running,
            "measurement_samples": total.samples,
            "measurement_seconds": total.seconds,
            "generation_tokens": int(round(total.tokens)),
            "steady_tps": total.tps,
            "steady_tpot_ms": total.tpot_ms,
            "window_tps_median": statistics.median(total.tps_windows),
            "window_tpot_ms_median": statistics.median(total.tpot_windows),
            "window_tps_cv": final_cv,
            "window_tps_p10": percentile(total.tps_windows, 0.10),
            "window_tps_p90": percentile(total.tps_windows, 0.90),
            "window_tpot_ms_p10": percentile(total.tpot_windows, 0.10),
            "window_tpot_ms_p90": percentile(total.tpot_windows, 0.90),
            "endpoints": endpoints,
            "model_name": args.model_name,
            "finished_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        }
        write_json(summary_path, summary)
        print(
            f"[DECODE_STEADY_FINAL] reason={end_reason} "
            f"samples={total.samples} duration={total.seconds:.3f}s "
            f"gen_tokens={int(round(total.tokens))} TPS={total.tps:.3f} "
            f"TPOT_ms={total.tpot_ms:.3f} TPS_CV={final_cv:.4f} "
            f"summary={summary_path} csv={csv_path}"
        )
        return 0

    write_json(
        summary_path,
        {
            "status": "no_steady_samples",
            "end_reason": end_reason,
            "expected_running": args.expected_running,
            "endpoints": endpoints,
            "model_name": args.model_name,
        },
    )
    print(
        f"[DECODE_STEADY_FINAL] status=no_steady_samples "
        f"reason={end_reason} summary={summary_path}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
