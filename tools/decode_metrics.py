#!/usr/bin/env python3
"""Collect decode-only TPOT and TPS from vLLM Decoder Prometheus metrics.

This tool is intentionally external to the inference hot path. It polls only the
Decoder /metrics endpoint and does not add logging, synchronization, or Python
work to DSA/MTP/model execution.

Decode token accounting follows vLLM's native TPOT convention: the first output
token of each request is excluded from decode. Therefore:

  decode_tokens = delta(generation_tokens_total) - delta(TTFT_count)
  decode_tps    = decode_tokens / elapsed_seconds

TPOT is read from vLLM's native per-request TPOT histogram:

  decode_tpot = delta(TPOT_sum) / delta(TPOT_count)

Only completed requests contribute TPOT samples, which matches the native vLLM
metric behavior.
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

GENERATION_TOKENS = "vllm:generation_tokens_total"
TTFT_COUNT = "vllm:time_to_first_token_seconds_count"
TPOT_SUM = "vllm:request_time_per_output_token_seconds_sum"
TPOT_COUNT = "vllm:request_time_per_output_token_seconds_count"
REQUIRED_METRICS = (GENERATION_TOKENS, TTFT_COUNT, TPOT_SUM, TPOT_COUNT)


@dataclass(frozen=True)
class Snapshot:
    monotonic_ts: float
    wall_ts: float
    generation_tokens: float
    ttft_count: float
    tpot_sum: float
    tpot_count: float


@dataclass(frozen=True)
class DecodeMetrics:
    elapsed_s: float
    generation_tokens: float
    first_tokens: float
    decode_tokens: float
    decode_tps: float
    decode_tpot_ms: float
    tpot_samples: float


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/metrics"):
        endpoint += "/metrics"
    return endpoint


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum the required metric series across labels from one endpoint."""
    values = {name: 0.0 for name in REQUIRED_METRICS}
    seen: set[str] = set()

    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue

        parts = line.split(None, 2)
        if len(parts) < 2:
            continue

        metric_with_labels, raw_value = parts[0], parts[1]
        metric_name = metric_with_labels.split("{", 1)[0]
        if metric_name not in values:
            continue

        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue

        values[metric_name] += value
        seen.add(metric_name)

    missing = [name for name in REQUIRED_METRICS if name not in seen]
    if missing:
        raise RuntimeError("missing metrics: " + ", ".join(missing))
    return values


def fetch_endpoint(endpoint: str, timeout: float) -> dict[str, float]:
    request = urllib.request.Request(
        endpoint,
        headers={"Accept": "text/plain; version=0.0.4"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")

    try:
        return parse_prometheus(body)
    except RuntimeError as exc:
        raise RuntimeError(f"{endpoint}: {exc}") from exc


def take_snapshot(endpoints: list[str], timeout: float) -> Snapshot:
    """Read all independent Decoder endpoints and aggregate their counters."""
    begin = time.monotonic()
    fetch = partial(fetch_endpoint, timeout=timeout)

    if len(endpoints) == 1:
        samples = [fetch(endpoints[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            samples = list(pool.map(fetch, endpoints))

    end = time.monotonic()
    totals = {name: 0.0 for name in REQUIRED_METRICS}
    for sample in samples:
        for name in REQUIRED_METRICS:
            totals[name] += sample[name]

    return Snapshot(
        monotonic_ts=(begin + end) / 2.0,
        wall_ts=time.time(),
        generation_tokens=totals[GENERATION_TOKENS],
        ttft_count=totals[TTFT_COUNT],
        tpot_sum=totals[TPOT_SUM],
        tpot_count=totals[TPOT_COUNT],
    )


def calculate_delta(old: Snapshot, new: Snapshot) -> DecodeMetrics | None:
    elapsed_s = new.monotonic_ts - old.monotonic_ts
    generation_tokens = new.generation_tokens - old.generation_tokens
    first_tokens = new.ttft_count - old.ttft_count
    tpot_sum = new.tpot_sum - old.tpot_sum
    tpot_samples = new.tpot_count - old.tpot_count

    # Any negative cumulative delta means at least one metrics process restarted.
    if (
        elapsed_s <= 0
        or generation_tokens < -1e-9
        or first_tokens < -1e-9
        or tpot_sum < -1e-9
        or tpot_samples < -1e-9
    ):
        return None

    decode_tokens = max(0.0, generation_tokens - first_tokens)
    decode_tps = decode_tokens / elapsed_s
    decode_tpot_ms = (
        1000.0 * tpot_sum / tpot_samples if tpot_samples > 0 else math.nan
    )

    return DecodeMetrics(
        elapsed_s=elapsed_s,
        generation_tokens=generation_tokens,
        first_tokens=first_tokens,
        decode_tokens=decode_tokens,
        decode_tps=decode_tps,
        decode_tpot_ms=decode_tpot_ms,
        tpot_samples=tpot_samples,
    )


def format_float(value: float, digits: int = 3) -> str:
    return "N/A" if math.isnan(value) else f"{value:.{digits}f}"


def run_self_test() -> None:
    sample = """\
# HELP vllm:generation_tokens_total Number of generation tokens processed.
vllm:generation_tokens_total{engine="0",model_name="glm5-1"} 120
vllm:generation_tokens_total{engine="1",model_name="glm5-1"} 80
vllm:time_to_first_token_seconds_count{engine="0",model_name="glm5-1"} 10
vllm:time_to_first_token_seconds_count{engine="1",model_name="glm5-1"} 5
vllm:request_time_per_output_token_seconds_sum{engine="0",model_name="glm5-1"} 0.8
vllm:request_time_per_output_token_seconds_sum{engine="1",model_name="glm5-1"} 0.4
vllm:request_time_per_output_token_seconds_count{engine="0",model_name="glm5-1"} 8
vllm:request_time_per_output_token_seconds_count{engine="1",model_name="glm5-1"} 4
"""
    parsed = parse_prometheus(sample)
    assert parsed[GENERATION_TOKENS] == 200
    assert parsed[TTFT_COUNT] == 15
    assert abs(parsed[TPOT_SUM] - 1.2) < 1e-12
    assert parsed[TPOT_COUNT] == 12

    old = Snapshot(10.0, 0.0, 100.0, 10.0, 0.4, 4.0)
    new = Snapshot(20.0, 0.0, 220.0, 20.0, 0.9, 9.0)
    result = calculate_delta(old, new)
    assert result is not None
    assert result.decode_tokens == 110.0
    assert result.decode_tps == 11.0
    assert abs(result.decode_tpot_ms - 100.0) < 1e-9

    reset = Snapshot(30.0, 0.0, 1.0, 1.0, 0.01, 1.0)
    assert calculate_delta(new, reset) is None
    print("self-test: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect decode-only TPOT/TPS from Decoder /metrics with no inference hot-path changes."
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        default=[],
        help=(
            "Decoder service URL or /metrics URL. Repeat only for independent Decoder replicas. "
            "Do not mix a load-balancer endpoint with its backend endpoints."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Sampling interval in seconds (default: 10).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="HTTP timeout per endpoint in seconds (default: 3).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after N seconds; 0 means run until Ctrl-C (default: 0).",
    )
    parser.add_argument(
        "--csv",
        default="decode_metrics.csv",
        help="CSV output path (default: decode_metrics.csv).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run parser/accounting checks without contacting a server.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.self_test:
        run_self_test()
        return 0
    if not args.endpoint:
        raise SystemExit("at least one --endpoint is required")
    if args.interval <= 0:
        raise SystemExit("--interval must be > 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    if args.duration < 0:
        raise SystemExit("--duration must be >= 0")

    endpoints = [normalize_endpoint(endpoint) for endpoint in args.endpoint]
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    stop = False

    def handle_signal(signum: int, frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Decoder endpoints:")
    for endpoint in endpoints:
        print(f"  {endpoint}")
    print(f"interval={args.interval}s csv={csv_path}")

    try:
        previous = take_snapshot(endpoints, args.timeout)
    except Exception as exc:
        print(f"initial snapshot failed: {exc}", file=sys.stderr)
        return 2

    baseline = previous
    started = time.monotonic()

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "timestamp",
                "window_elapsed_s",
                "window_generation_tokens",
                "window_first_tokens",
                "window_decode_tokens",
                "window_decode_tps",
                "window_decode_tpot_ms",
                "window_tpot_samples",
                "avg_decode_tps",
                "avg_decode_tpot_ms",
                "total_decode_tokens",
                "total_tpot_samples",
            ]
        )
        csv_file.flush()

        while not stop:
            if args.duration and time.monotonic() - started >= args.duration:
                break

            sleep_s = args.interval
            if args.duration:
                remaining = args.duration - (time.monotonic() - started)
                if remaining <= 0:
                    break
                sleep_s = min(sleep_s, remaining)
            time.sleep(sleep_s)

            try:
                current = take_snapshot(endpoints, args.timeout)
            except Exception as exc:
                print(f"[WARN] snapshot failed: {exc}", file=sys.stderr)
                continue

            window = calculate_delta(previous, current)
            total = calculate_delta(baseline, current)
            if window is None or total is None:
                print("[WARN] metric reset detected; reset baseline", file=sys.stderr)
                previous = current
                baseline = current
                continue

            timestamp = datetime.fromtimestamp(current.wall_ts).astimezone().isoformat(timespec="seconds")
            print(
                f"{timestamp} "
                f"decode_tokens={int(round(window.decode_tokens))} "
                f"TPS={window.decode_tps:.3f} "
                f"TPOT_ms={format_float(window.decode_tpot_ms)} "
                f"samples={int(round(window.tpot_samples))} "
                f"| avg_TPS={total.decode_tps:.3f} "
                f"avg_TPOT_ms={format_float(total.decode_tpot_ms)}"
            )

            writer.writerow(
                [
                    timestamp,
                    f"{window.elapsed_s:.6f}",
                    int(round(window.generation_tokens)),
                    int(round(window.first_tokens)),
                    int(round(window.decode_tokens)),
                    f"{window.decode_tps:.6f}",
                    "" if math.isnan(window.decode_tpot_ms) else f"{window.decode_tpot_ms:.6f}",
                    int(round(window.tpot_samples)),
                    f"{total.decode_tps:.6f}",
                    "" if math.isnan(total.decode_tpot_ms) else f"{total.decode_tpot_ms:.6f}",
                    int(round(total.decode_tokens)),
                    int(round(total.tpot_samples)),
                ]
            )
            csv_file.flush()
            previous = current

    try:
        final_snapshot = take_snapshot(endpoints, args.timeout)
        total = calculate_delta(baseline, final_snapshot)
    except Exception as exc:
        print(f"[WARN] final snapshot failed: {exc}", file=sys.stderr)
        total = None

    if total is not None:
        print(
            "FINAL "
            f"decode_tokens={int(round(total.decode_tokens))} "
            f"decode_TPS={total.decode_tps:.3f} "
            f"decode_TPOT_ms={format_float(total.decode_tpot_ms)} "
            f"TPOT_samples={int(round(total.tpot_samples))}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
