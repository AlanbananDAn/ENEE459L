from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    bench.workload.synchronize()
    times = []
    for _ in range(repeats):
        start = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()
        elapsed = (end - start) / 1_000_000.0
        times.append(elapsed)
    return times


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 4:
        return unknown("leading prefix above (1 + 0.5) x median of the run's second half", "too few samples")
    second_half = samples[len(samples) // 2:]
    median = statistics.median(second_half)
    if median <= 0:
        return unknown("leading prefix above (1 + 0.5) x median of the run's second half", "median <= 0")
    threshold = median * (1 + WARMUP_TOL)
    count = 0
    for s in samples:
        if s > threshold:
            count += 1
        else:
            break
    return measured(
        count,
        "leading prefix above (1 + 0.5) x median of the run's second half",
        settled_rate_ms=round(median, 4),
        threshold_ms=round(threshold, 4),
        tolerance=WARMUP_TOL,
        retained=len(samples) - count,
    )



def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    mean = statistics.fmean(sorted_samples)
    min_val = min(sorted_samples)
    max_val = max(sorted_samples)
    std = statistics.stdev(sorted_samples) if n > 2 else 0.0

    def percentile(q):
        h = (n - 1) * q / 100.0
        i = int(h)
        value = sorted_samples[i] + (h - i) * (sorted_samples[i + 1] - sorted_samples[i])
        return value

    p50 = percentile(50)
    p95 = percentile(95)
    p99 = percentile(99)

    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "p50": round(p50, 4),
        "p95": round(p95, 4),
        "p99": round(p99, 4),
    }

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    if len(samples) < MIN_SAMPLES_FOR_MODALITY:
        return unknown("widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side", "not enough samples")
    sorted_samples = sorted(samples)
    trim_low = int(len(sorted_samples) * 0.05)
    trim_high = int(len(sorted_samples) * 0.95)
    trimmed = sorted_samples[trim_low:trim_high] if trim_low < trim_high else sorted_samples
    if len(trimmed) < 2:
        return unknown("widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side", "trimmed too small")
    gaps = [trimmed[i + 1] - trimmed[i] for i in range(len(trimmed) - 1)]
    median_gap = statistics.median(gaps)
    if median_gap <= 0:
        return unknown("widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side", "timer resolution too coarse")
    widest_gap = max(gaps)
    ratio = widest_gap / median_gap
    split_index = gaps.index(widest_gap)
    left = trimmed[:split_index + 1]
    right = trimmed[split_index + 1:]
    left_n = len(left)
    right_n = len(right)
    total = len(trimmed)
    is_multimodal_result = ratio >= MULTIMODAL_GAP_RATIO and (left_n / total >= MIN_MODE_FRACTION) and (right_n / total >= MIN_MODE_FRACTION)
    return measured(
        is_multimodal_result,
        "widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side",
        gap_ratio=round(ratio, 2),
        widest_gap_ms=round(widest_gap, 4),
        typical_gap_ms=round(median_gap, 4),
        modes=[
            {"n": left_n, "share": round(left_n / total, 2), "median_ms": round(statistics.median(left), 4)},
            {"n": right_n, "share": round(right_n / total, 2), "median_ms": round(statistics.median(right), 4)},
        ],
    )

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    result = bench.runner(["nvpmodel", "-q"])
    if not result.ok or result.returncode != 0:
        return unknown("nvpmodel -q", f"command failed: {result.error or result.stdout}")
    mode_name = None
    mode_index = None
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if "NV Power Mode:" in line:
            mode_name = line.split(":")[-1].strip()
            if i + 1 < len(lines):
                try:
                    mode_index = int(lines[i + 1].strip())
                except ValueError:
                    pass
    min_text = read_text(bench.telemetry, CPUFREQ_MIN)
    max_text = read_text(bench.telemetry, CPUFREQ_MAX)
    jetson_clocks = None
    if min_text is not None and max_text is not None:
        try:
            min_freq = int(min_text)
            max_freq = int(max_text)
            jetson_clocks = min_freq == max_freq
        except ValueError:
            jetson_clocks = None
    else:
        jetson_clocks = None
    return measured(
        mode_name,
        "nvpmodel -q",
        mode_index=mode_index,
        jetson_clocks=jetson_clocks,
        jetson_clocks_source=measured(
            f"scaling_min_freq={min_text}, scaling_max_freq={max_text}",
            "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq vs sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
        ) if min_text is not None and max_text is not None else unknown(
            "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq vs sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
            "could not read min/max freq",
        ),
    )



def probe_telemetry(bench: Bench) -> dict[str, Any]:
    # Temperature
    zones_read = 0
    max_temp = None
    zone_name = None
    thermal_root = bench.telemetry / THERMAL_ZONES.lstrip("/")
    if thermal_root.exists():
        for zone_path in thermal_root.glob("thermal_zone*"):
            temp_path = zone_path / "temp"
            rel_path = THERMAL_ZONES + "/" + zone_path.name + "/temp"
            # Use errors="ignore" to avoid TypeError when read_text encounters binary/null data
            text = read_text(bench.telemetry, rel_path)
            if text is not None:
                try:
                    raw = int(text)
                    temp_c = raw / 1000.0
                    if temp_c > -1000:
                        zones_read += 1
                        if max_temp is None or temp_c > max_temp:
                            max_temp = temp_c
                            zone_name = zone_path.name
                except ValueError:
                    pass
    temperature_record = measured(max_temp, "sys/devices/virtual/thermal/*/temp", zone=zone_name, zones_read=zones_read) if max_temp is not None else unknown("sys/devices/virtual/thermal/*/temp", "no valid thermal zones found")

    # Power Draw
    power_result = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)
    if power_result is not None:
        path, text = power_result
        try:
            power_mw = int(text)
            power_record = measured(power_mw, path)
        except ValueError:
            power_record = unknown(path, "invalid integer")
    else:
        power_record = unknown("sys/bus/i2c/drivers/ina3221/*/in1_input | sys/bus/i2c/drivers/ina3221/*/in_power0_input", "none of the documented INA3221 rail paths could be read")

    # GPU Load
    gpu_result = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)
    if gpu_result is not None:
        path, text = gpu_result
        try:
            raw = int(text)
            gpu_util = raw / 10.0
            gpu_record = measured(gpu_util, path, units="per-mille / 10")
        except ValueError:
            gpu_record = unknown(path, "invalid integer")
    else:
        gpu_record = unknown("sys/devices/platform/gpu.0/load | sys/devices/gpu.0/load", "no GPU load file found")

    return {
        "temperature_c": temperature_record,
        "power_mw": power_record,
        "gpu_utilization_percent": gpu_record,
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)