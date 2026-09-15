#!/usr/bin/env python3
"""Analyze GPU execution gaps and host launch overhead from CUPTI capture journals.

Reads CDW1-framed capture.journal streams emitted by cudaprof/cupti_agent,
merges device activity timelines across streams per device, computes exact
inter-kernel idle gaps, and correlates each gap against host API records
(cuLaunchKernel / cudaLaunchKernel) to attribute idle time to:
  1. Host Runtime Delay (Go runtime / scheduling before launch call)
  2. Driver Submission (time inside cuLaunchKernel)
  3. Hardware / Queue Latency (time after launch call returned)
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import zipfile
import tarfile

KIND_BOOTSTRAP = 1
KIND_ACTIVITY = 2
KIND_FINALIZED = 3
KIND_CAPABILITY = 4
KIND_FAILURE = 5
KIND_EXTERNAL = 6
KIND_API = 7

WIRE_HEADER_SIZE = 80
MAGIC_CDW1 = b"CDW1"


class JournalError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise JournalError(message)


def parse_cdw1_records(stream):
    """Yields (kind, payload_bytes) from a CDW1 binary stream."""
    while True:
        hdr = stream.read(WIRE_HEADER_SIZE)
        if not hdr:
            break
        require(len(hdr) == WIRE_HEADER_SIZE, f"truncated header: got {len(hdr)} bytes, expected {WIRE_HEADER_SIZE}")
        magic, ver, kind, plen = struct.unpack(">4sHHI", hdr[:12])
        require(magic == MAGIC_CDW1, f"bad magic: {magic!r}")
        require(ver == 1, f"unsupported version: {ver}")
        payload = stream.read(plen)
        require(len(payload) == plen, f"truncated payload: expected {plen} bytes, got {len(payload)}")
        yield kind, payload


def extract_journal_bytes(path):
    """Returns raw journal bytes from a path (journal, directory, pprof bundle, zip, or tar)."""
    p = Path(path)
    if p.is_dir():
        # Scan directory for capture.journal
        journals = list(p.glob("**/capture.journal")) + [
            f for f in p.glob("**/*capture.journal") if not f.name.startswith(".")
        ]
        require(len(journals) > 0, f"{path}: no capture.journal found in directory")
        # Prefer largest if multiple worker journals exist
        journals.sort(key=lambda x: x.stat().st_size, reverse=True)
        return journals[0].read_bytes()

    require(p.is_file(), f"{path}: not a regular file or directory")
    data = p.read_bytes()
    require(len(data) > 0, f"{path}: empty file")

    if data.startswith(MAGIC_CDW1):
        return data

    # Try gzip decompression (e.g. pprof bundle)
    if data.startswith(b"\x1f\x8b"):
        import gzip
        import base64
        try:
            decomp = gzip.decompress(data)
            for prefix in [b"#cudaprof:capture:v3\n", b"#cudaprof:capture:v2\n", b"#cudaprof:capture:v1\n"]:
                idx = decomp.find(prefix)
                if idx != -1:
                    start = idx + len(prefix)
                    raw_json = decomp[start:]
                    obj, _ = json.JSONDecoder().raw_decode(raw_json.decode("utf-8", errors="ignore"))
                    files = obj.get("files", {})
                    # Check omitted files trap if capture-status is present
                    if "capture-status.json" in files:
                        status_raw = base64.b64decode(files["capture-status.json"])
                        status = json.loads(status_raw.decode("utf-8"))
                        for f in status.get("omitted_files", []):
                            if "capture.journal" in f.get("name", ""):
                                raise JournalError(
                                    f"{path}: capture.journal was dropped from bundle (size {f.get('bytes')} exceeds budget). "
                                    "Read directly from the scratch directory."
                                )
                    for fname, fcontent in files.items():
                        if fname.endswith("capture.journal"):
                            return base64.b64decode(fcontent)
        except (gzip.BadGzipFile, json.JSONDecodeError):
            pass

    # Try ZIP archive
    try:
        with zipfile.ZipFile(p, "r") as zf:
            for name in zf.namelist():
                if name.endswith("capture.journal"):
                    return zf.read(name)
            if "capture-status.json" in zf.namelist():
                status = json.loads(zf.read("capture-status.json").decode("utf-8"))
                omitted = status.get("omitted_files", [])
                for f in omitted:
                    if "capture.journal" in f.get("name", ""):
                        raise JournalError(
                            f"{path}: capture.journal was dropped from bundle (size {f.get('bytes')} exceeds budget). "
                            "Read directly from the scratch directory."
                        )
    except zipfile.BadZipFile:
        pass

    # Try tar archive
    try:
        with tarfile.open(p, "r:*") as tf:
            for member in tf.getmembers():
                if member.name.endswith("capture.journal"):
                    f = tf.extractfile(member)
                    if f:
                        return f.read()
    except tarfile.TarError:
        pass

    raise JournalError(f"{path}: not a recognized CDW1 journal or archive containing capture.journal")


def analyze_journal(journal_bytes, warmup_ms=200.0):
    """Analyzes activities, API records, and gaps from raw journal bytes."""
    import io

    stream = io.BytesIO(journal_bytes)
    api_by_corr = {}
    activities = []
    dropped_records = 0

    for kind, payload in parse_cdw1_records(stream):
        if kind == KIND_API:
            rec = json.loads(payload.decode("utf-8"))
            corr = rec.get("correlation")
            if corr is not None:
                api_by_corr[corr] = rec
        elif kind == KIND_ACTIVITY:
            rec = json.loads(payload.decode("utf-8"))
            if rec.get("valid", False) and rec.get("start", 0) > 0 and rec.get("end", 0) >= rec.get("start", 0):
                activities.append(rec)

    require(len(activities) > 0, "no valid activity records found in journal")

    activities.sort(key=lambda a: (a["start"], a["end"]))

    # Match correlation
    corr_matched = sum(1 for a in activities if a.get("correlation") in api_by_corr)
    corr_match_rate = corr_matched / len(activities) if activities else 0.0

    def compute_metrics(acts, label="overall"):
        if not acts:
            return None
        t_first = acts[0]["start"]
        t_last = acts[-1]["end"]
        span_ns = max(0, t_last - t_first)
        span_ms = span_ns / 1e6

        # Merge overlapping/contiguous intervals across streams
        merged = []
        for a in acts:
            s, e = a["start"], a["end"]
            if not merged:
                merged.append([s, e, [a]])
            else:
                last = merged[-1]
                if s <= last[1]:
                    if e > last[1]:
                        last[1] = e
                    last[2].append(a)
                else:
                    merged.append([s, e, [a]])

        active_ns = sum(intv[1] - intv[0] for intv in merged)
        active_ms = active_ns / 1e6

        gaps_ns = []
        host_ahead_count = 0
        host_during_count = 0
        host_behind_count = 0

        hw_queue_ns = 0
        driver_sub_ns = 0
        host_behind_ns = 0

        for i in range(len(merged) - 1):
            a0_end = merged[i][1]
            a1_start = merged[i + 1][0]
            gap = a1_start - a0_end
            if gap <= 0:
                continue
            gaps_ns.append(gap)

            next_op = merged[i + 1][2][0]
            corr1 = next_op.get("correlation")
            api1 = api_by_corr.get(corr1)
            if not api1:
                hw_queue_ns += gap
                continue

            api_s = api1["start"]
            api_e = api1["end"]

            if api_e <= a0_end:
                host_ahead_count += 1
                hw_queue_ns += gap
            elif api_s <= a0_end < api_e:
                host_during_count += 1
                d_part = min(api_e - a0_end, gap)
                h_part = max(0, a1_start - api_e)
                driver_sub_ns += d_part
                hw_queue_ns += h_part
            else:
                host_behind_count += 1
                h_starve = min(api_s - a0_end, gap)
                d_part = min(api_e - api_s, max(0, gap - h_starve))
                h_part = max(0, a1_start - api_e)
                host_behind_ns += h_starve
                driver_sub_ns += d_part
                hw_queue_ns += h_part

        total_gap_ns = sum(gaps_ns)
        total_gap_ms = total_gap_ns / 1e6
        total_time_ns = active_ns + total_gap_ns
        duty_cycle = (active_ns / total_time_ns * 100.0) if total_time_ns > 0 else 0.0

        gaps_us = sorted([g / 1e3 for g in gaps_ns])
        n_gaps = len(gaps_us)

        def pct(p):
            if n_gaps == 0:
                return 0.0
            return gaps_us[int(p / 100.0 * (n_gaps - 1))]

        stats = {
            "label": label,
            "span_ms": span_ms,
            "activity_count": len(acts),
            "merged_intervals": len(merged),
            "gap_count": n_gaps,
            "active_ms": active_ms,
            "gap_ms": total_gap_ms,
            "duty_cycle_pct": duty_cycle,
            "attribution": {
                "host_delay_ms": host_behind_ns / 1e6,
                "host_delay_pct": (host_behind_ns / total_gap_ns * 100.0) if total_gap_ns > 0 else 0.0,
                "driver_submission_ms": driver_sub_ns / 1e6,
                "driver_submission_pct": (driver_sub_ns / total_gap_ns * 100.0) if total_gap_ns > 0 else 0.0,
                "hardware_queue_ms": hw_queue_ns / 1e6,
                "hardware_queue_pct": (hw_queue_ns / total_gap_ns * 100.0) if total_gap_ns > 0 else 0.0,
            },
            "launch_state_counts": {
                "host_ahead": host_ahead_count,
                "host_during": host_during_count,
                "host_behind": host_behind_count,
            },
            "distribution_us": {
                "min": gaps_us[0] if n_gaps else 0.0,
                "p25": pct(25),
                "p50": pct(50),
                "p75": pct(75),
                "p90": pct(90),
                "p99": pct(99),
                "max": gaps_us[-1] if n_gaps else 0.0,
                "mean": statistics.mean(gaps_us) if n_gaps else 0.0,
            },
        }
        return stats

    overall = compute_metrics(activities, label="overall")

    # Steady-state: exclude initial warmup_ms
    t0 = activities[0]["start"]
    warmup_ns = int(warmup_ms * 1e6)
    steady_acts = [a for a in activities if (a["start"] - t0) > warmup_ns]
    steady = compute_metrics(steady_acts, label="steady-state") if steady_acts else None

    return {
        "total_activities": len(activities),
        "total_apis": len(api_by_corr),
        "correlation_match_rate": corr_match_rate,
        "warmup_ms": warmup_ms,
        "overall": overall,
        "steady_state": steady,
    }


def format_report(result):
    lines = []
    lines.append("=" * 78)
    lines.append(" CUPTI INTER-KERNEL GAP & HOST OVERHEAD ATTRIBUTION REPORT")
    lines.append("=" * 78)
    lines.append(f"Activities: {result['total_activities']:,} | API Records: {result['total_apis']:,} | Correlation Match: {result['correlation_match_rate']*100:.1f}%")
    lines.append("")

    for section_name, stats in [("OVERALL RUN", result["overall"]), ("STEADY-STATE (EXCLUDING WARMUP)", result["steady_state"])]:
        if not stats:
            continue
        lines.append("-" * 78)
        lines.append(f" {section_name}")
        lines.append("-" * 78)
        lines.append(f"  Timeline Span:       {stats['span_ms']:10.2f} ms")
        lines.append(f"  GPU Active Time:     {stats['active_ms']:10.2f} ms  ({stats['active_ms']/stats['span_ms']*100:5.1f}% duty)")
        lines.append(f"  GPU Idle Gap Time:   {stats['gap_ms']:10.2f} ms  ({stats['gap_ms']/stats['span_ms']*100:5.1f}% idle)")
        lines.append(f"  GPU Duty Cycle:      {stats['duty_cycle_pct']:10.2f}%")
        lines.append(f"  Total Gaps:          {stats['gap_count']:10,d}")
        lines.append("")
        lines.append("  Idle Gap Attribution:")
        attr = stats["attribution"]
        lines.append(f"    Host Runtime Delay:  {attr['host_delay_ms']:8.2f} ms  ({attr['host_delay_pct']:5.1f}% of gap) [Go runtime / scheduling]")
        lines.append(f"    Driver Submission:   {attr['driver_submission_ms']:8.2f} ms  ({attr['driver_submission_pct']:5.1f}% of gap) [inside cuLaunchKernel]")
        lines.append(f"    Hardware/Queue Delay:{attr['hardware_queue_ms']:8.2f} ms  ({attr['hardware_queue_pct']:5.1f}% of gap) [HW pipeline / dispatch]")
        lines.append("")
        lcounts = stats["launch_state_counts"]
        total_classified = sum(lcounts.values()) or 1
        lines.append("  Host Pipeline Concurrency at Op Completion:")
        lines.append(f"    Host Ahead (queued before op ended):  {lcounts['host_ahead']:6,d} ({lcounts['host_ahead']/total_classified*100:5.1f}%)")
        lines.append(f"    Host During (in launch call):         {lcounts['host_during']:6,d} ({lcounts['host_during']/total_classified*100:5.1f}%)")
        lines.append(f"    Host Behind (starved GPU):            {lcounts['host_behind']:6,d} ({lcounts['host_behind']/total_classified*100:5.1f}%)")
        lines.append("")
        dist = stats["distribution_us"]
        lines.append("  Gap Duration Distribution (microseconds):")
        lines.append(f"    Min:   {dist['min']:8.2f} us   |  p25:  {dist['p25']:8.2f} us   |  Median (p50): {dist['p50']:8.2f} us")
        lines.append(f"    p75:   {dist['p75']:8.2f} us   |  p90:  {dist['p90']:8.2f} us   |  p99:          {dist['p99']:8.2f} us")
        lines.append(f"    Max:   {dist['max']:8.2f} us   |  Mean: {dist['mean']:8.2f} us")
        lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("journal", help="path to capture.journal or .pprof bundle")
    parser.add_argument("--warmup-ms", type=float, default=200.0, help="warmup window in ms to exclude for steady-state (default: 200.0)")
    parser.add_argument("--json", action="store_true", help="output JSON summary instead of text report")
    args = parser.parse_args()

    try:
        raw_bytes = extract_journal_bytes(args.journal)
        result = analyze_journal(raw_bytes, warmup_ms=args.warmup_ms)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_report(result))
    except Exception as e:
        sys.stderr.write(f"error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
