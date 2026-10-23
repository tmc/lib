# cudaphase.py: summarizes a cudaphase.h trace buffer into one line,
#
#     CUDAPHASE {"name": ..., "blocks": ..., "span_us": ..., "phases": [...]}
#
# for a judge log, where only a short tail of stdout comes back. It needs
# only Python: the buffer is anything with tolist() (a torch tensor) or a
# list of ints, CUDAPHASE_SLOTS values per block. The Go package
# github.com/tmc/lib/nvidia/x/cudaphase parses the line and holds the
# reference implementation (Summarize, SummarizeClock) this mirrors.
import json


def cudaphase_summary(name, buf, phases, slots=16, sm_slot=None, extra=None, clock=None):
    """phases is a list of (name, from_slot, to_slot).

    Without clock, phase slots hold cudaphase_now, in ns. With clock, a
    dict of slots {"start", "end", "timer_start", "timer_end"} and
    optionally "sm_end" and "min_span_ns" (default 2000), phase slots hold
    cudaphase_clock cycles, converted at the measured SM clock rate.
    """
    vals = buf.tolist() if hasattr(buf, "tolist") else list(buf)
    rows = [vals[i:i + slots] for i in range(0, len(vals) - slots + 1, slots)]
    rows = [r for r in rows if any(r)]  # blocks that wrote nothing never ran
    s = {"name": name, "blocks": len(rows)}
    timers, placed, rate = set(), [], 1.0
    if clock is not None:
        c0, c1, t0, t1 = clock["start"], clock["end"], clock["timer_start"], clock["timer_end"]
        sm_end, min_span = clock.get("sm_end"), clock.get("min_span_ns") or 2000
        kept, ratios, migrated = [], [], 0
        for r in rows:
            if sm_end is not None and r[sm_slot] > 0 and r[sm_end] > 0 and r[sm_slot] != r[sm_end]:
                migrated += 1  # cycles from two SMs' clocks
                continue
            kept.append(r)
            if r[t0] > 0 and r[t1] >= r[t0] + min_span and r[c0] > 0 and r[c1] > r[c0]:
                ratios.append((r[t1] - r[t0]) / (r[c1] - r[c0]))
            timers.update(r[k] for k in (t0, t1) if r[k] > 0)
        if not ratios:
            raise ValueError(f"no block spans {min_span} ns with clock and timer stamps at both ends")
        ratios.sort()
        rate = ratios[(len(ratios) - 1) // 2]
        rows = kept
    out = []
    for pname, a, b in phases:
        d = sorted((r[b] - r[a]) * rate for r in rows if r[a] > 0 and r[b] > 0)
        for r in rows:
            for k in (a, b):
                if r[k] == 0:
                    continue
                if clock is None:
                    timers.add(r[k])
                    placed.append(r[k])
                elif r[t0] > 0 and r[c0] > 0:
                    placed.append(r[t0] + (r[k] - r[c0]) * rate)
        out.append({"name": pname, "blocks": len(d),
                    "median_us": d[(len(d) - 1) // 2] / 1e3 if d else 0.0,
                    "min_us": d[0] / 1e3 if d else 0.0,
                    "max_us": d[-1] / 1e3 if d else 0.0})
    ts = sorted(timers)
    gaps = [y - x for x, y in zip(ts, ts[1:])]
    s["span_us"] = (max(placed) - min(placed)) / 1e3 if placed else 0.0
    s["quantum_us"] = min(gaps) / 1e3 if gaps else 0.0
    if sm_slot is not None:
        s["sms"] = len({r[sm_slot] for r in rows if r[sm_slot] > 0})
    if clock is not None:
        s["ghz"], s["rate_blocks"] = 1 / rate, len(ratios)
        if migrated:
            s["migrated"] = migrated
    s["phases"] = out
    if extra:
        s["extra"] = extra
    return "CUDAPHASE " + json.dumps(s, separators=(",", ":"))
