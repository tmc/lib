# cudaphase.py: summarizes a cudaphase.h trace buffer into one line,
#
#     CUDAPHASE {"name": ..., "blocks": ..., "span_us": ..., "phases": [...]}
#
# for a judge log, where only a short tail of stdout comes back. It needs
# only Python: the buffer is anything with tolist() (a torch tensor) or a
# list of ints, CUDAPHASE_SLOTS values per block. The Go package
# github.com/tmc/lib/nvidia/x/cudaphase parses the line and holds the
# reference implementation (Summarize) this mirrors.
import json


def cudaphase_summary(name, buf, phases, slots=16, sm_slot=None, extra=None):
    """phases is a list of (name, from_slot, to_slot); times are in ns."""
    vals = buf.tolist() if hasattr(buf, "tolist") else list(buf)
    rows = [vals[i:i + slots] for i in range(0, len(vals) - slots + 1, slots)]
    rows = [r for r in rows if any(r)]  # blocks that wrote nothing never ran
    stamps, first, last = set(), None, None
    out = []
    for pname, a, b in phases:
        d = sorted(r[b] - r[a] for r in rows if r[a] > 0 and r[b] > 0)
        for r in rows:
            for s in (a, b):
                if r[s] > 0:
                    stamps.add(r[s])
        out.append({"name": pname, "blocks": len(d),
                    "median_us": d[(len(d) - 1) // 2] / 1e3 if d else 0.0,
                    "min_us": d[0] / 1e3 if d else 0.0,
                    "max_us": d[-1] / 1e3 if d else 0.0})
    ss = sorted(stamps)
    gaps = [y - x for x, y in zip(ss, ss[1:])]
    s = {"name": name, "blocks": len(rows),
         "span_us": (ss[-1] - ss[0]) / 1e3 if ss else 0.0,
         "quantum_us": min(gaps) / 1e3 if gaps else 0.0,
         "phases": out}
    if sm_slot is not None:
        s["sms"] = len({r[sm_slot] for r in rows if r[sm_slot] > 0})
    if extra:
        s["extra"] = extra
    return "CUDAPHASE " + json.dumps(s, separators=(",", ":"))
