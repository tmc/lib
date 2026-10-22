"""Decode a cudaprof capture journal and split a kernel's launches by call site.

The captures behind PARITY-LEDGER.md carry a per-record journal alongside the
pprof. The pprof is an aggregate with no timestamps and no per-launch records:
decoding is done against `*.journal.gz` and NOT the `.pprof`. That distinction
cost this campaign a wrong "the journal is missing, this needs a new capture"
claim -- see the correction under "The dense table merges a failing class with
a passing one" -- so it is worth restating here, next to the code.

Two things this decoder does that the ad-hoc ones did not.

It counts what it drops. The earlier `e0.py` wrapped json.loads in a bare
`except Exception: pass`, so a malformed record vanished without trace and the
record count read as a complete census. A decoder that fails open about its own
losses is the same defect this ledger spends twenty sections finding elsewhere.

It refuses a site attribution it cannot justify. Launch order is the only thing
separating same-grid call sites, so `sites()` requires the whole per-layer grid
pattern to hold in EVERY layer before assigning any label, and raises otherwise.
Order-based attribution is an assumption; this makes it a checked one.

Usage:
    python3 capture_journal.py <capture.journal.gz> [--sha256 HEX]
    python3 capture_journal.py <capture.journal.gz> --kernel ds4_dense_q8_d2r \
        --pattern 128,64,4096,128,128,128,128,128,128,128,128,512,256,256,512 \
        --names QA,KV,QB,attnOutA*8,attnOutB,gate,up,down
"""

import argparse
import collections
import gzip
import hashlib
import json
import sys


def records(blob):
    """Yield every {"kind": ...} object in blob, plus a count of what failed.

    Returns (records, malformed, truncated). A brace-depth scan that respects
    string escapes; the journal is concatenated objects, not newline-delimited,
    and is not valid UTF-8 throughout, so it is scanned as bytes.
    """
    out, malformed, truncated = [], 0, 0
    i, n = 0, len(blob)
    while True:
        start = blob.find(b'{"kind":', i)
        if start < 0:
            return out, malformed, truncated
        depth, quoted, escaped, end = 0, False, False, -1
        for j in range(start, n):
            c = blob[j]
            if quoted:
                if escaped:
                    escaped = False
                elif c == 0x5C:
                    escaped = True
                elif c == 0x22:
                    quoted = False
            elif c == 0x22:
                quoted = True
            elif c == 0x7B:
                depth += 1
            elif c == 0x7D:
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end < 0:
            truncated += 1
            return out, malformed, truncated
        try:
            out.append(json.loads(blob[start:end]))
        except ValueError:
            malformed += 1
        i = end


def kernels(recs, name):
    ks = [r for r in recs
          if r.get("kind") == "activity"
          and r.get("subtype") == "kernel"
          and r.get("name") == name]
    ks.sort(key=lambda r: r["start"])
    return ks


def sites(ks, pattern, names):
    """Label each launch by its position in a repeating per-layer pattern.

    Raises if any layer deviates, if the launch count is not a whole number of
    layers, or if any record has a non-positive duration. Refusing is the point:
    a partial match means the pattern is not the one the source executes, and a
    label assigned anyway would be a guess wearing a measurement's clothes.
    """
    if len(pattern) != len(names):
        raise ValueError("pattern has %d entries, names %d" % (len(pattern), len(names)))
    if not ks:
        raise ValueError("no launches to attribute")
    if len(ks) % len(pattern):
        raise ValueError("%d launches is not a whole number of %d-launch layers"
                         % (len(ks), len(pattern)))
    layers = len(ks) // len(pattern)
    for layer in range(layers):
        group = ks[layer * len(pattern):(layer + 1) * len(pattern)]
        got = [r["grid_x"] for r in group]
        if got != pattern:
            raise ValueError("layer %d grid pattern is %s, want %s" % (layer, got, pattern))
        for r in group:
            if r["end"] <= r["start"]:
                raise ValueError("layer %d has a non-positive duration" % layer)
    by = collections.defaultdict(list)
    for i, r in enumerate(ks):
        by[names[i % len(names)]].append(r)
    return by, layers


def expand(spec):
    """"a,b*3" -> ["a","b","b","b"]. Keeps a 15-entry pattern readable."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "*" in part:
            label, count = part.rsplit("*", 1)
            out.extend([label] * int(count))
        else:
            out.append(part)
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("journal")
    ap.add_argument("--sha256", help="refuse to decode unless the file hashes to this")
    ap.add_argument("--kernel", default="ds4_dense_q8_d2r")
    ap.add_argument("--pattern", help="comma-separated grid_x per layer")
    ap.add_argument("--names", help="comma-separated labels, name*n to repeat")
    args = ap.parse_args(argv[1:])

    raw = open(args.journal, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    if args.sha256 and digest != args.sha256:
        print("sha256 is %s, want %s" % (digest, args.sha256), file=sys.stderr)
        return 1
    print("journal %s\nsha256  %s" % (args.journal, digest))

    recs, malformed, truncated = records(gzip.decompress(raw))
    print("records %d  malformed %d  truncated %d" % (len(recs), malformed, truncated))
    if malformed or truncated:
        print("WARNING: the census below is missing records; it is a sample, not a count.")

    ks = kernels(recs, args.kernel)
    print("%s launches %d" % (args.kernel, len(ks)))
    if not args.pattern:
        return 0

    pattern = [int(x) for x in args.pattern.split(",")]
    by, layers = sites(ks, pattern, expand(args.names))
    print("complete layer patterns %d" % layers)
    print("%-10s %9s %14s %14s" % ("site", "launches", "total_ms", "ms/launch"))
    total = 0.0
    for name in dict.fromkeys(expand(args.names)):
        rs = by[name]
        ms = sum(r["end"] - r["start"] for r in rs) / 1e6
        total += ms
        print("%-10s %9d %14.6f %14.9f" % (name, len(rs), ms, ms / len(rs)))
    print("%-10s %9d %14.6f" % ("TOTAL", len(ks), total))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
