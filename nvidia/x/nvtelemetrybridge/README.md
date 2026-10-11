# Pinned telemetry snapshots

Create or open a non-per-CPU hash map with `KeySize`, `ValueSize` and
`MaxEntries=1`. Inspect all map information before passing it to `New`.
`ebpfhost/bpfload.Object` satisfies `Store`; copy its map ID, layout and BTF
fields into `Info`. BTF-backed and array maps are refused.

`Publish` replaces key `uint32(0)` with one complete snapshot. Independent
readers call `Read(maxAge)`. Generation advances only after a successful update;
a random 128-bit owner changes when a publisher restarts. A consumer must not
combine publications with different owners or epochs. There is no precise clock
mapping or sensor freshness inference.

The caller enforces an exclusive publisher lease and chooses an accessible pin
namespace. For cleanup, use a private owner-controlled directory and a
cooperative namespace lock. `RemovePin` must verify the kernel map ID while
holding that lock. Reopening an ID then unlinking a path does not protect against
hostile concurrent replacement. `OwnPin=false` and `RetainPin=true` never unlink.
Permission, missing-map, update and cleanup errors remain explicit.

The version 1 layout is little-endian; `layout.h` declares matching C structs
and size assertions. Header (208 bytes), eight device slots (256 bytes each),
and 128 reading slots (256 bytes each) total 35024 bytes. Unused bytes must be
zero. Strings are bounded and NUL-terminated. The bridge exports UUID/MIG
identity and reading values/timing/presence, not the complete JSONL manifest.
Oversized device sets are refused rather than truncated. No BTF relocation or
CO-RE compatibility is assumed. BPF programs may read the hash value but must
not mutate it in place.
