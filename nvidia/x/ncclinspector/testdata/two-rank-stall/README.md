# Two-rank joined consumer controls

These files are deterministic, constructed host controls for the public
`Read` → `nccljoin.Join` → `ObservationFromRecord` → `TwoRankAttributor`
consumer path. They are not physical captures, producer receipts, or native
qualification evidence. `manifest.json` records the fixture identity, clock
mapping assumptions, independent completion control facts, expected reports,
and SHA-256 hashes for each JSONL file.

Inspector JSONL supplies rank identity, operation identity, and event-trace
timestamps. It does not supply completion, calibrated cross-host clock facts,
boot/process generation, or a communicator epoch. The manifest's completion
and clock entries are separately constructed control facts bound to the rank
and epoch. The consumer attaches them only for the matching test control;
missing completion stays missing. Neither `kernel_record_ts`, aggregate
duration, publication time, nor peer delay is used to manufacture a stage.

Each rank file contains its own record. The normal pair has equal stage timing;
the delayed pair places rank 1's host enqueue 1,000 ns after rank 0. No test
uses runtime sleeps or current wall time as timing evidence. Mutated controls
in the test change one declared input at a time and retain the original joined
result for inspection.
