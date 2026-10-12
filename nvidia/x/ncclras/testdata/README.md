# RAS fixture

`status.json` is the literal NCCL 2.29.1 JSON example from NVIDIA NCCL's
`docs/userguide/source/troubleshooting/ras.rst`, commit
`12df1a11afad322be5a204a2db890161cbf8131d` (repository version 2.32.3).
Only documentation indentation was removed and a final newline retained.
It contains one responding rank and one unresponsive rank; it is documentation,
not an observation from this host. Source SHA256:
`5c8863d41777cc395ca30d79d5614939384752417ed52c76af9ff47e16d717d2`.

Source: https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/docs/userguide/source/troubleshooting/ras.rst

`status-2.32.3.json` is constructed, not captured. It was rendered by
transcribing the `rasOutAppend` format strings of `jsonWriteHeader`,
`jsonStartCommunicator`, `jsonWriteRankData`, `jsonStartMissingRanks`,
`jsonWriteMissingRank`, `jsonEndCommunicator` and `jsonWriteFooter` in
`src/ras/client_support.cc` (lines 2116-2213, called from
`rasDumpCommsToJSON` at 2292-2363) at the same commit, which is tag
`v2.32.3-1`. The version string is `NCCL_MAJOR.NCCL_MINOR.NCCL_PATCH` with an
empty suffix from `makefiles/version.mk`, and the five `collective_counts`
keys are `ncclFuncToString` of `ncclFunc_t` 0 through `NCCL_NUM_FUNCTIONS`-1
(`src/include/nccl_common.h`, `src/collectives.cc`). All hashes, hosts, PIDs,
devices, counts, the timestamp and the collection time are invented. It shows
the emitter's layout, including an empty `missing_ranks` array; it is not
evidence that a running 2.32.3 job produces this response. Source SHA256s:
`client_support.cc` `b97f36cef1864940bd7eecd32ebfdf1053841fe85c6cb21fa3e60e1fc935a376`,
`version.mk` `87f07999de3fd29c5b1829d698036082e4cdc6eb16fba6449276af2677d1c8be`,
`nccl_common.h` `f5c19dc634ebfdfdc735a19ef3ef21363eb5f3ff8d238ee6c349b7e7baabfc86`.
Fixture SHA256: `61880717da37499e5e206f31cc60d78225ef5b7b5bc6e2148db4e79e61506a42`.

The same emitter writes `"host": "unknown"` and `"pid": -1` for a rank whose
RAS peer it cannot resolve. Read refuses such a rank as an invalid identity;
neither fixture contains one.

The client implements protocol 2 from `src/ras/client.cc` and
`src/ras/ras_internal.h` at the same revision: CLIENT PROTOCOL, TIMEOUT,
SET FORMAT json, STATUS, then EOF. Other JSON version schemas are explicitly
unsupported. Unknown JSON fields remain in Raw. The separate experimental
2.24.3 text profile is described in legacy-2.24.3.md; its fixture is constructed
from the producer source and does not satisfy authentic capture validation.
NVIDIA's source license is retained in LICENSE.txt.
