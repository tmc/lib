# RAS fixture

`status.json` is the literal NCCL 2.29.1 JSON example from NVIDIA NCCL's
`docs/userguide/source/troubleshooting/ras.rst`, commit
`12df1a11afad322be5a204a2db890161cbf8131d` (repository version 2.32.3).
Only documentation indentation was removed and a final newline retained.
It contains one responding rank and one unresponsive rank; it is documentation,
not an observation from this host. Source SHA256:
`5c8863d41777cc395ca30d79d5614939384752417ed52c76af9ff47e16d717d2`.

Source: https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/docs/userguide/source/troubleshooting/ras.rst

The client implements protocol 2 from `src/ras/client.cc` and
`src/ras/ras_internal.h` at the same revision: CLIENT PROTOCOL, TIMEOUT,
SET FORMAT json, STATUS, then EOF. Other JSON version schemas are explicitly
unsupported. Unknown JSON fields remain in Raw. The separate experimental
2.24.3 text profile is described in legacy-2.24.3.md; its fixture is constructed
from the producer source and does not satisfy authentic capture validation.
NVIDIA's source license is retained in LICENSE.txt.
