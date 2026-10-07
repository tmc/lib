# Constructed legacy STATUS fixture

legacy-2.24.3.txt is constructed, not a measured NCCL response. Its headings,
field order, status and error spellings follow NVIDIA/nccl commit
6aae37927840b2bd7b7d42d2f0050e75f88ee97f (v2.24.3-1),
src/ras/client_support.cc lines 471–536, 863–880, 1017–1063 and 1207.
The counts describe a uniform two-process job solely for parser tests.
Protocol 2 is not a product version. This profile does not infer individual
ranks or timeout counts from aggregate text.

Source: https://github.com/NVIDIA/nccl/blob/6aae37927840b2bd7b7d42d2f0050e75f88ee97f/src/ras/client_support.cc

The stored text ends with one newline after the warning separator. Every test
and fuzz consumer uses legacyFixture to append the protocol's final empty line,
reconstructing the required `Warnings\n========\n\n` wire ending. The parser's
empty-line requirement and exact response comparisons remain unchanged.
