# Inspector fixtures

The three `.json` files are the literal collective, dump-statistics and verbose
examples from NVIDIA NCCL `plugins/profiler/inspector/README.md`, commit
`12df1a11afad322be5a204a2db890161cbf8131d` (NCCL 2.32.3).
README SHA256: `505fd2905cd8f16597214971f2eb69396500b8c1a8f92c924e58c9e4a14a35f6`.
The `.jsonl` counterparts compact whitespace only and append newline framing;
keys and values are unchanged. These are documentation excerpts, not complete
captures and not measurements made here. In particular the statistics example
reports dropped records, which the parser preserves.

Source: https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/plugins/profiler/inspector/README.md

The point-to-point test record is explicitly constructed using the real
`inspector_json.cc` fields, not presented as an upstream or native fixture.
The reader supports format v4.1 only, preserving raw unknown fields and separate
communicator/rank/collective/P2P sequence domains. It does not certify a complete
session or reconcile ring-buffer statistics against these excerpts.
NVIDIA's source license is retained in LICENSE.txt.

## Full communicator hashes

At the pinned NCCL revision above, `NCCL_COMM_HASH_LENGTH` is 17. A complete
64-bit hexadecimal hash needs 19 bytes: `0x`, 16 digits, and a terminator.
The producer can truncate the last byte; the native RAS/Inspector test rejects
that identity mismatch.

`inspector-full-comm-hash.patch` changes the producer buffer to 19 bytes. Apply
it only to the pinned source before building Inspector. Keep the patch and
resulting plugin SHA256 with qualification receipts; the embedded Git revision
alone does not identify a patched plugin. The reader and diagnostic identity
checks are unchanged.
