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

`inspector-full-comm-hash.patch` is byte-identical to the independently
reviewed producer repair. It changes the buffer to 19 bytes, formats the hash
with `PRIx64`, returns `inspectorBadHash` when `snprintf` fails or would
truncate, and propagates that result from `inspectorFillCommInfo` with
`INS_CHK`. Apply it only to the pinned source before building Inspector.
Keep the patch and resulting plugin SHA256 with qualification receipts; the
embedded Git revision alone does not identify a patched plugin. The reader and
diagnostic identity checks are unchanged. `TestProducerPatchPin` checks the
patch against the line below.

Patch SHA256: `c825c908d737876e370fbdbc0edef6ee6270370388ec34b6dc0aa3016233563e`

Review of the patch accepted source and CPU behavior only (a formatter
extracted from the patched and unpatched sources; no plugin build):
`~/tmp/lib-wt-next/20261010-full-testing-readiness/independent-nccl-producer-review/GRADE.md`
(SHA256 `d29e8164965d9c08f7c9fb3dfd91f6ccbbf2e6eb9587085515aa72705e9b23b7`).

One plugin has been built from this patch and pinned commit, on a Linux
aarch64 host (GB10), with CUDA 13.4 and `make -j2` in
`plugins/profiler/inspector`. It is 232232 bytes with SHA256
`e41cb2f3749b0d60b0fd39ec94bcf03fbc7893492de5403b62748794bea4ac3a`: an AArch64
ELF shared object exporting `ncclProfiler_v5`, needing `libcudart.so.13` and
`GLIBC_2.38`. Review accepted that compile and link only:
`~/tmp/lib-wt-next/20261010-full-testing-readiness/independent-nccl-target-build/GRADE.md`
(SHA256 `1e5dd5659cc149077077d638e82a55f2ba8df6b78b8fcc1da3526695da0aa30f`).
Plugin loading, dependency resolution and native RAS/Inspector identity parity
have not been observed. No x86_64 plugin has been built or reviewed; an x86_64
host needs its own build, pins and review.
