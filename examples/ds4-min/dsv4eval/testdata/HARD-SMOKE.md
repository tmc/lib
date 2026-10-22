# Hard-smoke corpus

Extracted without text reconstruction from antirez/ds4 commit
`11e8cad4473716a41baf4d62e92ed9764a00fce6`, `ds4_eval_cases.c`, using
`extract_smoke.c`. The subset is the twelve entries carrying
`EVAL_SUITE_HARD_SMOKE`; choices, answers, aliases and budgets are unchanged.

Upstream `EVAL_DATA.md` attribution:

- MMLU-Pro, TIGER-AI-Lab/MMLU-Pro revision
  `f418b116db00b065c2aea046518d8fcf74d39872`, Apache-2.0.
- OlympiadBench, Hothan/OlympiadBench revision
  `91184b52131e7fc9455fef848035173aea8cc01a`, Apache-2.0.
- LiveBench, livebench/reasoning revision
  `6fc6498a5dfba553f69f4413feabade1f1a2d384`, Apache-2.0.
- NIST Juliet 1.3, SARD suite 112 archive SHA-256
  `ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb`,
  CC0-1.0/public domain. These are upstream's locally written reductions.

Generation copies upstream's Apache-2.0.txt into the external output. This subset is an integration
test, not an official benchmark distribution or a leaderboard score.

Questions and licenses are not committed or embedded. From dsv4eval, generate:

```
DS4_ORACLE_SOURCE=/path/to/pinned/ds4 \
DS4_EVAL_CORPUS_OUT=/path/outside/git/new-corpus go generate
```

Generation rejects unqualified sources, extractor changes, existing output,
and output inside any Git worktree. The external manifest records source and
extractor identity, extraction binary hash and build command. Binary hashes
are build receipts, not canonical corpus identity. The canonical corpus hash
pins all twelve questions, answers and budgets. Supply `-corpus` to ds4-eval
or `-eval-corpus` to ds4-min; missing external admission fails closed.

Runtime qualification requires matching model SHA, context, temperature 0,
HIGH thinking, soft reserve 1024, hard reserve 512 and close rank 3. Use each
case's declared token budget, with a context large enough for prompt+budget.
Run the oracle and Go under gpu-lock, preserve raw traces and binary hashes,
then cross-regrade and compare all twelve statuses and close kinds. A partial
run cannot qualify. Host fixture tests only test controller/trace behavior.

## Raw oracle trace admission

The pinned C trace records `temperature: 0` and
`think_mode_requested: high`; it records neither a source build nor a model
hash. `ds4-eval -import-smoke RAW -import-provenance RECEIPT.json
-oracle-binary ORACLE -model MODEL` hashes all retained artifacts before
adding normalized provenance fields. The receipt must contain `build`,
`binary_sha256`, `model_sha256`, `raw_trace_sha256`, `raw_model_path`,
`backend`, `ctx`, `temp` and `think`. `temp` is `0`, `think` is `HIGH`.
The source revision is independently attested by the build receipt; a binary
hash alone cannot establish it. The imported trace retains the original case,
output, status and summary suffix byte-for-byte, and binds the original hash.
Do not discard the raw trace or the receipt after import.

Go trace writers emit matching per-case source/id fields as well as the case
header: the upstream regrader requires these fields. Header/field mismatches
are rejected. Regrading uses exact counted blocks, including upstream's
uncounted LF separator for bodies without a final LF. Embedded marker text
and long lines are data; malformed lengths and missing summaries are errors.
This strict framing rejects malformed delimiter-only traces rather than
silently using the upstream parser's legacy delimiter fallback.

An opt-in CPU test compiles the actual hash-qualified upstream regrader:
`DS4_ORACLE_SOURCE=/path/to/pinned/ds4 go test -run TestUpstreamReadsGeneratedTrace`.
It checks a generated Go trace with embedded markers and a long line, then
removes its source field and requires an upstream parse error.

The existing strict smoke comparator is stronger than Spec08 G7: it requires
at least eleven correct cases on each arm, as well as eleven agreeing statuses
and twelve agreeing close kinds. Spec08 requires the agreement counts, not
eleven correct answers per arm. This implementation leaves that stronger gate
unchanged; failure of the stronger gate must not be described as a different
numerical or semantic result than the raw traces support.
