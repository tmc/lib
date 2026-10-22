# Output agreement gate

Performance captures are not comparable until the arms are shown to compute
the same outputs. `output_agreement.py` compares retained little-endian
float32 logits and can also compare teacher-forced state directories.

First establish a bound from a C-vs-C control that changes only arithmetic
order (for example a different prefill chunk size or graphs-off), before
looking at C-vs-Go. Two identical C runs are a zero control, not a usable
spread:

```sh
python3 bench/output_agreement.py c1.f32le c2.f32le --control-c c1.f32le c2.f32le --vocab 129280
```

The control-only invocation exits 3 (`NO_VERDICT`). If it reports
`control_zero: true`, collect a perturbed control rather than registering a
zero KL bound. Record the nonzero control's top-1 agreement and final-position
KL in the campaign protocol. Then run the gate with those preregistered
limits (do not derive limits from the Go result):

```sh
python3 bench/output_agreement.py c.f32le go.f32le --vocab 129280 \
  --gate fixed --min-top1 0.99 --max-kl 0.001 --json
```

The fixed gate must be named explicitly. For a C-variance-relative gate, name
the control and preregister its multiplier explicitly:

```sh
python3 bench/output_agreement.py c.f32le go.f32le --vocab 129280 \
  --gate relative --control-c c1.f32le c2.f32le \
  --relative-kl-factor 2 --json
```

Relative mode refuses a bit-identical C-vs-C control. The tool exits 2 on a
gate failure and never exits 0 without a verdict.
`--teacher-left` and `--teacher-right` compare state-`NN`.json files under the
same forced token history. Every state must carry the same
`forced_tokens_sha256`; a legacy `token_ids_sha256` alone is not
teacher-forced provenance, and missing or non-contiguous steps fail closed.
`accepted_decode_count` is zero-based: the seventh generated token is state 6.
The report includes the top-two margin and classifies that token as `near_tie`
only when the C margin is smaller than the measured C-vs-Go max logit error;
otherwise it is a `mismatch` to localize.

Do not call a final-only dump a 99% top-1 result: with one position it is one
bit. Prefer sampled prompt positions (for example every 64th and the last 32)
and record their position indices. Full 2048-position dumps are intentionally
not required by this tool.

The raw dumps and the C-vs-C control must be retained with the capture
manifest. Each raw dump requires its adjacent JSON sidecar; the tool checks
phase, position, vocabulary, and `token_ids_sha256` before comparing values.
A per-arm digest is not a cross-implementation agreement gate.

For meaningful top-1 coverage, use a teacher-forced run that emits one state
row per accepted token (the first 32 C tokens are the registered workload).
Both runners must use the same int32 little-endian token-history hash and the
same state numbering.

Run the no-GPU mutation checks after changing this tool:

```sh
python3 bench/output_agreement_test.py
```
