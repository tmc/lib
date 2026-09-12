# Checking a token-file workload

`ds4-min -prompt-tokens FILE` reads a JSON array without chat rendering or added
special tokens. Parsing rejects empty arrays, nulls, negative IDs, fractions,
and trailing data. After reading model metadata, the command checks IDs against
the loaded tokenizer. It requires context for the prompt and every requested
accepted token before opening the backend.

For the retained 2048-token fixture and two generated tokens, `-ctx 2049` fails
admission and `-ctx 2050` has exactly enough capacity. Use the same context as
the reference run: changing context is a different workload configuration.

## Supplied logits expectations

Two optional flags compare SHA-256 hashes of finite float32 logits encoded as
little-endian words. All float32 bits, including signed zero, are preserved.
There is no built-in expected hash or tolerance. A mismatch returns an error
and prints the measured hash. Malformed or explicitly empty expectations are
rejected during argument parsing.

```sh
# Run only under the current device owner's admission procedure.
./ds4-min -backend "$BACKEND" -model "$MODEL" -ctx 4096 \
  -prompt-tokens scripts/testdata/oracle-workload-tokens2048.json -tokens 2 \
  -expect-prefill-logits "$MEASURED_PREFILL_SHA256" \
  -expect-final-logits "$MEASURED_DECODE2_SHA256"
```

The prefill check runs after `session.Sync`. The final check runs after
generation and also requires exactly `-tokens` accepted tokens. An early stop
cannot satisfy final acceptance even if its logits match the supplied hash.
Without the final expectation, `-tokens` remains an upper limit and generation
may stop early.

With either check enabled, stderr records the resolved backend, model path,
context, requested count, prompt count and canonical token hash (little-endian
uint32 IDs), and each accepted token ID. The flags do not hash the model file,
certify a reference receipt, or establish device/host qualification. Retain the
source revision, executable hash, model hash, profiler version, and owner-run
admission/cleanup receipts externally with the expectations.

Do not substitute the 64-token Metal region anchor, the 2060-token deep-regime
anchor, or the 2048+128-token oracle workload's final hash for this 2048+2 run.
At base revision 8fd3f4e9, the latter's hashes have no retained measurement
provenance, and `measureOracleWorkload` hardcodes 128 non-EOS decodes. Its third
argument is an EOS ID, not a decode count. Supply measured values from the
matching workload rather than changing constants after a mismatch.

The requested decode positions, 2048 and 2049, precede the compressed indexer
transition at position 2051. A four-token extension can test that boundary,
but needs a separate receipt and final hash; it does not replace the requested
two-token profile.

## Capture and comparison

Go's `-cpuprofile`, `-trace`, and `-spans` flags can accompany the expectations.
Hashing and accepted-token logging add CPU work when checks are enabled. Retain
the flag set and do not label these runs as uninstrumented timing. Reported
prefill and generation durations exclude their trailing digest checks;
generation includes enabled accepted-token logging.

For Nsight, export SQLite and use `examples/cmd/profile2trace`. For a matching
Metal receipt, use its `-metal` input. Merging host and device sources requires
measured clock offsets; separate executions cannot be aligned into one capture
by matching their first interval.

Use `examples/cmd/profile2trace/check.sql` with the offline Perfetto trace
processor to compare source kernel counts with parsed slices and inspect
unfinished slices and parser errors. Reading a trace validates the converter;
live capture qualification additionally needs owner-run provenance, a complete
workload window, timing-domain evidence, and cleanup results.
