# Repeated campaign runner

Run `cudaprof campaign` inside one existing `gpu-lock` hold. The runner refuses
an unset `GPU_LOCK_HELD` and does not acquire a nested lock. This declaration
cannot exclude GPU users that ignore the shared lock.

```sh
cudaprof campaign -o "$HOME/tmp/gemma-campaign" \
  -base './reference input.bin' -target './candidate input.bin' \
  -blocks 5 -timeout 10m -expected-output expected.stdout \
  -stamp-file source=source.tar -stamp-file input=input.bin \
  -stamp-file model=model.manifest \
  -iteration-marker 'arg_reduce' \
  -base-expect-markers 33 -target-expect-markers 33 -skip-iterations 2 \
  -- --inject /path/to/agent.so
```

The example describes a future run, not a collected experiment. Supply exact
expected stdout and stamps for the actual source, input and optional model.
A model manifest stamp verifies its bytes; it does not recursively hash the
files named by a manifest. Command strings are split without shell expansion.
Wrapper commands such as `env` stamp that wrapper executable; stamp scripts
and downstream executable files explicitly when using wrappers.

Each independent block runs reference, target, target, reference. Every arm
runs once profiled and once as an ordinary process, with ordinary/profiled
then profiled/ordinary order balanced within each arm. Each launch has its own
timeout. There are 3–100 blocks and a maximum one-hour per-launch timeout.
Use `-domain=driver` (the default) or `-domain=runtime` for the observed
submission identity. The runner records the selected domain and never falls
back to another domain to make a missing relationship match.

Source, input, command binary, additional stamps and expected output are hashed
before the campaign and checked before and after every launch. The inherited
environment is recorded as a digest; capture files also retain the collector's
actual target environment receipt. Environment values are not published.

Individual stdout, stderr, collector warnings and captures are retained. Any
process error, output mismatch, artifact change, incomplete capture, failed
iteration count/stationarity gate, or unqualified shape comparison excludes
the whole block. Both paired shape comparisons use only kernels inside the
retained steady iteration window. Work crossing either selected window edge
is refused rather than clipped or silently omitted. Failed blocks and reasons remain in
`campaign.json`; fewer than three qualified blocks refuses an interval.

The device-window effect compares the block's two target steady device windows
with its two reference windows. It is separate from summed kernel duration.
The deterministic percentile bootstrap resamples whole independent block
effects, not launches, and reports a conditional 95% interval. Its unverified
assumption is independent exchangeable blocks; ABBA removes linear within-block
drift but does not prove that assumption. Ordinary/process control ratios
measure end-to-end launch, injection and collection overhead. They do not
isolate kernel overhead or establish a slowdown cause.

CPU-only orchestration and refusal tests:

```sh
cd nvidia
go test ./x/cmd/cudaprof -run '^TestCampaign' -count=1
```

For the checked-in Go/Python control wrappers, build the Go wrapper with a
dynamic loader so preload injection can observe its parent generation:

```sh
cd nvidia
CGO_ENABLED=1 GOWORK=off go build -ldflags=-linkmode=external \
  -o "$HOME/tmp/spin-go-dynamic" ./x/cmd/cudaprof/testdata/spin-go
readelf -l "$HOME/tmp/spin-go-dynamic"
```

The ELF must contain an `INTERP` segment. An ordinary static Go wrapper can
launch a finalized native child while leaving no matching parent injection
handshake; that capture stays refused. Dynamic linking permits the parent
bootstrap and does not relax handshake/finalization requirements. Instrument
application ranges in the native submitting thread, and stamp the native child
binary as an additional artifact. A successful CPU wrapper build is not a GPU
capture validation.

Excluded blocks omit delivery and overhead effects rather than report zero.
A qualified block's unavailable overhead arm is null, never a measured zero.
