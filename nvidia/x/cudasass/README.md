# Real-tool checks

`TestInstalledTools` requires a real CUBIN in `CUDASASS_TEST_CUBIN`. It exercises
text, JSON, graph and register-life output and a known `cu++filt` symbol. Set
`NVDISASM` and `CUFILT` to choose executables explicitly.

On Linux, `TestInstalledCancellation` uses an exec launcher to record the child
PID, waits until `/proc/PID/exe` identifies the selected `nvdisasm`, then cancels
the context. It requires `context.Canceled`, a reaped process, and empty scratch
storage. If the tool finishes before it can be observed, the test fails as
inconclusive. It does not suspend the process or substitute a fake disassembler.

From the `nvidia` module directory:

```sh
TMPDIR="$HOME/tmp" GOWORK=off \
  CUDASASS_TEST_CUBIN="$HOME/tmp/cuobjdump-fixture/addone.cubin" \
  NVDISASM=/usr/local/cuda/bin/nvdisasm \
  go test -count=1 -v -run '^TestInstalledCancellation$' ./x/cudasass
```

The fixture recipe in `x/cudaobjdump/README.md` creates a real CUBIN without
executing GPU code. These tests inspect binaries and do not require a GPU.
Cancellation evidence covers an observed live executable; it does not establish
that a particular instruction or output record had already been processed.
