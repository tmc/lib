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

`TestInstalledKnownJSON` separately checks the retained `sass.cubin` from the
native JSON fixture, using nvdisasm 13.0.85. Set `CUDASASS_TEST_KNOWN_CUBIN` to
that file and `NVDISASM` to the pinned executable. The required CUBIN SHA-256 is
`cc5746145dc7f00f91c458925f3513a4e4c91ff5a252be744272501055d41027`.
This fixture uses the source in `testdata/README.md`, not the broader-indexing
cuobjdump example. The test compares metadata and complete function records,
including instruction operands, padding and control-flow attributes. Host
negative controls alter valid JSON and require these changes to be detected.
Neither test executes the kernel.
