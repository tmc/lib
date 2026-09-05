# cuobjdump validation

Host tests use subprocess fixtures. The installed-tool test requires one known
CUBIN and one known PTX image, packaged together. It compares extracted CUBIN
bytes directly and assembles both PTX sources with the same `ptxas` to compare
compiled bytes: fatbinary removes comments and normalizes PTX whitespace.

From the `nvidia` module directory, with CUDA 13 tools on PATH:

```sh
dir="$HOME/tmp/cuobjdump-fixture"
mkdir -p "$dir"
nvcc -arch=sm_121 --cubin -Xptxas=-v \
  x/cudaobjdump/testdata/addone.cu -o "$dir/addone.cubin"
nvcc -arch=compute_121 --ptx \
  x/cudaobjdump/testdata/addone.cu -o "$dir/addone.ptx"
fatbinary --64 --create="$dir/addone.fatbin" \
  --image3=kind=elf,sm=121,file="$dir/addone.cubin" \
  --image3=kind=ptx,sm=121,file="$dir/addone.ptx"
TMPDIR="$dir" GOWORK=off \
  CUDAOBJDUMP_TEST_BINARY="$dir/addone.fatbin" \
  CUDAOBJDUMP_TEST_CUBIN="$dir/addone.cubin" \
  CUDAOBJDUMP_TEST_PTX="$dir/addone.ptx" \
  CUDAOBJDUMP_TEST_ARCH=sm_121 \
  go test -count=1 -v ./x/cudaobjdump
```

Use matching architecture values throughout for another target. `CUOBJDUMP` and
`PTXAS` select explicit executable paths. These checks compile and inspect device
code; they do not execute it and do not require a GPU.

The test retains tool identity, arguments and diagnostics in its output, verifies
input preservation and scratch cleanup, exercises SASS/ELF/resource inspection,
and requires diagnostics for a truncated ELF. For resource validation, compare
the reported register count with the independent `ptxas -v` compilation log.
Changing an instruction in the PTX reference must make the comparison fail;
formatting-only differences may compile to identical bytes.
