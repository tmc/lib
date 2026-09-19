# tklayout.json

ThunderKittens' register-tile and shared-tile layouts, measured on a GPU.

Produced by `kittensfixture/probe.cu` in the thundergophers repository, at
commit 4f31513. Built on the GPU host with CUDA 13.4
(`/usr/local/cuda/bin/nvcc`) for sm_121a with KITTENS_SM120, against
ThunderKittens 67845f5. Copied here byte for byte; sha1
f7f30f61189236cf5e83ebbc57ad892bdbf4dbca.

Two maps per shape, both written through ThunderKittens' own coordinate
indexer and read back through its raw-storage indexer, so the arithmetic
recorded is TK's and not a transcription of it:

  - `storage[i]` is the logical (row, col) TK placed at raw storage element
    i. Layout plus swizzle; the oracle for `SwizzledTile.ByteOffset`.
  - `regs[lane][k]` is the logical (row, col) that lane holds in its k-th
    element slot after `warp::load`. The oracle for `Fragment`.

`regs` is the half that cannot be obtained by reading the headers. At one and
two bytes TK loads through `ldsm4`, so the lane assignment is `ldmatrix`
hardware behaviour, which no header describes. That is why this file exists
rather than a second transcription: a transcription can only reproduce the
reading that produced it.

Only the row layout is measured. `warp::load` produces one layout, so
`FragmentCol` has no witness here.

To add a shape, ask the fixture's author rather than extrapolating from the
ten recorded; adding a row to probe.cu's main is a one-liner.
