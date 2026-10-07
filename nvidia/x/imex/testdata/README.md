# IMEX specification profile

`dcgm-spec-580.30.json` is the exact `actualSampleJson` literal in
NVIDIA/DCGM revision `64df9f894541e426e416131a9820cae97aa4dd81`,
`dcgmlib/src/tests/DcgmImexManagerTests.cpp`, lines 283–296:
https://github.com/NVIDIA/DCGM/blob/64df9f894541e426e416131a9820cae97aa4dd81/dcgmlib/src/tests/DcgmImexManagerTests.cpp#L283

The upstream test describes it as a specification example. It is not a measured
capture from our hardware. Node version 580.30 is a producer value; it is not
used to infer CLI format compatibility. `DCGMSpec58030` selects this pinned
profile explicitly. Node IDs are canonical unsigned decimal integers, bounded
to uint64 by this parser; this is a reader limit, not a daemon ABI claim.

Official command documentation advertises finite `nvidia-imex-ctl -N -j`:
https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/cmdservice.html

No public versioned socket protocol or channel-count schema was established.
Raw bytes and unknown fields are retained. Missing/empty version and hostname
remain unavailable. Connection targets absent from the node map are retained,
not synthesized as members. Reported timestamps lack a timezone and stay text.
