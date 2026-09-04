# NVIDIA binding feature probe

This program exercises a safe lifecycle or query feature in every package in
`github.com/tmc/lib/nvidia`, including the `x/nvlshm`, `x/nvlshm/tcpboot`, and
`x/moonep` packages. It does not change GPU configuration, create MIG
instances, or change Fabric Manager partitions.

Run it from this directory:

    GOWORK=off go run .

The output labels a library unavailable when the corresponding shared library
or host service is absent. That is distinct from a successful probe. For a
complete kernel workflow, run `GOWORK=off go run .` from
`examples/nvidia/simple-addition`.

Device pointers are passed only to the CUDA calls that consume them; their
owning allocations remain live until those calls return.
