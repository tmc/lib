# GEMM and all-reduce

This composition leaves the 2x2 FP32 GEMM result on the device between
cuBLAS and NCCL. On one GPU, the all-reduce is an identity operation; increase
the communicator size and use one process per GPU for a multi-GPU reduction.
The device allocations remain live across both library calls and are released
only after the result is copied back.

    GOWORK=off go run .
