# Campaign samples and spread

`cudaprof qualify` scores four-run C1/G1/G2/C2 blocks. Each reported
`block_samples` is one retained block; `run_samples` is four times that
count. Kernel launches and the two pairs within one block do not increase
this sample count. The empirical minimum, maximum, and median absolute
deviation of block means describe observed spread, not confidence limits.
Blocks can share thermal state and other drift; the report makes no
independence claim. These fields do not change the existing noise or sign
gates. Excluding any arm drops the whole block and preserves all captures
and exclusion reasons in the receipt.

To reproduce a small campaign, capture at least three blocks, preserving
C1/G1/G2/C2 order in each. Use the same capture options and explicit workload
arguments throughout. Write a manifest with one line per run:

```
B1 C1 B1-C1.pprof
B1 G1 B1-G1.pprof
B1 G2 B1-G2.pprof
B1 C2 B1-C2.pprof
```

Repeat those lines for B2 and B3, with the corresponding paths, then run:

```
cudaprof qualify -json -warmup-until='first_steady_kernel' campaign.manifest
```

Choose a real kernel boundary shared by both arms. Keep failed gates and
unchecked identity/order or host-load evidence in the result. This procedure
does not establish output equality: retain and compare deterministic workload
outputs separately before interpreting timing. Run a corresponding campaign
without collection to estimate profiler overhead; the captured device span
alone cannot measure that overhead. No GPU campaign was run for this change.

`cudaprof ab` deliberately does not acquire a GPU lock. The inspected local
GPU lock wrapper uses tickets to admit waiters to a directory mutex;
within each lane tickets are FIFO, short jobs have preference, and batch jobs
age to avoid starvation. Dead-process tickets can be reaped, live holders
are not removed merely by age, and configured time limits terminate the
held command. Hold the shared lock once around the entire campaign script,
including profiled and unprofiled controls, rather than reacquiring between
arms or nesting acquisition inside it. This describes the inspected local
script; deployments using another lock must verify its admission rules.
