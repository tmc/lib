# laya-infer

A Go + CUDA port of [laya.cpp](https://github.com/shpati/laya.cpp), the
single-file C implementation of the Laya decision agent: a ModernBERT-large
encoder (28 layers, d=1024, 16 heads) followed by a two-layer decision head, an
option-marker scorer and an action head.

It answers structured questions about a state:

    laya-infer MODEL_DIR request.json
    laya-infer MODEL_DIR request.json --exact
    laya-infer MODEL_DIR request.json --lines
    laya-infer MODEL_DIR request.json --pprof FILE
    laya-infer MODEL_DIR --tokenize TEXT
    laya-infer MODEL_DIR --list-tensors
    laya-infer MODEL_DIR --serve [PORT] [--bind ADDR]

`MODEL_DIR` holds `model.safetensors`, `tokenizer.json`, `config.json` and
`rl_agent_config.json` from `convaiinnovations/laya`. A request is

    {"state": ..., "questions": {"id": {"type": "choice", "instructions": ..., "criteria": ...}}}

and each question is answered as `choice`, `score` or `noul`.

## Structure

| File | Contents |
| --- | --- |
| `forward.cu` | Encoder and head kernels, compiled at startup with `x/nvrtcx` |
| `kernels.go` | Module compilation and entry-point lookup |
| `config.go` | Hyper-parameters, with the reference's defaults |
| `weights.go` | safetensors load and upload via `x/cudasafetensors` |
| `json.go` | Ordered JSON with Python's `json.dumps` formatting |
| `sequence.go` | Question parsing and input assembly |
| `forward.go` | The batched GPU forward pass |
| `perf_test.go` | `cuptibench` benchmarks of the forward pass, by phase |
| `head.go` | Option scorer and action head, on the host |
| `predict.go` | Request handling and reply rendering |
| `main.go` | CLI and HTTP server |

Matrix products go through `cublas.SgemmEx`; everything else is a kernel in
`forward.cu`. The projection and MLP output products accumulate into the
residual with beta=1, so no separate add kernel follows them. Attention is
`FlashAttention`, one kernel that keeps a 64-query tile of scores in registers
and shared memory and the softmax online, with both products on the tensor
cores as `mma.sync` tf32. `--exact` splits it back into `GemmStridedBatchedEx`
for QK^T and for PV with `AttnSoftmax` in between, and drops the products to
float32. Unlike the reference, every question in a request is padded into one
batch and evaluated together.

## Performance

`perf_test.go` measures the forward pass with `x/cuptibench`, which names each
phase so the region table says where the time goes:

	LAYA_MODEL=$HOME/laya go test -run '^$' -bench ForwardRegions -benchtime 20x

`BenchmarkForwardWhole` and `BenchmarkForwardWholeExact` time the same pass in
each precision, so the speedup below can be reproduced from the tree.

On a GB10, four questions padded to 388 tokens take 49 ms of GPU time, against
140 ms for `--exact`. Both halves of that 2.9x are in the matrix products: the
tf32 tensor-core kernels replace `magma_sgemmEx`, and the fused attention never
writes the 16 by 388 by 388 score matrix to DRAM, which takes the attention core
from 32 ms to 6 ms. The elementwise kernels are the same either way and already
run at memory bandwidth. Wall clock moves far less, because loading the weights
costs about 1.5 s whichever way the pass runs.

### Against the C reference

[laya.cpp] built the way its README says (`cc -O3 -march=native -fopenmp laya.c
-o laya -lm`), given the same weights and the same request on a 20-core GB10
host:

	                     per request   one-shot   model load
	laya.cpp                28885 ms   30178 ms      1075 ms
	laya-infer                 74 ms    1801 ms      1535 ms
	laya-infer --exact        159 ms    1890 ms      1521 ms

Per request is the median of a warm `--serve` process answering `POST /predict`;
one-shot is the median of five CLI runs, each of which starts a process and
loads the weights to answer once. The columns disagree by more than twenty times
because a one-shot run is mostly fixed cost: laya.cpp loads faster, having no
CUDA context and no cuBLAS to initialize, and only what happens after the load
is a comparison of the two implementations. Of laya-infer's 74 ms, 49 ms is
device time; the rest is JSON, tokenization, launch and synchronization, which
both precisions pay alike.

### Nsight Systems

`newModel` opens a profiler capture range and `Close` ends it, so a timeline of
the inference itself — not of weight loading — comes out of:

	nsys profile -t cuda-sw --capture-range=cudaProfilerApi ./laya-infer $HOME/laya request.json

Both flags are needed on this hardware. Without the capture range, nsys arms its
CUDA injection asynchronously and a run that reaches cuBLAS first records
nothing; the range makes the start deterministic. Without `cuda-sw`, the
hardware trace path drops every kernel on the GB10 iGPU ("incomplete CUPTI
events dropped"), because the GPU timestamps are not back by the time the
session stops.

Nsight Compute needs no flags to work, but honors the same range: `ncu
--profile-from-start off` profiles the inference kernels and skips whatever
ran before them.

The range also closes the process: [driver.ProfilerStop] flushes the injected
library, which otherwise never gets the chance — it flushes from a C atexit
handler, and the Go runtime exits through exit_group without running one.

### Source lines

`--lines` answers the request under CUPTI PC sampling and reports the device
time by line of `forward.cu`:

	laya-infer $HOME/laya request.json --lines

The hardware samples a warp scheduler's program counter every 2^16 cycles
(`--lines=N` for 2^N) and reports each sample as a kernel name and a byte
offset into that kernel. `nvdisasm` prints the same offsets with the line each
instruction came from, so the two join with no relocation. The join holds only
if the module that ran is the module that was disassembled, so `--lines`
compiles once through `cudalineinfo` and runs the model on that module.

Of 95969 samples in one request, 38749 landed on a line of `forward.cu`:

	forward: 95969 samples by program line
	  SITE            SOURCE                                                  SASS  SAMPLES  SHARE
	  forward.cu:10   return 0.5f * x * (1.0f + erff(x * 0.7071...f));        64    8652     9.0%
	  forward.cu:120  for (int j = threadIdx.x; j < hd; j += blockDim.x) ...  18    5301     5.5%
	  forward.cu:362  sKP[r * FA_KPAD + c] = kv;                              21    3045     3.2%
	  forward.cu:215  out[idx] = gelu_erf(src[j]) * src[dff + j];             15    2014     2.1%

The other 57220 are the cuBLAS matrix products. They are in another module with
another disassembly, so the profile counts them as foreign rather than charging
them to a line here, and keeps that separate from an unresolved PC, which would
mean the disassembly no longer described the binary that ran. The top line is
`gelu_erf`, which is `erff` and not memory traffic — the same conclusion Nsight
Compute reaches for `GeGLU` from the other side, at 9.4% of peak DRAM and 10.8%
of peak SM.

Sampling needs a second run of the request, so it is off by default and changes
nothing about the run that is timed.

### As a pprof profile

`--pprof FILE` writes the same samples to `FILE` as a pprof profile, which the
usual tooling reads:

	laya-infer $HOME/laya request.json --pprof lines.pb.gz
	go tool pprof -lines -sample_index=pc_samples lines.pb.gz

Naming a file implies `--lines`, since the profile is made of the samples that
run collects; the table is printed either way. Each stack runs from the sampled
SASS instruction up through PTX to the line of `forward.cu` it was compiled
from, so one profile reads at any of the three. Beside `pc_samples` and
`inst_executed` it carries a count per stall class, which is the part a table
of shares cannot hold: `-sample_index=stall_mem` ranks the same lines by the
samples in which the warp was stalled on memory rather than by samples
outright.

In one run of the request above, `forward.cu:10` took 8511 of 95422 samples,
and the profile says 8374 of them are `stall_mem` against 23
`stall_scoreboard`. `stall_mem` merges a wait on load data with a full memory
queue; which of the two this is lives in the capture's per-reason counts and
not in the profile. The cuBLAS kernels are in the profile too, under their own
names and with SASS offsets in place of source lines, which is where the
samples `--lines` calls foreign went.

## Fidelity

Token ids, marker positions and the serialized state match `laya.c` exactly.
With `--exact`, logits agree to within 2e-4 — the difference between a tiled CPU
GEMM and cuBLAS in float32 — and replies are byte-identical except where a
probability lands on a 4th-decimal rounding boundary. The default widens that to
1e-3, the price of tf32 operands, which moves the reported probabilities in the
4th decimal and leaves the choices themselves unchanged.

Against laya.cpp itself, on the request above, the default answers agree to
within 2.1e-3 and `--exact` to within 1e-4 across every probability, score and
confidence in the reply, with the same choice named in every question either
way.

`LAYA_DEBUG=1` makes both implementations dump the state, token ids, markers and
raw logits, which is how that comparison is made.

[laya.cpp]: https://github.com/shpati/laya.cpp
