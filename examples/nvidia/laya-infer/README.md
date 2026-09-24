# laya-infer

A Go + CUDA port of [laya.cpp](https://github.com/shpati/laya.cpp), the
single-file C implementation of the Laya decision agent: a ModernBERT-large
encoder (28 layers, d=1024, 16 heads) followed by a two-layer decision head, an
option-marker scorer and an action head.

Build it from the `examples` module directory with
`go build ./nvidia/laya-infer`. It answers structured questions about a state:

    laya-infer MODEL_DIR request.json
    laya-infer MODEL_DIR request.json --exact
    laya-infer MODEL_DIR request.json --lines
    laya-infer MODEL_DIR request.json --pprof FILE
    laya-infer MODEL_DIR --tokenize TEXT
    laya-infer MODEL_DIR --list-tensors
    laya-infer MODEL_DIR --serve [PORT] [--bind ADDR] [--api-key KEY]

`MODEL_DIR` is a checkout of `convaiinnovations/laya`: `model.safetensors` and
`rl_agent_config.json` beside `encoder/config.json` and
`tokenizer/tokenizer.json`. The flat layout laya.c reads, with `config.json`
and `tokenizer.json` beside the weights, loads too. A request is

    {"state": ..., "questions": {"id": {"type": "choice", "instructions": ..., "criteria": ...}}}

and each question is answered as `choice`, `score` or `noul`. The questions
may instead be a list, `[{"id": "id", "type": "choice", ...}]`, as
mlx-go-sys1 accepts; either way ids must be unique and the reply lists the
answers in request order. A choice's criteria are an object from label to
description, a list of labels, or a list of `{"label": ..., "description":
...}` objects, which render into the prompt exactly as the object form does.

`--serve` answers `POST /predict`. With `--api-key KEY`, or `LAYA_API_KEY` in
the environment, every POST must carry `Authorization: Bearer KEY` and is
refused with 401 otherwise; `GET /status` and the health checks stay open.

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
phase so the region table says where the time goes. From this directory:

	LAYA_MODEL=$HOME/laya go test -run '^$' -bench ForwardRegions -benchtime 20x

`BenchmarkForwardWhole` and `BenchmarkForwardWholeExact` time the same pass in
each precision, and `BenchmarkPredict` and `BenchmarkPredictExact` a whole
request. The benchmarks skip unless `LAYA_MODEL` names a model directory.

The default is faster than `--exact` in the matrix products: the tf32
tensor-core kernels replace the float32 GEMMs, and the fused attention never
writes the score matrix to DRAM. The elementwise kernels are the same either
way. Wall clock moves far less than GPU time, because loading the weights
costs the same whichever way the pass runs.

### Against the C reference

To compare with [laya.cpp], build it the way its README says (`cc -O3
-march=native -fopenmp laya.c -o laya -lm`) and give both the same weights and
request. Measure two things separately: a warm `--serve` process answering
`POST /predict`, and one-shot CLI runs that start a process and load the
weights to answer once. A one-shot run is mostly fixed cost, and laya.cpp
loads faster, having no CUDA context or cuBLAS to initialize, so only the
per-request time compares the two implementations.

### Nsight Systems

`newModel` opens a profiler capture range and `Close` ends it, so a timeline of
the inference itself — not of weight loading — comes out of:

	nsys profile -t cuda-sw --capture-range=cudaProfilerApi ./laya-infer $HOME/laya request.json

Without the capture range, nsys arms its CUDA injection asynchronously and a
run that reaches cuBLAS first can record nothing; the range makes the start
deterministic. `-t cuda-sw` is needed where the hardware trace drops kernels
as "incomplete CUPTI events dropped", as it can on an integrated GPU.

Nsight Compute honors the same range: `ncu --profile-from-start off` profiles
the inference kernels and skips whatever ran before them.

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

The table gives, per line of `forward.cu`, the source, the SASS instructions
compiled from it, the samples that landed on them, and their share of all
samples. Samples in the cuBLAS matrix products are in another module with
another disassembly, so they are counted as foreign rather than charged to a
line here, and kept separate from unresolved PCs, which would mean the
disassembly no longer described the binary that ran.

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
outright. `stall_mem` merges a wait on load data with a full memory queue;
which of the two it is lives in the capture's per-reason counts and not in the
profile. The cuBLAS kernels are in the profile too, under their own names and
with SASS offsets in place of source lines, which is where the samples
`--lines` calls foreign went.

## Fidelity

Token ids, marker positions and the serialized state match `laya.c` exactly.
With `--exact`, logits differ only by float32 rounding — a tiled CPU GEMM
against cuBLAS — and replies are byte-identical except where a probability
lands on a 4th-decimal rounding boundary. The default's tf32 operands move the
reported probabilities in the 4th decimal and leave the choices themselves
unchanged; `TestPrecisionFastAgreesWithExact` holds the drift to 1e-3.

`LAYA_DEBUG=1` makes both implementations dump the state, token ids, markers and
raw logits, which is how a comparison with laya.cpp is made.

[laya.cpp]: https://github.com/shpati/laya.cpp
