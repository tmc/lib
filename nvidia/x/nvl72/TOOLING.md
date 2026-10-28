# GB300 NVL72 bring-up tooling

This is the inventory of tools in this module for bringing up a GB300
(sm_103a) or GB200 (sm_100a) NVL72 rack, made on 2026-10-28 at
work/sota-integrate 6dc15b32f. The only GPU available was a GB10 (sm_121a,
driver 580.173.02 / CUDA 13.0, toolkit 13.4, no NVLink ports, no IMEX).

Every package below passes `go vet` for darwin/arm64, linux/arm64 and
linux/amd64, and its tests pass on darwin. One exception: the
`TestProbe/healthy` case in x/gpuhealth timed out once under heavy machine
load and passed when rerun alone.

Evidence labels: EXECUTED-GB10 (ran on the dgx GB10), COMPILED (nvcc/ptxas
for the named arch, no GPU), EMULATED (cudahostemu or a CPU reference),
EMITTED (source generated only), MODELLED (cost model only).

## Readiness command

`x/cmd/nvl72doctor` checks one compute tray and reports what it skipped:

    nvl72doctor [-profile gb300-nvl72|gb200-nvl72] [-json] [-save-facts f.json]
    nvl72doctor -facts f.json    # re-evaluate a capture, no hardware

It checks:

- the driver and CUDA versions (NVML) and nvcc (≥ 12.9 for sm_103a);
- the GPU count and compute capability (4 × sm_103 per tray);
- each GPU's NVLink ports: active count (18), version code 7 (NVLink 5.0)
  and remote device type (NVSwitch);
- the GPU fabric probe (`nvmlGpuFabricInfo_v3`): completed, success, health
  mask;
- MNNVL cluster UUID and clique agreement across the tray's GPUs;
- the `/dev/nvidia-caps-imex-channels` nodes, and the IMEX domain from
  `nvidia-imex-ctl -N -j` (package imex);
- a local Fabric Manager's partitions and NVLink training failures, if one
  answers;
- an NCCL all-reduce and all-to-all sweep (1 KiB to 256 MiB) over the local
  GPUs, run in a child process, with every size checked against a host
  oracle (package nccltest);
- an nvcc + ptxas -v compile suite for the node's arch and the profile's
  arch. The suite covers the e-graph BF16 tile GEMM, the grouped NVFP4 MoE
  W13 GEMM, the sm_12x FP4 mma.sync GEMM, the tcgen05 scale_vec::4X NVFP4
  GEMM, and the 72-rank multimem all-reduce+RMSNorm and MoE combine. Each
  compile is gated on errors, spills, and static plus dynamic shared memory
  against the per-block limit (227 KiB on sm_10x, 99 KiB on sm_12x).

On a node that is not the profile's architecture, checks that only a rack
node can pass are reported as `skip` with the reason, and the command exits
0. testdata/script/gb10.txt evaluates a capture from the GB10
(EXECUTED-GB10, 2026-10-28). The GB300 behaviour has been checked only
against constructed facts (main_test.go), not against a real tray.

## Inventory

| Need | Package | Status | Gap |
| --- | --- | --- | --- |
| Device discovery, compute capability | nvml, x/cudadevicetable | works (EXECUTED-GB10) | cudadevicetable has no measured sm_100/sm_103 rows |
| NVLink 5 ports, version, remote type | nvml (`NvLinkState`, `NvLinkVersion`, `NvLinkRemoteDeviceType`), x/cmd/nvl72doctor | builds; on GB10, NVML returns Not Supported | untested on Blackwell. NVML documents the DL CRC, recovery and replay counters as unsupported on Blackwell+, so link error rates need the PRM counters, which nothing reads yet |
| NVSwitch / fabric state | nvml `FabricInfo`, x/nvtelemetry `NVMLFabricReader`, fabricmanager | builds; FM library absent on dgx | untested on Blackwell. On NVL72 the switch trays manage the fabric (NMX), and no binding reads the NMX API |
| MNNVL cluster / clique | nvml `GpuFabricInfoV`, x/cmd/nvl72doctor | builds | rack-wide membership needs every tray's capture merged. No aggregator exists: use `-save-facts` per tray |
| IMEX channels, domain | x/imex (reads `nvidia-imex-ctl -N -j`), x/cmd/nvl72doctor | parser tested on DCGM's spec fixture | no real capture yet. Channel *allocation* (beyond channel0) and the IMEX daemon config are not modelled |
| Fabric Manager | fabricmanager (libnvfm bindings) | builds, layout tests | never connected to a live FM |
| NCCL bring-up, bandwidth sweeps | nccl, x/nccltest (+ cmd/nccltest, cmd/ncclcompare) | works single-process (EXECUTED-GB10, 1 rank, 1 KiB to 512 KiB all-reduce checked against the oracle) | single-process only: no multi-process launcher, so no cross-tray (MNNVL) collective. No NVLS/multimem algorithm selection check |
| NVSHMEM | nvshmem, x/nvlshm | builds, unit tests | no bandwidth test. nvlshm VMM/multicast never ran across GPUs |
| Multi-node launch / rendezvous | x/nccljoin (aligns NCCL Inspector records), x/cudacollective (membership plans) | parsers tested | **missing**: nothing launches ranks or distributes an NCCL unique ID across trays. nccljoin is post-hoc analysis, not rendezvous |
| NCCL post-mortem | x/ncclinspector, x/ncclras | fixture-tested | no capture from a 72-rank job |
| Cross-rank profiling, timeline merge | x/nsightsystems (SQLite import), cupti, x/cuptiactivity → x/perfetto, x/nvtx, x/cudacapture `MergeRanks` | works per process | `MergeRanks` produces an additive aggregate, not a clock-aligned timeline. **Missing**: a per-rank clock-offset merge into one Perfetto trace |
| Per-kernel profiling | x/nsightcompute (CSV 2026.3), x/cuptipcsampling, x/cuptisass, cuptipmsampling | works on sm_121 | untested on sm_100/sm_103. cuptisass marks sm_100f `ErrHardwarePending` |
| SASS analysis for tcgen05 | x/cudasass (`SummarizeSM100Function`, `AuditSM100Function`), x/sassequiv | parsers tested | SASS for sm_103a comes only from compile-only builds (COMPILED) |
| Health, Xid, DCGM | x/gpuhealth, x/xidlog + x/xidcatalog (GB200 rows), dcgm, x/dcgmjson, x/nvtelemetry, x/cmd/cudadoctor `-health` | works on GB10 | xidcatalog has GB200 applicability but not GB300. gpuhealth expects exactly one GPU when no UUID is given |
| Compile-only arch checks | x/ptxasinfo, x/cmd/cudacompilecheck, x/cudacensus, x/sassequiv, x/cmd/nvl72doctor suite | works (COMPILED sm_100a/sm_103a/sm_121a on dgx) | cudaroofline and egraph `CompileLayer` know no sm_100/sm_103 device ("unknown architecture NVIDIA GB300") |
| Host emulation | x/cudahostemu (tcgen05, TMEM, CLC, multi-rank) | works (EMULATED) | the semantics are a model, not hardware |
| Paired timing statistics | x/pairedtiming, x/cmd/pairedtiming | works | arch-independent |
| Target profiles for absent hardware | x/cudadevicetable `Profile` (work/nvl72-arch) | datasheet figures marked Unverified; GB10 probed | sm_103a figures are from documents, not a probe |
| Launch-overhead projection | x/cmd/nsysgaps, x/cudaprof/advise `ProjectTo` (work/nvl72-launch, work/nvl72-advise) | works on GB10 nsys captures | MODELLED: assumes GB10 launch and drain latencies carry over |
| Multi-rank code on one GPU | x/rankgroup (work/nvl72-tp), egraph/expertparallel `Plan.Source` | EXECUTED-GB10 (TP=1,2,4 ranks sharing one GPU) | ranks share one GPU; NCCL backend needs one GPU per rank |

## Commands from the other NVL72 lanes

These are on sibling branches, not on work/nvl72-tooling. Branch and
commit are given so a reader can check them with `git show`.

- `x/cmd/nsysgaps` (work/nvl72-launch, f0925936c, c7d6e8809): reads the
  `nsys stats --report cuda_gpu_trace` CSV of a decode run, splits it into
  steps ending at `-end` (default Argmax) and reports launches, busy and
  idle time per step, the kernels with the most idle time before them, and
  kernels shorter than `-short` (latency-bound). `-floor 2ms` models the
  step on a device that runs its bandwidth-bound work in 2 ms, which is
  what GB300 does for Llama-3-8B decode. MODELLED.

      nsys stats --report cuda_gpu_trace --format csv run.nsys-rep >gpu.csv
      nsysgaps -floor 2ms -floor 1ms gpu.csv

- `llama-infer -egraph-pdl` (work/nvl72-launch, 77ebc701b): launches each
  generated kernel as the programmatic dependent of the previous one.
  EXECUTED-GB10 with identical tokens.
- `llama-infer -egraph-emit DIR -egraph-target T` (work/nvl72-arch,
  37e9eeca4): writes the kernels llama-infer would compile, planned for
  `-model`'s config.json and target T (sm_103a, GB300, H200, ...), and a
  list of their name expressions, without a GPU. Then
  `x/cmd/cudacompilecheck` reports registers, spills and shared memory
  for T. EMITTED, then COMPILED with nvcc 13.4 on dgx:

      llama-infer -model $M -egraph-emit out -egraph-target GB300
      cudacompilecheck -arch sm_103a -no-spill -sass \
          -keep $(paste -sd, out/names.txt) out/llama_egraph.cu

  -egraph-emit writes out/llama_egraph.cu and out/names.txt. Run
  cudacompilecheck on a host with nvcc 12.9 or later, or give it a remote
  toolkit with -exec (see its package doc).
- `x/cmd/cudaprof advise` (work/nvl72-advise): `-target sm_103a` projects
  one capture's step onto a target, several captures are analyzed as the
  ranks of one job, and a `.sass`/`.ptxas.txt`/`.cubin` from a
  compile-only build is analyzed without a capture. MODELLED for the
  projection, COMPILED for the compile-only findings.
- `llama-infer -egraph -tp N` (work/nvl72-tp, 5a42cda31) with
  `-tp-mode serial|streams|green` and `-tp-backend device|nccl`: runs N
  tensor-parallel ranks; on one GB10, all ranks share the GPU.
  EXECUTED-GB10: TP=1, 2, 4 match the single-GPU tokens.
- `x/cudahostemu` tcgen05 MMA and TMEM (work/nvl72-tcgen05, 2d00e0c58):
  runs tcgen05 GEMM source on the CPU and checks descriptors, TMEM
  bounds and commit ordering. EMULATED.
- `x/egraph/expertparallel` `TestSearchScale` and `BenchmarkSearchScale`
  (work/nvl72-ep, 86953f039): e-graph search time on DeepSeek-V3-shaped
  graphs. `EGRAPH_SEARCH_SCALE=1` runs the full curve. Whole-graph
  ExtractDAG grows superlinearly (about 300 s at 61 layers on an M4 Max);
  per-layer search picks the same launches in about 0.55 s.

## Findings from the compile suite (COMPILED, nvcc 13.4 on dgx)

- `nvl72.AllReduceResidualRMSNorm` with `DType: BF16` fails ptxas on
  sm_90a, sm_103a and sm_121a ("Arguments mismatch for instruction
  'multimem.ld_reduce'"). collective.go emits `.add.v2.bf16` with one
  32-bit register. `.add.bf16x2` compiles. F32 compiles. Fixed on
  work/nvl72-tp in a64a53754, which is not merged here. Until it is, this
  branch's doctor reports the compile as a failure.
- `hopper.TileGEMM` at its default 128×128×3 stages needs 110592 bytes of
  dynamic shared memory, which does not fit sm_121's 99 KiB. The suite
  drops to 2 stages there, as the llama-infer plans do.
- `hopper.CompileLayer(...).CUDASource` is not a standalone translation
  unit: on a small dense config, nvcc rejects a zero-sized array (M=1
  megakernel) and a duplicate `eg_wgmma_desc_128b` (M=64). It is emitted
  but never compile-gated (EMITTED only).

## First GB300 NVL72 session checklist

None of the following can be verified without the hardware.

1. On every tray, run `nvl72doctor -save-facts tray-$HOST.json` under the
   site GPU lock and keep the files. Expect all checks to pass except
   `fabric manager` (info) and `nccl cross-tray` (skip).
2. Compare the 18 trays' `cluster_uuid` and `clique_id` (all must be
   equal) and the IMEX domain node count (18, all READY).
3. Check that NVML reports NVLink version code 7 and remote type switch on
   all 18 ports of each GPU. If the driver reports a different code, the
   profile's `LinkVersion` is wrong, not the rack.
4. Check `GpuFabricInfoV` `HealthMask` bit meanings against the installed
   nvml.h and decode them in the doctor.
5. Run a 72-rank NCCL all-reduce and all-to-all (nccl-tests under srun or
   mpirun, one process per GPU) and compare against nccltest's 4-rank
   numbers. No in-repo launcher exists.
6. Check `NCCL_NVLS_ENABLE` / NVLS selection and multimem availability
   (`cuMulticastGetGranularity`) before relying on nvl72 multimem kernels.
7. Run the tcgen05 and nvl72 native gates with
   `CUDA_NVL72_NATIVE_ARCH=sm_103a` on one GPU, and compare against
   cudahostemu.
8. Run CUPTI PC sampling, SASS metrics and Nsight Compute on sm_103.
   Record which metrics and stall reasons exist. They are untested on
   Blackwell.
9. Read the NVLink error counters that Blackwell supports (PRM counters),
   since the DL CRC, recovery and replay fields are unsupported.
10. Capture a real `nvidia-imex-ctl -N -j` and `dcgmi diag -j` output for
    the imex and dcgmjson fixtures, and record the Xid IDs that GB300
    reports.
