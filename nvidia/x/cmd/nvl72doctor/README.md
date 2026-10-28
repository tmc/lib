# nvl72doctor

Check that one compute tray is ready for a GB300 or GB200 NVL72 rack job:

```
nvl72doctor                         # gb300-nvl72 profile, text report
nvl72doctor -json -save-facts tray.json
nvl72doctor -facts tray.json        # re-evaluate a capture without hardware
```

| Check | Source | Rack node passes when |
| --- | --- | --- |
| `driver` | NVML | driver CUDA version ≥ 12.9 (gb300) or 12.8 (gb200) |
| `toolkit` | `nvcc --version` | toolkit ≥ the same version |
| `gpus` | NVML | 4 GPUs, all sm_103 (gb300) or sm_100 (gb200) |
| `nvlink gpuN` | NVML NvLink state/version/remote type | 18 active links, version code 7 (NVLink 5.0), each to an NVSwitch |
| `fabric gpuN` | `nvmlGpuFabricInfo_v3` | probe completed with success; zero health mask (else warn) |
| `mnnvl clique` | same | every local GPU has the same nonzero cluster UUID and clique ID |
| `imex channels` | `/dev/nvidia-caps-imex-channels` | at least one channel node |
| `imex domain` | `nvidia-imex-ctl -N -j` | domain UP, every node READY, every connection CONNECTED |
| `fabric manager` | libnvfm at 127.0.0.1:6666 | no NVLink training failures. If unreachable, the check reports info, because NVL72 compute trays do not run FM |
| `nccl all_reduce`, `nccl alltoall` | x/nccltest in a child process | all local GPUs, every size matches the host oracle |
| `nccl cross-tray` | — | always skip: needs a multi-process launcher |
| `compile K ARCH` | nvcc + ptxas -v | compiles, no spills, shared memory within the per-block limit |

On a node whose GPUs are not the profile's architecture, such as a GB10,
the rack-only checks report `skip` with the reason. The exit status is 1
only if a check fails. The NCCL sweep uses the GPUs, so run it under the
site GPU lock. `-nccl=false` and `-compile=false` drop those parts.

testdata/script/gb10.txt evaluates a capture from a GB10. The GB300 expectations are
tested only against constructed facts.
