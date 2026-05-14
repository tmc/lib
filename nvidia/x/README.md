# nvidia/x

Hand-written packages, as opposed to the generated bindings in the sibling
directories (`driver`, `nccl`, `nvml`, ...). Code here is authored and
maintained by hand and builds on those bindings.

- `nvlshm` — cross-process NVLink shared GPU buffers (CUDA VMM + NVSwitch
  multicast), with `tcpboot` as a test bootstrap.
- `moonep` — Go port of MoonEP's host layer (expert-parallel MoE buffer
  layout and lifecycle) on top of `nvlshm`.
