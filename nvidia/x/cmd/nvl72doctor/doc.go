// Command nvl72doctor checks that one compute node is ready to join a GB300
// or GB200 NVL72 rack job, and reports what it could not check.
//
// Usage:
//
//	nvl72doctor [-profile gb300-nvl72] [-json] [-save-facts file] [-nccl=false] [-compile=false]
//	nvl72doctor -facts file
//
// It reads, through NVML, the driver and CUDA versions, each GPU's compute
// capability, NVLink port state, version and remote device type, and the
// GPU fabric probe (nvmlGpuFabricInfo_v3: MNNVL cluster UUID, clique ID,
// state, health). It lists the IMEX channel nodes, parses nvidia-imex-ctl
// -N -j, and asks a local Fabric Manager, if one answers, for partitions
// and NVLink training failures. It then runs an NCCL all-reduce and
// all-to-all sweep over the local GPUs in a child process, each size
// checked against a host oracle (package nccltest), and compiles the
// e-graph tile GEMMs, the tcgen05 block-scaled GEMM and the NVL72 multimem
// collectives with nvcc for the node's architecture and the profile's,
// reading registers, spills and shared memory from ptxas -v.
//
// A node whose GPUs are not the profile's architecture, such as a GB10
// (sm_121) development box, is not a rack node: checks only a rack node
// can pass are reported as skip, with the reason, and the command exits 0
// unless a check that applies fails. The NCCL sweep runs one communicator
// in one process, so it covers the GPUs of one tray; a cross-tray
// collective needs one process per GPU under a multi-node launcher and is
// always reported as skipped.
//
// -save-facts writes what was observed as JSON; -facts evaluates such a
// file without touching the node, so a capture from a rack can be
// reviewed elsewhere. The NCCL sweep uses the GPUs: run it under the
// site's GPU lock.
package main
