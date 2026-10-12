// Package cudatma describes checked tensor-map transfers and device-local
// multicast participation. EncodeTiled delegates opaque descriptor creation to
// the CUDA Driver API; it never packs vendor descriptor bytes.
//
// NewMulticastPlan emits real rank-one through rank-five tiled multicast loads
// for sm_100a, including FP16, BF16 and FP32 byte-preserving consumers. It
// admits unswizzled boxes and noninterleaved full-width 32-, 64- and 128-byte
// swizzles. Interleaved and narrow swizzled physical storage remain pending an
// authoritative byte-span/index contract and are explicitly refused. SM121a
// multicast is refused because the pinned upstream implementation excludes it.
//
// One-shot transaction bytes must fit the PTX mbarrier tx-count limit of
// 2^20-1 independently of instruction width and shared-resource admission.
//
// Source emission and host controls do not establish compiler or native
// qualification. Launches require actual device cluster/shared-resource
// admission. Tensor-map, allocation and output ownership lasts through physical
// stream completion and all selected destination reads plus the final cluster
// rendezvous. DescriptorSlot and Multicast completion are opt-in host models;
// caller-supplied completion is not evidence of device consumption.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudatma
