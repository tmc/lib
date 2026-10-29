// Package expertparallel plans an NVFP4 Mixture-of-Experts layer for
// expert parallelism (EP) over N ranks and emits it as CUDA.
//
// A [Plan] splits the layer the way DeepEP, MoonEP and NCCL EP do. Each rank
// owns Tokens/Ranks tokens and Experts/Ranks experts:
//
//	route     top-K expert ids and weights for the rank's tokens
//	layout    count and order routes by destination rank (stable)
//	dispatch  all-to-all: each rank pulls the token rows routed to it
//	expert    grouped NVFP4 W13+SwiGLU and W2+post-norm GEMMs on received rows
//	combine   all-to-all back: each rank pulls its routes' expert rows and
//	          sums them, weighted, in increasing expert order
//	shared    optional shared expert, on a side stream, overlapping
//	          route through expert
//
// The expert and shared stages are hopper term graphs: [New] builds them
// with [hopper.Builder], fuses them with [hopper.Optimize] (an e-graph
// saturation and extraction) and lowers them with [hopper.Compile]. Route,
// layout, dispatch and combine are kernels this package emits. The dispatch
// and combine collectives are reported as [egraph.CollectiveIntent] values
// ([Plan.Collectives]) so a collective planner sees them.
//
// Because combine sums each token's routes in increasing global expert order
// with the same fused multiply-add the single-rank [hopper.ExpertSum] uses,
// the result does not depend on the rank count: EP=N reproduces EP=1.
//
// # Ranks on one GPU
//
// [Plan.Source] emits a translation unit whose host function runs every rank
// on one device. A rank is a pair of streams (main and side); rank-wide
// barriers are events; each rank's buffers live in a shared workspace and
// ranks read each other's through a device table of peer pointers, the
// model NVLink symmetric memory gives across GPUs. The dispatch and combine
// kernels are pull-based and need only those pointers, so on a multi-GPU
// system the table would hold peer-mapped addresses (package nvlshm) or the
// collectives would be replaced by NCCL EP (package ncclep, see
// [Plan.NCCLEPGroupConfig]). The same source runs on the CPU under
// cudahostemu, where streams and events are synchronous.
//
// The layout kernel is a simple single-block counter, not MoonEP's planning
// kernels; receive buffers are sized for the worst case (every route to one
// rank), so no route is dropped.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch
// releases.
package expertparallel
