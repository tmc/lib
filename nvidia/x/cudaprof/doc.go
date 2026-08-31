// Package cudaprof enables deep Go to CUDA, PTX, and SASS profiling integration
// compatible with standard Go tooling (go tool pprof).
//
// # Go -> CUDA -> PTX -> SASS Source Correlation
//
// cudaprof collects GPU PC sampling data and maps device instruction addresses (SASS)
// and PTX line information directly into the standard Go profile.proto format.
//
// # pprof Labels Integration
//
// When profiling GPU workloads, cudaprof propagates active [runtime/pprof.Labels] from
// the calling Go goroutines into GPU sample tags, enabling multidimensional filtering
// and attribution by model layer, request ID, or tenant.
package cudaprof
