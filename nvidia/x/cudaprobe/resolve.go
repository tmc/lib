package cudaprobe

import (
	"context"
	"fmt"
)

// Target identifies a CUDA entry point in a specific library file. A nonempty
// SymbolVersion requests that exact ELF symbol version, not a CUDA version.
type Target struct {
	Library, Symbol, SymbolVersion string
}

// Targets lists the supported launch and synchronous allocation entry points.
// The library names are lookup hints; replace them with resolved absolute paths
// before attaching probes. Runtime and driver probes have distinct provenance.
func Targets() []Target {
	return []Target{
		{Library: "libcuda.so.1", Symbol: "cuLaunchKernel"},
		{Library: "libcuda.so.1", Symbol: "cuMemAlloc_v2"},
		{Library: "libcuda.so.1", Symbol: "cuMemFree_v2"},
		{Library: "libcudart.so", Symbol: "cudaLaunchKernel"},
		{Library: "libcudart.so", Symbol: "cudaMalloc"},
		{Library: "libcudart.so", Symbol: "cudaFree"},
	}
}

// Resolution describes a file offset, not an ELF virtual address or a process
// address. Resolver must positively identify executable function code; PLT
// stubs, indirect resolvers and ambiguous symbols are not probe targets.
type Resolution struct {
	Target           Target
	BuildID          string
	Offset, FileSize uint64
	Function, PLT    bool
}

// Resolver supplies host-generic symbol resolution. Implementations belong to
// the collector and must handle dynamic symbols, versioning and ELF translation.
// cudaprobe does not parse symbol tables itself.
type Resolver interface {
	Resolve(context.Context, Target) (Resolution, error)
}

// Resolve verifies that the collector found the requested CUDA function in the
// specified file. It refuses PLT and out-of-file offsets and missing build IDs.
// Symbol parsing and version selection remain the resolver's responsibility.
func Resolve(ctx context.Context, resolver Resolver, target Target) (Resolution, error) {
	if resolver == nil || target.Library == "" {
		return Resolution{}, fmt.Errorf("cudaprobe: resolve %s: missing resolver or library", target.Symbol)
	}
	supported := false
	for _, t := range Targets() {
		if t.Symbol == target.Symbol {
			supported = true
			break
		}
	}
	if !supported {
		return Resolution{}, fmt.Errorf("cudaprobe: resolve %s: unsupported symbol", target.Symbol)
	}
	if err := ctx.Err(); err != nil {
		return Resolution{}, fmt.Errorf("cudaprobe: resolve %s: %w", target.Symbol, err)
	}
	r, err := resolver.Resolve(ctx, target)
	if err != nil {
		return Resolution{}, fmt.Errorf("cudaprobe: resolve %s: %w", target.Symbol, err)
	}
	if r.Target != target || r.BuildID == "" || !r.Function || r.PLT || r.Offset >= r.FileSize {
		return Resolution{}, fmt.Errorf("cudaprobe: resolve %s: identity or executable file offset not established", target.Symbol)
	}
	return r, nil
}
