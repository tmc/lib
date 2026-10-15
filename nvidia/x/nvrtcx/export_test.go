package nvrtcx

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func ExportStoreCache(key string, cm *CompiledModule) error { return storeCache(key, cm) }
func ExportLookupCache(key string) (*CompiledModule, bool)  { return lookupCache(key) }

func ExportJITRequest(opts ...Option) (cudakernel.JITOptions, string, error) {
	var cfg options
	for _, option := range opts {
		if option == nil {
			return cudakernel.JITOptions{}, "", fmt.Errorf("nil test option")
		}
		option(&cfg)
	}
	_, _, extracted, _ := SeparateFlags(cfg.flags)
	jit := resolveJITOptions(cfg.jitOptions, extracted)
	return jit, jitCacheKey("input", jit), jit.Validate()
}

// ExportResolveExternalDependencies exports resolveExternalDependencies for testing.
func ExportResolveExternalDependencies(src string, flags []string, inMemoryHeaders map[string]string) ([]cudakernel.Dependency, bool) {
	return resolveExternalDependencies(src, flags, inMemoryHeaders)
}

// ExportComputeDependencyKey exports computeDependencyKey for testing.
func ExportComputeDependencyKey(deps []cudakernel.Dependency) string {
	return computeDependencyKey(deps)
}

// ExportNewCompiledModule creates a mock CompiledModule for adapter testing.
func ExportNewCompiledModule(kind OutputKind, ptxText string, bytesData []byte) *CompiledModule {
	cm := &CompiledModule{output: kind, compilerOptionsKnown: true}
	switch kind {
	case OutputPTX:
		cm.ptx = ptxText
	case OutputCUBIN:
		cm.cubin = bytesData
	case OutputLTOIR:
		cm.ltoir = bytesData
	case OutputTileIR:
		cm.tileir = bytesData
	}
	return cm
}
