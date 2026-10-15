package nvrtcx

import "github.com/tmc/lib/nvidia/x/cudakernel"

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
