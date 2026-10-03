package cudaprof

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// BindEvidence retains a validated artifact, the existing workload key and
// its producer-supplied compiler cache key. It never substitutes an image hash
// for the original compiler input key. The caller must retain the receipt
// associating compilerKey with this artifact; this function cannot recreate
// or authenticate a compiler receipt from the final image alone.
func BindEvidence(a *cudakernel.Artifact, w cudacache.WorkloadIdentity, compilerKey string) (cudaevidence.Identity, error) {
	if a == nil {
		return cudaevidence.Identity{}, fmt.Errorf("bind evidence: nil artifact")
	}
	if a.Kind < cudakernel.KindPTX || a.Kind > cudakernel.KindTileIR {
		return cudaevidence.Identity{}, fmt.Errorf("bind evidence: unknown artifact kind")
	}
	if err := a.Validate(); err != nil {
		return cudaevidence.Identity{}, fmt.Errorf("bind evidence: %w", err)
	}
	if strings.TrimSpace(w.ProgramID) == "" || strings.TrimSpace(w.NumericalMode) == "" {
		return cudaevidence.Identity{}, fmt.Errorf("bind evidence: incomplete workload")
	}
	for _, dim := range w.ShapeDims {
		if dim < 0 {
			return cudaevidence.Identity{}, fmt.Errorf("bind evidence: negative workload extent")
		}
	}
	id := cudaevidence.Identity{ArtifactDigest: a.Digest, Entry: a.EntryPoint, Target: a.Target, CompilerKey: compilerKey, WorkloadKey: w.Key()}
	if err := id.Validate(); err != nil {
		return cudaevidence.Identity{}, fmt.Errorf("bind evidence: %w", err)
	}
	return id, nil
}
