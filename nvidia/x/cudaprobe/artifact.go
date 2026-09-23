package cudaprobe

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// Binding is a collector's explicit registration of a function to a loaded
// artifact. First and Last bound the inclusive process observation sequence
// interval during which the registration held. ModuleGeneration identifies
// the loaded module lifetime; Evidence identifies the registration receipt.
// A symbol-name match or disassembly alone cannot establish a Binding.
type Binding struct {
	Process                                               Process
	AddressSpace, Function, ModuleGeneration, First, Last uint64
	Evidence                                              string
	Artifact                                              *cudakernel.Artifact
}

// KernelIdentity is an artifact join supported by a caller-supplied binding.
// It identifies code; it does not establish successful GPU execution.
type KernelIdentity struct {
	Digest, EntryPoint, Target, Evidence string
	ModuleGeneration                     uint64
}

// Identify joins a decoded launch to exactly one explicit function registration.
// It validates the artifact digest and ABI descriptor and refuses ambiguous,
// stale or absent bindings. It does not infer registrations from kernel names.
func Identify(e Event, bindings []Binding) (KernelIdentity, error) {
	if !e.Decoded || e.Launch == nil || e.Sequence == 0 || e.Identity.Process.PID == 0 || e.Identity.Process.Generation == 0 || e.Identity.AddressSpace == 0 {
		return KernelIdentity{}, fmt.Errorf("cudaprobe: launch identity unavailable")
	}
	var selected *Binding
	for i := range bindings {
		b := &bindings[i]
		if b.Process != e.Identity.Process || b.AddressSpace != e.Identity.AddressSpace || b.Function != e.Launch.Function || e.Sequence < b.First || e.Sequence > b.Last {
			continue
		}
		if b.First == 0 || b.ModuleGeneration == 0 || b.Evidence == "" {
			return KernelIdentity{}, fmt.Errorf("cudaprobe: incomplete function binding")
		}
		if selected != nil {
			return KernelIdentity{}, fmt.Errorf("cudaprobe: ambiguous function binding")
		}
		selected = b
	}
	if selected == nil {
		return KernelIdentity{}, fmt.Errorf("cudaprobe: function binding unavailable")
	}
	if err := selected.Artifact.Validate(); err != nil {
		return KernelIdentity{}, fmt.Errorf("cudaprobe: bound artifact: %w", err)
	}
	a := selected.Artifact
	return KernelIdentity{Digest: a.Digest, EntryPoint: a.EntryPoint, Target: a.Target, Evidence: selected.Evidence, ModuleGeneration: selected.ModuleGeneration}, nil
}
