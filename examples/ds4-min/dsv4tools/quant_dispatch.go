package dsv4tools

import "fmt"

// ExpertBackend identifies a serving backend for routed experts.
type ExpertBackend string

const (
	ExpertBackendCUDA  ExpertBackend = "cuda"
	ExpertBackendMetal ExpertBackend = "metal"
	ExpertBackendCPU   ExpertBackend = "cpu"
)

// ExpertDispatch describes the activation contract selected for one layer.
// The plan is produced before a backend allocates buffers or launches a
// kernel.  It is deliberately small: weight format remains in Format, while
// activation formats describe the buffers passed to the kernels.
type ExpertDispatch struct {
	Layer       int
	Format      string
	GateUpInput string
	DownInput   string
}

// PlanExpertDispatch validates a complete expert layout and returns the
// dispatch plan for backend. CUDA supports the IQ2_XXS gate and up plus Q2_K
// down pair and the scope-A final-six-layer Q4_K splice. Metal supports those
// layouts with float32 expert activations. Other layouts are rejected.
//
// This is a host-side admission check.  A successful plan does not qualify a
// CUDA or Metal kernel; device qualification remains a separate gate.
func PlanExpertDispatch(records []ExpertLayoutRecord, backend ExpertBackend) ([]ExpertDispatch, error) {
	if err := ValidateExpertLayout(records); err != nil {
		return nil, fmt.Errorf("expert dispatch layout: %w", err)
	}
	if backend != ExpertBackendCUDA && backend != ExpertBackendMetal && backend != ExpertBackendCPU {
		return nil, fmt.Errorf("expert dispatch backend %q is unsupported", backend)
	}
	if backend == ExpertBackendCPU {
		return nil, fmt.Errorf("expert dispatch backend %q has no DSV4 expert kernel", backend)
	}
	variant, err := ClassifyExpertLayout(records)
	if err != nil {
		return nil, fmt.Errorf("expert dispatch: %w", err)
	}
	if variant != ExpertVariantIQ2Q2K && variant != ExpertVariantIQ2Q4K {
		return nil, fmt.Errorf("expert dispatch: %s backend has no scope-A dispatch for variant %d", backend, variant)
	}
	plan := make([]ExpertDispatch, len(records))
	for i, record := range records {
		if record.GateType == "Q4_K" {
			plan[i] = ExpertDispatch{Layer: record.Layer, Format: "Q4_K/Q4_K", GateUpInput: "Q8_K", DownInput: "Q8_K"}
			if backend == ExpertBackendMetal {
				plan[i].GateUpInput, plan[i].DownInput = "F32", "F32"
			}
			continue
		}
		if record.GateType != "IQ2_XXS" || record.UpType != "IQ2_XXS" || record.DownType != "Q2_K" {
			return nil, fmt.Errorf("expert layer %d: %s backend has no dispatch for %s/%s/%s", record.Layer, backend, record.GateType, record.UpType, record.DownType)
		}
		plan[i] = ExpertDispatch{
			Layer:       record.Layer,
			Format:      "IQ2_XXS/Q2_K",
			GateUpInput: "Q8_1",
			DownInput:   "Q8_1",
		}
	}
	return plan, nil
}
