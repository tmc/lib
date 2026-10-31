package hopper

import "strings"

// CType returns the C storage type name used in emitted kernel signatures.
func (d DType) CType() string { return d.cType() }

// IsZero reports whether l is the zero [Lambda].
func (l Lambda) IsZero() bool { return l.root == nil }

// Chunk returns the inner K chunk divisor required by w.
func (w WeightFormat) Chunk() int { return w.chunk() }

// BufferName returns the underlying C parameter name for t.
func (t RowTensor) BufferName() string { return t.buffer() }

// Validate checks that k has valid inputs, values, and outputs.
func (k RowKernel) Validate() error {
	_, err := k.check()
	return err
}

// Params returns the C parameter declarations of k.
func (k RowKernel) Params() []string { return k.params() }

// Norm validates g and fills default schedule parameters.
func (g SkinnyGEMM) Norm() (SkinnyGEMM, error) { return g.norm() }

// Params returns the C parameter declarations of g.
func (g SkinnyGEMM) Params() []string { return g.params() }

// Norm validates g and fills default schedule parameters.
func (g TileGEMM) Norm() (TileGEMM, error) { return g.norm() }

// Params returns the C parameter declarations of g.
func (g TileGEMM) Params() []string { return g.params() }

// Norm validates a and fills default schedule parameters.
func (a PagedAttention) Norm() (PagedAttention, error) {
	normed, _, err := a.norm()
	return normed, err
}

// Params returns the C parameter declarations of a.
func (a PagedAttention) Params() []string { return a.params() }

// Norm validates a and fills default schedule parameters.
func (a MTPPagedAttention) Norm() (MTPPagedAttention, error) { return a.norm() }

// Norm validates k and fills default schedule parameters.
func (k RowTopK) Norm() (RowTopK, error) { return k.norm() }

// Params returns the C parameter declarations of k.
func (k RowTopK) Params() []string { return k.params() }

// Norm validates a and fills default schedule parameters.
func (a ExpertAlign) Norm() (ExpertAlign, error) { return a.norm() }

// Params returns the C parameter declarations of a.
func (a ExpertAlign) Params() []string { return a.params() }

// Norm validates s and fills default schedule parameters.
func (s ExpertSum) Norm() (ExpertSum, error) { return s.norm() }

// Params returns the C parameter declarations of s.
func (s ExpertSum) Params() []string { return s.params() }

// Norm validates c and fills hardware roofline defaults.
func (c CostConfig) Norm() (CostConfig, error) { return c.norm() }

// IsHopper reports whether c targets an sm_90a Hopper device.
func (c CostConfig) IsHopper() bool { return c.isHopper() }

// IsBlackwell reports whether c targets an sm_100a/sm_120a/sm_121a Blackwell device.
func (c CostConfig) IsBlackwell() bool { return c.isBlackwell() }

// EstimateStepNs returns the analytical latency estimate in nanoseconds for k under cfg.
func (k PlanKernel) EstimateStepNs(cfg CostConfig) float64 { return k.estimateStepNs(cfg) }

// PTXASSharedBytes returns the static shared-memory byte count recorded from ptxas.
func (l Launch) PTXASSharedBytes() int { return l.ptxasShared }

// SetPTXASSharedBytes sets the static shared-memory byte count recorded from ptxas.
func (l *Launch) SetPTXASSharedBytes(n int) { l.ptxasShared = n }

// DeduplicateCUDAHelpers strips already-emitted shared WGMMA / FP4 device
// helper blocks from s, recording newly seen helper blocks in seen.
func DeduplicateCUDAHelpers(s string, seen map[string]bool) string {
	for _, key := range []struct {
		name  string
		block string
	}{
		{"nvfp4WGMMA", nvfp4WGMMAHelpers},
		{"wgmmaSwizzle", wgmmaSwizzleHelpers},
		{"blackwellFP4", blackwellFP4Helpers},
	} {
		if strings.Contains(s, key.block) {
			if seen[key.name] {
				s = strings.Replace(s, key.block, "", 1)
			}
			seen[key.name] = true
		}
	}
	return s
}
