package cudasearch

import (
	"fmt"
	"github.com/tmc/lib/egraphcore"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamoe"
	"github.com/tmc/lib/nvidia/x/cudaprepack"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// PreparationPhase records host preparation separately from kernel receipts.
// WallNs is CPU wall elapsed time, never a device-time estimate. Complete marks
// a successful phase; unavailable phases use Unmeasured (-1) for WallNs. Failed
// attempts may retain measured time with Complete false. Byte counts describe
// logical phase input/output sizes, not
// peak memory use. Phases are reported individually, not added to device cost.
type PreparationPhase struct {
	Name                    string
	WallNs                  int64
	InputBytes, OutputBytes int64
	Complete                bool
}

type preparationReporter interface{ Preparation() []PreparationPhase }

func measurePreparation(phases *[]PreparationPhase, name string, inputBytes, outputBytes int64, run func() error) error {
	start := time.Now()
	err := run()
	*phases = append(*phases, PreparationPhase{Name: name, WallNs: time.Since(start).Nanoseconds(), InputBytes: inputBytes, OutputBytes: outputBytes, Complete: err == nil})
	return err
}

func preparationOf(x Executable) []PreparationPhase {
	if p, ok := x.(preparationReporter); ok {
		return slices.Clone(p.Preparation())
	}
	return nil
}

// NewPrefillCompilerFloat32 converts A[256,K] and B[K,N] to finite IEEE half,
// using nearest-even rounding, then binds them through NewPrefillCompiler.
// A is already grouped for one expert and row-major. If bColumnMajor is true,
// B[col*K+row] is repacked to B[row*N+col]; otherwise B is copied row-major.
// Conversion, B packing and binding are timed independently on the CPU. Input
// grouping happened before this call and remains explicitly unmeasured. Values
// overflowing finite half are rejected. This constructor performs no CUDA calls.
func NewPrefillCompilerFloat32(dev *cuda.Device, target string, k, n int, a, b []float32, bColumnMajor bool, checkLease func() bool) (*PrefillCompiler, error) {
	if dev == nil || checkLease == nil {
		return nil, fmt.Errorf("cudasearch: prefill requires device and lease checker")
	}
	if target != "sm_100a" && target != "sm_121a" {
		return nil, fmt.Errorf("cudasearch: unsupported prefill target %q", target)
	}
	if _, err := cudamoe.PrefillW256Layouts(k, n); err != nil {
		return nil, err
	}
	if n > math.MaxInt32/2 || len(a) != 256*k || len(b) != k*n {
		return nil, fmt.Errorf("cudasearch: prefill float32 extent mismatch")
	}
	start := time.Now()
	convert := func(src []float32) ([]uint16, error) {
		dst := make([]uint16, len(src))
		for i, v := range src {
			h := cudaprepack.Float32ToFloat16(v)
			if h&0x7c00 == 0x7c00 {
				return nil, fmt.Errorf("cudasearch: prefill input is nonfinite or overflows half at index %d", i)
			}
			dst[i] = h
		}
		return dst, nil
	}
	ah, err := convert(a)
	if err != nil {
		return nil, err
	}
	bh, err := convert(b)
	if err != nil {
		return nil, err
	}
	phases := []PreparationPhase{{Name: "fp32_to_fp16", WallNs: time.Since(start).Nanoseconds(), InputBytes: int64(len(a)+len(b)) * 4, OutputBytes: int64(len(ah)+len(bh)) * 2, Complete: true}}
	start = time.Now()
	packed := make([]uint16, len(bh))
	if bColumnMajor {
		for row := 0; row < k; row++ {
			for col := 0; col < n; col++ {
				packed[row*n+col] = bh[col*k+row]
			}
		}
	} else {
		copy(packed, bh)
	}
	phases = append(phases, PreparationPhase{Name: "b_row_major_pack", WallNs: time.Since(start).Nanoseconds(), InputBytes: int64(len(bh)) * 2, OutputBytes: int64(len(packed)) * 2, Complete: true})
	start = time.Now()
	c, err := NewPrefillCompiler(dev, target, k, n, ah, packed, checkLease)
	if err != nil {
		return nil, err
	}
	c.bindings.Name += fmt.Sprintf("/source-fp32/b-column-major%t", bColumnMajor)
	c.bindings.InputIdentities["source_a"] = floatDigest(a)
	c.bindings.WeightIdentities["source_b"] = floatDigest(b)
	phases = append(phases, PreparationPhase{Name: "bind_inputs", WallNs: time.Since(start).Nanoseconds(), InputBytes: int64(len(ah)+len(packed)) * 2, OutputBytes: int64(len(ah)+len(packed)) * 2, Complete: true}, PreparationPhase{Name: "external_expert_grouping", WallNs: cudatrial.Unmeasured})
	c.preparation = phases
	return c, nil
}

// Preparation returns detached host-phase observations for this compiler.
// Supplied-half inputs have unmeasured upstream conversion and packing costs.
func (c *PrefillCompiler) Preparation() []PreparationPhase { return slices.Clone(c.preparation) }

func (x *prefillExecutable) Preparation() []PreparationPhase { return slices.Clone(x.preparation) }

func clonePreparationWithMemory(scope *egraphcore.MemoryScope, in []PreparationPhase) ([]PreparationPhase, error) {
	if in == nil {
		return nil, nil
	}
	if err := bookkeepingSlots[PreparationPhase](scope, len(in)); err != nil {
		return nil, err
	}
	for _, p := range in {
		if err := scope.Reserve(int64(len(p.Name))); err != nil {
			return nil, err
		}
	}
	out := slices.Clone(in)
	for i := range out {
		out[i].Name = strings.Clone(out[i].Name)
	}
	return out, nil
}

// Built-ins expose their borrowed slice internally, so the reservation precedes
// the first copy. Opaque reporters own any allocation inside Preparation.
func executablePreparation(scope *egraphcore.MemoryScope, x Executable) ([]PreparationPhase, error) {
	var in []PreparationPhase
	switch x := x.(type) {
	case *prefillExecutable:
		in = x.preparation
	case *sequenceExecutable:
		in = x.preparation
	default:
		if p, ok := x.(preparationReporter); ok {
			in = p.Preparation()
		}
	}
	return clonePreparationWithMemory(scope, in)
}
