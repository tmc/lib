package cudamegacpu

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudamega"
)

// MoEResult contains FP32 task ranges described by Program.Storage(I).
// Scratch and StartOrder belong to the caller after all worker threads join.
type MoEResult struct {
	Scratch    []float32
	StartOrder []uint64
	Abort      uint32
	// Threads counts all locked OS worker threads, including every thread of
	// every group. It does not count Go runtime or test-harness threads.
	Threads uint64
}

// RunMoE executes a sealed typed program over an immutable input binding.
// Shape, layer, finite-value and slab bounds are checked by BindMoE; this
// method snapshots that binding and checks every program configuration before
// allocating output or starting any thread. Scalar Run refuses typed programs.
//
// Epoch must be nonzero. Abort is a pre-launch host request, not an asynchronous
// cancellation handle. Dependency polling is iteration-bounded, not a wall
// deadline. This CPU execution does not qualify CUDA source or hardware.
func (e *Executor) RunMoE(p *cudamega.Program, binding cudamega.MoEBinding, epoch uint64, abort uint32) (MoEResult, error) {
	if p == nil {
		return MoEResult{}, fmt.Errorf("cudamegacpu: nil program")
	}
	input, err := binding.Snapshot()
	if err != nil {
		return MoEResult{}, fmt.Errorf("cudamegacpu: binding: %w", err)
	}
	steps := p.Steps()
	if len(steps) == 0 {
		return MoEResult{}, fmt.Errorf("cudamegacpu: empty typed program")
	}
	for _, s := range steps {
		d, ok := s.Operation.MoE()
		if !ok || d.Config != input.Config {
			return MoEResult{}, fmt.Errorf("cudamegacpu: task %q differs from input configuration", s.Task.ID)
		}
	}
	r, err := e.run(p, nil, &input, input.Config.Intermediate, epoch, abort)
	if err != nil {
		return MoEResult{}, err
	}
	out := make([]float32, len(r.Scratch))
	for i, v := range r.Scratch {
		out[i] = math.Float32frombits(v)
	}
	return MoEResult{Scratch: out, StartOrder: r.StartOrder, Abort: r.Abort, Threads: uint64(e.admission.ResidentCTAs) * uint64(e.admission.WorkerThreads)}, nil
}

// Each product and addition is explicitly converted to float32 so the Go
// compiler cannot contract the ordered dot product into a fused multiply-add.
func (w *worker) computeMoE(i int, sources []int, d cudamega.MoEOperation) error {
	input := w.r.moe
	if input == nil || input.Config != d.Config {
		return fmt.Errorf("unbound typed MoE operation")
	}
	s := w.r.layout[i]
	o := w.r.steps[i].Owner
	stride := int(o.Count()) * int(w.r.threads)
	for n := int(w.cta-o.Lo)*int(w.r.threads) + int(w.tid); n < s.Elements; n += stride {
		var result float32
		switch d.Kind {
		case cudamega.MoEGateUp:
			row := (input.Layer*d.Config.Experts+d.Expert)*2*d.Config.Intermediate + (n/8)*16 + n%8
			var gate, up float32
			for h, x := range input.Hidden {
				gate = float32(gate + float32(decodeMoE(input.Weights[row*d.Config.Hidden+h])*x))
				up = float32(up + float32(decodeMoE(input.Weights[(row+8)*d.Config.Hidden+h])*x))
			}
			gate = float32(gate * input.Scales[row])
			up = float32(up * input.Scales[row+8])
			exponential := float32(math.Exp(float64(-gate)))
			result = float32(float32(gate/float32(1+exponential)) * up)
		case cudamega.MoECombine:
			for _, dep := range sources {
				expert, ok := w.r.steps[dep].Operation.MoE()
				if !ok || expert.Kind != cudamega.MoEGateUp {
					return fmt.Errorf("unbound expert operand")
				}
				v := math.Float32frombits(w.r.scratch[w.r.layout[dep].Offset+n])
				result = float32(result + float32(input.Routing[expert.Expert]*v))
			}
		default:
			return fmt.Errorf("unsupported typed MoE operation")
		}
		w.r.scratch[s.Offset+n] = math.Float32bits(result)
	}
	return nil
}

// decodeMoE independently implements the E4M3FN storage value used by the
// portable executor. BindMoE's decoder is only a preflight overflow bound.
func decodeMoE(raw byte) float32 {
	magnitude := int(raw & 127)
	exponent, fraction := magnitude>>3, magnitude&7
	v := float32(math.Ldexp(float64(fraction), -9))
	if exponent != 0 {
		v = float32(math.Ldexp(1+float64(fraction)/8, exponent-7))
	}
	if raw&128 != 0 {
		v = -v
	}
	return v
}
