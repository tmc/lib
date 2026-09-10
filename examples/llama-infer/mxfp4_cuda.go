//go:build !darwin && !tpu

package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/mxfp4"
)

// mxfp4Enabled reports whether the single-stream decode should run its linear
// projections through MXFP4 weights. Off unless LLAMA_INFER_MXFP4=1.
func mxfp4Enabled() bool { return os.Getenv("LLAMA_INFER_MXFP4") == "1" }

// mxfp4GroupSize is the OCP Microscaling block size. The package's scale
// encoding is not the OCP one (its doc says so); the block size is.
const mxfp4GroupSize = 32

// mxfp4Weights holds MXFP4 copies of every linear projection in the model,
// alongside the full-precision copies the rest of the engine keeps using.
//
// UNQUALIFIED. Nothing here has run on a device. What is covered: the host
// quantizer has its own round-trip tests in x/mxfp4, and the upload arithmetic
// is checked at load. What is not: that MXFP4LinearGEMV produces the same
// logits as LinearGEMV within any stated tolerance for any model. The x/mxfp4
// doc says adoption requires a matched model-workload campaign against the
// production precision path; that campaign has not been run, and until it
// has, a token produced under this mode is not evidence of anything.
//
// This is a kernel-path adoption, not a memory saving. The f32 or half copy
// of every matrix stays resident because the prefill GEMMs run on cublasLt
// and the fused kernels read the full-precision layout; only the seven
// per-layer projections and the LM head of the single-stream decode step
// read the MXFP4 copy. A run in this mode holds more weight memory, not less.
//
// Bias is added by a separate BatchAddBias launch with one row, because the
// MXFP4 kernel takes none. Those launches count; the MXFP4 launches do not,
// because x/mxfp4 owns its launcher, so a launch figure reported under this
// mode is low by one per projection.
type mxfp4Weights struct {
	engine *mxfp4.Engine
	arena  *cudamem.Arena
	layers []mxfp4Layer
	lmHead mxfp4Matrix
}

// mxfp4Layer is one decoder layer's projections.
type mxfp4Layer struct {
	WQ, WK, WV, WO, WGate, WUp, WDown mxfp4Matrix
}

// mxfp4Matrix is one [outDim, inDim] weight matrix on the device.
type mxfp4Matrix struct {
	packed cudamem.Slice[uint8]
	scales cudamem.Slice[float32]
	inDim  int
	outDim int
}

func (l *mxfp4Layer) q() *mxfp4Matrix {
	if l == nil {
		return nil
	}
	return &l.WQ
}

func (l *mxfp4Layer) k() *mxfp4Matrix {
	if l == nil {
		return nil
	}
	return &l.WK
}

func (l *mxfp4Layer) v() *mxfp4Matrix {
	if l == nil {
		return nil
	}
	return &l.WV
}

func (l *mxfp4Layer) o() *mxfp4Matrix {
	if l == nil {
		return nil
	}
	return &l.WO
}

func (l *mxfp4Layer) gate() *mxfp4Matrix {
	if l == nil {
		return nil
	}
	return &l.WGate
}

func (l *mxfp4Layer) up() *mxfp4Matrix {
	if l == nil {
		return nil
	}
	return &l.WUp
}

func (l *mxfp4Layer) down() *mxfp4Matrix {
	if l == nil {
		return nil
	}
	return &l.WDown
}

// head returns the LM head matrix, or nil when the model carries no MXFP4
// copies.
func (m *mxfp4Weights) head() *mxfp4Matrix {
	if m == nil {
		return nil
	}
	return &m.lmHead
}

// layer returns layer i's projections, or nil when the model carries none.
func (m *mxfp4Weights) layer(i int) *mxfp4Layer {
	if m == nil || i < 0 || i >= len(m.layers) {
		return nil
	}
	return &m.layers[i]
}

// uploadMXFP4 quantizes every linear projection of w on the host and uploads
// the packed weights and scales. Each matrix's input dimension must be a
// multiple of the group size, which is checked before anything is allocated.
func uploadMXFP4(dev *cuda.Device, tracker *cudamem.Tracker, w *ModelWeights, cfg *ModelConfig) (_ *mxfp4Weights, resultErr error) {
	if dev == nil || w == nil || cfg == nil {
		return nil, errors.New("mxfp4 upload: nil argument")
	}
	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	qDim := cfg.NumAttentionHeads * cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * cfg.HeadDim()
	for _, d := range []struct {
		name string
		dim  int
	}{{"hidden", hidden}, {"intermediate", inter}, {"query", qDim}} {
		if d.dim <= 0 || d.dim%mxfp4GroupSize != 0 {
			return nil, fmt.Errorf("mxfp4 upload: %s dimension %d is not a multiple of the group size %d", d.name, d.dim, mxfp4GroupSize)
		}
	}
	if len(w.Layers) != cfg.NumHiddenLayers {
		return nil, fmt.Errorf("mxfp4 upload: %d layers of weights for %d configured", len(w.Layers), cfg.NumHiddenLayers)
	}

	engine, err := mxfp4.NewEngine(dev)
	if err != nil {
		return nil, fmt.Errorf("mxfp4 upload: %w", err)
	}
	arena := cudamem.NewTrackedArena(dev, tracker)
	m := &mxfp4Weights{engine: engine, arena: arena}
	defer func() {
		if resultErr != nil {
			_ = m.close()
		}
	}()

	upload := func(name string, host []float32, outDim, inDim int) (mxfp4Matrix, error) {
		q, err := mxfp4.QuantizeMatrix(outDim, inDim, host, mxfp4GroupSize)
		if err != nil {
			return mxfp4Matrix{}, fmt.Errorf("%s: %w", name, err)
		}
		packed := cudamem.AllocIn[uint8](arena, "mxfp4-packed", len(q.Packed))
		scales := cudamem.AllocIn[float32](arena, "mxfp4-scales", len(q.Scales))
		if err := arena.Err(); err != nil {
			return mxfp4Matrix{}, fmt.Errorf("%s: %w", name, err)
		}
		if err := packed.CopyFromHost(q.Packed); err != nil {
			return mxfp4Matrix{}, fmt.Errorf("%s: %w", name, err)
		}
		if err := scales.CopyFromHost(q.Scales); err != nil {
			return mxfp4Matrix{}, fmt.Errorf("%s: %w", name, err)
		}
		return mxfp4Matrix{packed: packed, scales: scales, inDim: inDim, outDim: outDim}, nil
	}

	m.layers = make([]mxfp4Layer, cfg.NumHiddenLayers)
	for i := range m.layers {
		hw := &w.Layers[i]
		l := &m.layers[i]
		for _, p := range []struct {
			name          string
			host          []float32
			outDim, inDim int
			dst           *mxfp4Matrix
		}{
			{"wq", hw.WQ, qDim, hidden, &l.WQ},
			{"wk", hw.WK, kvDim, hidden, &l.WK},
			{"wv", hw.WV, kvDim, hidden, &l.WV},
			{"wo", hw.WO, hidden, qDim, &l.WO},
			{"wgate", hw.WGate, inter, hidden, &l.WGate},
			{"wup", hw.WUp, inter, hidden, &l.WUp},
			{"wdown", hw.WDown, hidden, inter, &l.WDown},
		} {
			mat, err := upload(fmt.Sprintf("layer %d %s", i, p.name), p.host, p.outDim, p.inDim)
			if err != nil {
				return nil, fmt.Errorf("mxfp4 upload: %w", err)
			}
			*p.dst = mat
		}
	}
	if m.lmHead, err = upload("lm_head", w.LMHead, cfg.VocabSize, hidden); err != nil {
		return nil, fmt.Errorf("mxfp4 upload: %w", err)
	}
	return m, nil
}

// linear computes out = W x + bias for one projection on stream, reading W
// from its MXFP4 copy. bias may be empty.
func (m *mxfp4Weights) linear(stream *cuda.Stream, launcher func(*cuda.Kernel) *cudalaunch.Launcher, gm *GPUModules, x cudamem.Slice[float32], mat *mxfp4Matrix, bias WeightSlice, out cudamem.Slice[float32]) error {
	if m == nil || mat == nil {
		return errors.New("mxfp4 linear: no weights")
	}
	if err := m.engine.LinearGEMVOn(stream, x, mat.packed, mat.scales, out, mat.inDim, mat.outDim, mxfp4GroupSize); err != nil {
		return err
	}
	if bias.Len() == 0 {
		return nil
	}
	return launcher(gm.BatchAddBias).Grid(1, 1).Block(256).
		Launch(out, bias, int32(mat.outDim), int32(1))
}

// close releases the engine and the device copies. Safe to call more than
// once and on nil.
func (m *mxfp4Weights) close() error {
	if m == nil {
		return nil
	}
	var errs []error
	if m.arena != nil {
		errs = append(errs, m.arena.Free())
		m.arena = nil
	}
	if m.engine != nil {
		errs = append(errs, m.engine.Close())
		m.engine = nil
	}
	return errors.Join(errs...)
}
