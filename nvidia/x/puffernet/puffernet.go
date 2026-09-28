package puffernet

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// Config is the shape of a network. PufferLib reads InputSize and
// ActionSizes from the environment (OBS_SIZE and ACT_SIZES) and HiddenSize
// and NumLayers from the [policy] section of its config.
type Config struct {
	InputSize   int
	HiddenSize  int
	NumLayers   int
	ActionSizes []int
}

func (c Config) validate() error {
	if c.InputSize <= 0 || c.HiddenSize <= 0 || c.NumLayers <= 0 {
		return fmt.Errorf("puffernet: invalid config: input %d, hidden %d, layers %d",
			c.InputSize, c.HiddenSize, c.NumLayers)
	}
	if len(c.ActionSizes) == 0 {
		return errors.New("puffernet: invalid config: no action heads")
	}
	for i, n := range c.ActionSizes {
		if n <= 0 {
			return fmt.Errorf("puffernet: invalid config: action head %d has size %d", i, n)
		}
	}
	if c.NumFloats() > math.MaxInt32 {
		return errors.New("puffernet: invalid config: too many weights")
	}
	return nil
}

// Continuous reports whether the actions are continuous, which PufferLib
// signals by giving every action head size 1.
func (c Config) Continuous() bool {
	for _, n := range c.ActionSizes {
		if n != 1 {
			return false
		}
	}
	return true
}

// OutputSize is the length of one row of output: the action logits and the
// value.
func (c Config) OutputSize() int {
	n := 1
	for _, a := range c.ActionSizes {
		n += a
	}
	return n
}

// NumFloats is the number of float32 values in a checkpoint, including the
// padding that aligns each tensor to 8 values.
func (c Config) NumFloats() int {
	n := 0
	for _, t := range c.tensors() {
		n = align8(n + t)
	}
	return n
}

// tensors returns the length of each tensor in checkpoint order.
func (c Config) tensors() []int {
	h := c.HiddenSize
	t := []int{h * c.InputSize, c.OutputSize() * h}
	if c.Continuous() {
		t = append(t, len(c.ActionSizes))
	}
	for range c.NumLayers {
		t = append(t, 3*h*h)
	}
	return t
}

func align8(n int) int {
	return (n + 7) &^ 7
}

// Weights are the parameters of a network. The slices share one backing
// array laid out as in the checkpoint.
type Weights struct {
	Config Config

	// Encoder is HiddenSize×InputSize, row major.
	Encoder []float32

	// Decoder is OutputSize×HiddenSize, row major.
	Decoder []float32

	// LogStd holds one value per action head if the actions are continuous,
	// and is nil otherwise.
	LogStd []float32

	// Layers holds, for each MinGRU layer, its 3·HiddenSize×HiddenSize
	// projection: the rows of the hidden, gate and highway blocks in that
	// order.
	Layers [][]float32

	flat []float32
}

// NewWeights returns the weights stored in flat, which must be laid out as a
// checkpoint for cfg. It keeps flat.
//
// PufferLib does not write the padding after the last tensor, so flat may be
// up to 7 values shorter than cfg.NumFloats.
func NewWeights(cfg Config, flat []float32) (*Weights, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	need := cfg.NumFloats()
	if len(flat) > need || len(flat) < need-7 {
		return nil, fmt.Errorf("puffernet: have %d weights, config needs %d", len(flat), need)
	}
	if len(flat) < need {
		flat = append(flat, make([]float32, need-len(flat))...)
	}
	w := &Weights{Config: cfg, flat: flat}
	off := 0
	next := func(n int) []float32 {
		s := flat[off : off+n : off+n]
		off = align8(off + n)
		return s
	}
	h := cfg.HiddenSize
	w.Encoder = next(h * cfg.InputSize)
	w.Decoder = next(cfg.OutputSize() * h)
	if cfg.Continuous() {
		w.LogStd = next(len(cfg.ActionSizes))
	}
	for range cfg.NumLayers {
		w.Layers = append(w.Layers, next(3*h*h))
	}
	return w, nil
}

// ReadWeights reads a checkpoint for cfg from r: little-endian float32
// values, as written by ./puffer train.
func ReadWeights(r io.Reader, cfg Config) (*Weights, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	b, err := io.ReadAll(io.LimitReader(r, int64(cfg.NumFloats())*4+1))
	if err != nil {
		return nil, fmt.Errorf("puffernet: read weights: %w", err)
	}
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("puffernet: weights are %d bytes, not a whole number of float32s", len(b))
	}
	flat := make([]float32, len(b)/4)
	for i := range flat {
		flat[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return NewWeights(cfg, flat)
}

// LoadWeights reads the checkpoint file at path.
func LoadWeights(path string, cfg Config) (*Weights, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("puffernet: %w", err)
	}
	defer f.Close()
	return ReadWeights(f, cfg)
}

// Host runs the network on the CPU. It computes what puffercpu.c computes, in
// the same order, and is the reference [Net] is tested against.
type Host struct {
	w     *Weights
	batch int
	state []float32 // NumLayers×batch×HiddenSize
	x, y  []float32 // batch×HiddenSize
	fused []float32 // batch×3·HiddenSize
}

// NewHost returns a Host for batch agents with a zero recurrent state.
func NewHost(w *Weights, batch int) *Host {
	h := w.Config.HiddenSize
	return &Host{
		w:     w,
		batch: batch,
		state: make([]float32, w.Config.NumLayers*batch*h),
		x:     make([]float32, batch*h),
		y:     make([]float32, batch*h),
		fused: make([]float32, batch*3*h),
	}
}

// Reset zeroes the recurrent state of every agent.
func (n *Host) Reset() {
	clear(n.state)
}

// Step advances every agent by one step and writes batch×OutputSize values
// to out. obs holds batch×InputSize observations. If terminals is not nil,
// the state of each agent whose terminal flag is above 0.5 is zeroed first,
// as the trainer does after an episode ends.
func (n *Host) Step(out, obs, terminals []float32) error {
	cfg := n.w.Config
	B, H := n.batch, cfg.HiddenSize
	if len(obs) != B*cfg.InputSize || len(out) != B*cfg.OutputSize() ||
		terminals != nil && len(terminals) != B {
		return fmt.Errorf("puffernet: step: have obs %d, out %d, terminals %d for batch %d",
			len(obs), len(out), len(terminals), B)
	}
	for b := range B {
		if terminals != nil && terminals[b] > 0.5 {
			for l := range cfg.NumLayers {
				clear(n.state[(l*B+b)*H : (l*B+b+1)*H])
			}
		}
	}
	linear(n.x, obs, n.w.Encoder, B, cfg.InputSize, H)
	x, y := n.x, n.y
	for l, proj := range n.w.Layers {
		linear(n.fused, x, proj, B, H, 3*H)
		state := n.state[l*B*H : (l+1)*B*H]
		for b := range B {
			c := n.fused[b*3*H : (b+1)*3*H]
			for h := range H {
				hidden, gate, hw := c[h], c[H+h], c[2*H+h]
				s := state[b*H+h]
				var tilde float32
				if hidden >= 0 {
					tilde = hidden + 0.5
				} else {
					tilde = sigmoid(hidden)
				}
				m := s + float32(sigmoid(gate)*(tilde-s))
				hs := sigmoid(hw)
				y[b*H+h] = float32(hs*m) + float32((1-hs)*x[b*H+h])
				state[b*H+h] = m
			}
		}
		x, y = y, x
	}
	linear(out, x, n.w.Decoder, B, H, cfg.OutputSize())
	return nil
}

// linear computes out = in·wᵀ for in batch×k and w n×k, summing in index
// order. The float32 conversions stop the compiler fusing a multiply and
// add, so each product is rounded as the C source writes it.
func linear(out, in, w []float32, batch, k, n int) {
	for b := range batch {
		row := in[b*k : (b+1)*k]
		for o := range n {
			col := w[o*k : (o+1)*k]
			var sum float32
			for i, v := range row {
				sum += float32(v * col[i])
			}
			out[b*n+o] = sum
		}
	}
}

func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(float64(-x))))
}
