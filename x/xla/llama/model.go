package llama

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/tmc/lib/openxla"
	"github.com/tmc/lib/x/stablehlo/builder"
	"github.com/tmc/lib/x/xla"
)

// Model owns device weights, history and compiled programs. Its zero value is
// unusable. All calls, including Close, must be externally synchronized.
type Model struct {
	config                Config
	engine                *xla.Engine
	device                *openxla.Device
	types                 []builder.Type
	parameters, history   []*xla.Buffer
	layers                []layer
	embedding, norm, head int
	frequencies           []float64
	lengths               []int64
	plans                 map[int64]*xla.Executable
	closed                bool
}

// New validates and uploads finite f32 weights and initializes zero history.
// It borrows engine, which must remain open until the model is closed. Weight
// slices must not change during New; they may be reused after New returns.
// Cancellation waits for any in-flight native call before releasing resources.
func New(ctx context.Context, engine *xla.Engine, config Config, weights Weights) (*Model, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	spec, err := describe(config, weights)
	if err != nil {
		return nil, err
	}
	return newModel(ctx, engine, config, spec)
}

func newModel(ctx context.Context, engine *xla.Engine, c Config, s layout) (*Model, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	devices, err := engine.Devices()
	if err != nil {
		return nil, err
	}
	if len(devices) == 0 {
		return nil, fmt.Errorf("llama: no addressable devices")
	}
	m := &Model{config: c, engine: engine, layers: s.layers, embedding: s.embedding, norm: s.norm, head: s.head, frequencies: s.frequencies, lengths: make([]int64, c.BatchSize), plans: make(map[int64]*xla.Executable)}
	if err := m.initialize(ctx); err != nil {
		return nil, errors.Join(err, m.Close())
	}
	for _, p := range s.parameters {
		if err := ctx.Err(); err != nil {
			return nil, errors.Join(err, m.Close())
		}
		data := p.value.raw
		if p.value.data != nil {
			data = encodeFloats(p.value.data)
		}
		buffer, err := engine.NewBufferFromHost(data, p.value.kind, p.typ.Shape, m.device)
		if err != nil {
			return nil, errors.Join(fmt.Errorf("llama: upload %s: %w", p.name, err), m.Close())
		}
		m.parameters = append(m.parameters, buffer)
		m.types = append(m.types, p.typ)
	}
	if err := ctx.Err(); err != nil {
		return nil, errors.Join(err, m.Close())
	}
	return m, nil
}

// Device returns the device selected by the initialization graph.
func (m *Model) Device() (*openxla.Device, error) {
	if m == nil || m.closed || m.device == nil {
		return nil, fmt.Errorf("llama: model is closed")
	}
	return m.device, nil
}

// Lengths returns a copy of the cached length for each batch member. It returns
// nil for a closed or uninitialized model.
func (m *Model) Lengths() []int64 {
	if m == nil || m.closed {
		return nil
	}
	return slices.Clone(m.lengths)
}

// Reset discards all logical history while retaining buffers and programs.
func (m *Model) Reset() error {
	if _, err := m.Device(); err != nil {
		return err
	}
	clear(m.lengths)
	return nil
}

// Forward appends token IDs and returns f32 logits shaped [BatchSize, chunk,
// VocabSize]. IDs are row-major; chunk is inferred from len(tokenIDs)/BatchSize
// and must be positive and at most Capacity. Nil counts means a full chunk for
// every member. Otherwise counts contains BatchSize entries in [0,chunk], each
// fitting its member's remaining capacity. Padded IDs are ignored.
//
// Only each member's first count output rows are meaningful. The caller owns
// the returned buffer. Execution and cancellation errors preserve every layer's
// history. Failure to release metadata, replaced history or an evicted program
// closes the model. Inputs must not change during the call.
func (m *Model) Forward(ctx context.Context, tokenIDs []int32, counts []int64) (*xla.Buffer, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if _, err := m.Device(); err != nil {
		return nil, err
	}
	b := m.config.BatchSize
	if int64(len(tokenIDs))%b != 0 {
		return nil, fmt.Errorf("llama: token count is not divisible by batch size")
	}
	chunk := int64(len(tokenIDs)) / b
	if chunk <= 0 || chunk > m.config.Capacity {
		return nil, fmt.Errorf("llama: invalid chunk length")
	}
	if counts == nil {
		counts = make([]int64, b)
		for i := range counts {
			counts[i] = chunk
		}
	} else {
		counts = slices.Clone(counts)
	}
	if len(counts) != len(m.lengths) {
		return nil, fmt.Errorf("llama: count length does not match batch size")
	}
	ids := make([]int64, len(tokenIDs))
	for batch, n := range counts {
		if n < 0 || n > chunk || n > m.config.Capacity-m.lengths[batch] {
			return nil, fmt.Errorf("llama: batch %d count exceeds chunk or capacity", batch)
		}
		for row := int64(0); row < n; row++ {
			i := int64(batch)*chunk + row
			id := int64(tokenIDs[i])
			if id < 0 || id >= m.config.VocabSize {
				return nil, fmt.Errorf("llama: batch %d token %d outside vocabulary", batch, row)
			}
			ids[i] = id
		}
	}
	return m.forward(ctx, ids, counts, chunk)
}

func (m *Model) forward(ctx context.Context, ids, counts []int64, chunk int64) (*xla.Buffer, error) {
	p, err := m.program(chunk)
	if err != nil {
		return nil, err
	}
	c := m.config
	indices := make([]int64, c.BatchSize*c.KVHeads*chunk)
	cosine, sine := make([]float32, c.BatchSize*chunk*(c.HeadSize/2)), make([]float32, c.BatchSize*chunk*(c.HeadSize/2))
	for batch, n := range counts {
		for head := int64(0); head < c.KVHeads; head++ {
			row := int64(batch)*c.KVHeads + head
			for t := int64(0); t < chunk; t++ {
				index := int64(-1)
				if t < n {
					index = row*c.Capacity + m.lengths[batch] + t
				}
				indices[row*chunk+t] = index
			}
		}
		for t := int64(0); t < chunk; t++ {
			pos := int64(0)
			if t < n {
				pos = m.lengths[batch] + t
			}
			for i, f := range m.frequencies {
				index := (int64(batch)*chunk+t)*(c.HeadSize/2) + int64(i)
				s, v := math.Sincos(float64(pos) * f)
				cosine[index], sine[index] = float32(v), float32(s)
			}
		}
	}
	data := [][]byte{encodeInts(ids), encodeInts(m.lengths), encodeInts(counts), encodeInts(indices), encodeFloats(cosine), encodeFloats(sine)}
	types := m.metadataTypes(chunk)
	var metadata []*xla.Buffer
	for i, typ := range types {
		element := openxla.Buffer_Type_S64
		if typ.ElementType == "f32" {
			element = xla.F32
		}
		buffer, err := m.engine.NewBufferFromHost(data[i], element, typ.Shape, m.device)
		if err != nil {
			if cleanupErr := closeBuffers(metadata); cleanupErr != nil {
				return nil, errors.Join(err, cleanupErr, m.Close())
			}
			return nil, err
		}
		metadata = append(metadata, buffer)
	}
	args := append(slices.Clone(m.parameters), m.history...)
	args = append(args, metadata...)
	outputs, err := p.Execute(ctx, args)
	if cleanupErr := closeBuffers(metadata); cleanupErr != nil {
		return nil, errors.Join(err, cleanupErr, closeBuffers(outputs), m.Close())
	}
	if err != nil {
		return nil, err
	}
	if len(outputs) != len(m.history)+1 {
		return nil, errors.Join(fmt.Errorf("llama: unexpected output count"), closeBuffers(outputs))
	}
	return m.commit(outputs, counts, (*xla.Buffer).Close)
}

func (m *Model) commit(outputs []*xla.Buffer, counts []int64, release func(*xla.Buffer) error) (*xla.Buffer, error) {
	var err error
	for _, b := range m.history {
		err = errors.Join(err, release(b))
	}
	m.history = nil
	if err != nil {
		return nil, errors.Join(err, closeBuffers(outputs), m.Close())
	}
	m.history = outputs[:len(outputs)-1]
	for i, n := range counts {
		m.lengths[i] += n
	}
	return outputs[len(outputs)-1], nil
}

// Close attempts to release all owned resources and invalidates the model,
// even if native destruction fails. It does not close the borrowed engine.
func (m *Model) Close() error {
	if m == nil {
		return nil
	}
	m.closed = true
	m.device = nil
	m.lengths = nil
	err := errors.Join(closeBuffers(m.history), closeBuffers(m.parameters))
	m.history = nil
	m.parameters = nil
	for n, p := range m.plans {
		err = errors.Join(err, p.Close())
		delete(m.plans, n)
	}
	return err
}

func closeBuffers(buffers []*xla.Buffer) (err error) {
	for _, b := range buffers {
		err = errors.Join(err, b.Close())
	}
	return err
}
func encodeInts(values []int64) []byte {
	b := make([]byte, 8*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint64(b[i*8:], uint64(v))
	}
	return b
}
func encodeFloats(values []float32) []byte {
	b := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}
