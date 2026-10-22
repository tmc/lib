package dsparkstate

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
)

// This envelope owns host staging, not an attached target or a drafter's KV.
// Platform adapters must perform their actual synchronization and owner checks.
type Envelope struct {
	Platform uint8 // 1 CUDA, 2 Metal, 3 CPU

	Model          [32]byte
	Context, Block int
	Targets        []int
	Journal        *Journal
	History        []float32
	Capture        *Capture // CUDA only; Metal resets transient capture
}

type Capture struct {
	Generation      uint64
	Position, Token int
	Published       bool
	Captured        []bool
	Status          uint32
	Features        []float32
}

var magic = [8]byte{'D', 'S', '4', 'S', 'U', 'P', '0', '1'}

// validate does not infer a platform's synchronization, liveness or owner.
// CUDA padding preserves exact bits; only Metal's actual snapshot requires
// the entire feature ring to be finite. CPU snapshots use the same finite
// host-ring requirement, but carry their own distinct platform identity.
func (v *Envelope) Validate(frontier int, generation uint64) error {
	if v == nil || (v.Platform != 0 && v.Platform != 1 && v.Platform != 2 && v.Platform != 3) || (v.Platform == 1 && v.Capture == nil) || (v.Platform >= 2 && v.Capture != nil) || v.Model == ([32]byte{}) || len(v.Targets) == 0 {
		return errors.New("dspark support codec: missing identity")
	}
	rows, values, err := Shape(v.Context, v.Block, len(v.Targets), 4096)
	if err != nil {
		return err
	}
	if v.Context > math.MaxUint32 || v.Block > math.MaxUint32 || v.Journal == nil || len(v.Journal.Rows) != rows || len(v.History) != values || frontier > v.Context {
		return errors.New("dspark support codec: shape mismatch")
	}
	seen := make(map[int]bool)
	for _, layer := range v.Targets {
		if layer < 0 || layer >= 43 || seen[layer] {
			return errors.New("dspark support codec: invalid selected layers")
		}
		seen[layer] = true
	}
	if err := v.Journal.Validate(frontier); err != nil {
		return fmt.Errorf("dspark support codec: Journal: %w", err)
	}
	if v.Capture == nil {
		for _, value := range v.History {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return errors.New("dspark support codec: nonfinite host ring")
			}
		}
		return nil
	}
	c := v.Capture
	if len(c.Captured) != len(v.Targets) || len(c.Features) != len(v.Targets)*4096 || c.Generation > generation {
		return errors.New("dspark support codec: capture shape or generation mismatch")
	}
	if c.Published {
		if c.Position < 0 || c.Position != frontier-1 || c.Token < 0 || c.Token >= 129280 || c.Generation == math.MaxUint64 || c.Generation+1 != generation {
			return errors.New("dspark support codec: published frontier mismatch")
		}
		row, err := v.Journal.Row(c.Position, v.Journal.Epoch)
		if err != nil || row.Token != c.Token {
			return errors.New("dspark support codec: published journal mismatch")
		}
	} else if c.Position != -1 || c.Token != -1 {
		return errors.New("dspark support codec: pending capture")
	}
	for _, captured := range c.Captured {
		if captured != c.Published {
			return errors.New("dspark support codec: incomplete capture")
		}
	}
	return nil
}

type writer struct {
	w   io.Writer
	err error
}

type fullWriter struct{ dst io.Writer }

func (w fullWriter) Write(data []byte) (int, error) {
	n, err := w.dst.Write(data)
	if n < 0 || n > len(data) {
		return 0, io.ErrShortWrite
	}
	if n != len(data) && err == nil {
		err = io.ErrShortWrite
	}
	return n, err
}

func (w *writer) value(value any) {
	if w.err == nil {
		w.err = binary.Write(fullWriter{w.w}, binary.LittleEndian, value)
	}
}

func (w *writer) floats(values []float32) {
	for _, value := range values {
		w.value(math.Float32bits(value))
	}
}

// Write is shared by serialization and the observing count sink.
// It writes no addresses, capacities, inferred device allocations or draft KV.
func Write(dst io.Writer, v *Envelope, frontier int, generation uint64) error {
	if dst == nil {
		return errors.New("dspark support codec: nil writer")
	}
	if err := v.Validate(frontier, generation); err != nil {
		return err
	}
	w := writer{w: dst}
	w.value(magic)
	kind := uint8(2) // actual Metal ring representation
	if v.Capture != nil {
		kind = 1 // actual CUDA selected row plus history representation
	}
	if v.Platform != 0 {
		kind = v.Platform
	}
	w.value(kind)
	w.value(v.Model)
	w.value(uint32(v.Context))
	w.value(uint32(v.Block))
	w.value(uint32(len(v.Targets)))
	for _, layer := range v.Targets {
		w.value(uint32(layer))
	}
	j := v.Journal
	w.value(int64(j.Frontier))
	w.value(j.Epoch)
	w.value(j.Prefix)
	w.value(j.BasePrefix)
	for _, row := range j.Rows {
		w.value(int64(row.Position))
		w.value(int64(row.Token))
		w.value(row.Prefix)
		w.value(row.Epoch)
	}
	w.floats(v.History)
	if c := v.Capture; c != nil {
		w.value(c.Generation)
		w.value(int64(c.Position))
		w.value(int64(c.Token))
		w.value(c.Published)
		w.value(c.Captured)
		w.value(c.Status)
		w.floats(c.Features)
	}
	if w.err != nil {
		return fmt.Errorf("dspark support codec: write: %w", w.err)
	}
	return nil
}

// Read checks the destination's trusted identity and exact extent
// before allocating staging. It never mutates or binds a destination owner.
func Read(data []byte, expected *Envelope, frontier int, generation uint64) (*Envelope, error) {
	if expected == nil || expected.Context <= 0 || expected.Block <= 0 || len(expected.Targets) == 0 {
		return nil, errors.New("dspark support codec: missing destination shape")
	}
	rows, values, err := Shape(expected.Context, expected.Block, len(expected.Targets), 4096)
	if err != nil {
		return nil, err
	}
	// All counts derive from trusted destination geometry, never wire counts.
	need := uint64(133) + uint64(len(expected.Targets))*4 + uint64(rows)*56 + uint64(values)*4
	if expected.Capture != nil {
		need += 29 + uint64(len(expected.Targets)) + uint64(len(expected.Targets))*uint64(4096)*4
	}
	if uint64(len(data)) != need {
		return nil, errors.New("dspark support codec: encoded extent mismatch")
	}
	r := bytes.NewReader(data)
	var gotMagic [8]byte
	var kind uint8
	var model [32]byte
	var context, block, count uint32
	for _, value := range []any{&gotMagic, &kind, &model, &context, &block, &count} {
		if err := binary.Read(r, binary.LittleEndian, value); err != nil {
			return nil, fmt.Errorf("dspark support codec: header: %w", err)
		}
	}
	wantKind := uint8(2)
	if expected.Capture != nil {
		wantKind = 1
	}
	if expected.Platform != 0 {
		wantKind = expected.Platform
	}
	if gotMagic != magic || kind != wantKind || model != expected.Model || uint64(context) != uint64(expected.Context) || uint64(block) != uint64(expected.Block) || uint64(count) != uint64(len(expected.Targets)) {
		return nil, errors.New("dspark support codec: destination identity mismatch")
	}
	v := &Envelope{Platform: wantKind, Model: model, Context: int(context), Block: int(block), Targets: make([]int, len(expected.Targets))}
	for i := range v.Targets {
		var layer uint32
		if err := binary.Read(r, binary.LittleEndian, &layer); err != nil {
			return nil, err
		}
		v.Targets[i] = int(layer)
	}
	if !slices.Equal(v.Targets, expected.Targets) {
		return nil, errors.New("dspark support codec: selected layer order mismatch")
	}
	j := &Journal{Rows: make([]Row, rows)}
	v.Journal = j
	var position int64
	for _, value := range []any{&position, &j.Epoch, &j.Prefix, &j.BasePrefix} {
		if err := binary.Read(r, binary.LittleEndian, value); err != nil {
			return nil, err
		}
	}
	if position != int64(frontier) {
		return nil, errors.New("dspark support codec: journal frontier mismatch")
	}
	j.Frontier = frontier
	for i := range j.Rows {
		var pos, token int64
		row := &j.Rows[i]
		for _, value := range []any{&pos, &token, &row.Prefix, &row.Epoch} {
			if err := binary.Read(r, binary.LittleEndian, value); err != nil {
				return nil, err
			}
		}
		if pos < -1 || pos > int64(expected.Context) || token < 0 || token >= int64(129280) {
			return nil, errors.New("dspark support codec: invalid physical row")
		}
		row.Position, row.Token = int(pos), int(token)
	}
	readFloats := func(n int) ([]float32, error) {
		out := make([]float32, n)
		for i := range out {
			var bits uint32
			if err := binary.Read(r, binary.LittleEndian, &bits); err != nil {
				return nil, err
			}
			out[i] = math.Float32frombits(bits)
		}
		return out, nil
	}
	v.History, err = readFloats(values)
	if err != nil {
		return nil, err
	}
	if kind == 1 {
		c := &Capture{Captured: make([]bool, len(v.Targets))}
		v.Capture = c
		var pos, token int64
		for _, value := range []any{&c.Generation, &pos, &token} {
			if err := binary.Read(r, binary.LittleEndian, value); err != nil {
				return nil, err
			}
		}
		readBool := func() (bool, error) {
			value, err := r.ReadByte()
			if err != nil {
				return false, err
			}
			if value > 1 {
				return false, errors.New("dspark support codec: noncanonical boolean")
			}
			return value == 1, nil
		}
		c.Published, err = readBool()
		if err != nil {
			return nil, err
		}
		for i := range c.Captured {
			c.Captured[i], err = readBool()
			if err != nil {
				return nil, err
			}
		}
		if err := binary.Read(r, binary.LittleEndian, &c.Status); err != nil {
			return nil, err
		}
		if pos < -1 || pos > int64(expected.Context) || token < -1 || token >= int64(129280) {
			return nil, errors.New("dspark support codec: invalid capture frontier")
		}
		c.Position, c.Token = int(pos), int(token)
		c.Features, err = readFloats(len(v.Targets) * 4096)
		if err != nil {
			return nil, err
		}
	}
	if r.Len() != 0 {
		return nil, errors.New("dspark support codec: trailing bytes")
	}
	if err := v.Validate(frontier, generation); err != nil {
		return nil, err
	}
	return v, nil
}
