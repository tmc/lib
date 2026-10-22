package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

var retainedSupportMagic = [8]byte{'D', 'S', '4', 'S', 'U', 'P', '0', '1'}

// validate does not infer a platform's synchronization, liveness or owner.
// CUDA padding preserves exact bits; only Metal's actual snapshot requires
// the entire feature ring to be finite.
func (v *dsparkSupportEnvelope) validateRetained(frontier int, generation uint64) error {
	if v == nil || v.model == ([32]byte{}) || len(v.targets) == 0 {
		return errors.New("dspark support codec: missing identity")
	}
	rows, values, err := dsparkFeatureJournalShape(v.context, v.block, len(v.targets), Flash.Embed)
	if err != nil {
		return err
	}
	if v.context > math.MaxUint32 || v.block > math.MaxUint32 || v.journal == nil || len(v.journal.rows) != rows || len(v.history) != values || frontier > v.context {
		return errors.New("dspark support codec: shape mismatch")
	}
	seen := make(map[int]bool)
	for _, layer := range v.targets {
		if layer < 0 || layer >= Flash.Layers || seen[layer] {
			return errors.New("dspark support codec: invalid selected layers")
		}
		seen[layer] = true
	}
	if err := v.journal.validate(frontier); err != nil {
		return fmt.Errorf("dspark support codec: journal: %w", err)
	}
	if v.capture == nil {
		for _, value := range v.history {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return errors.New("dspark support codec: nonfinite Metal ring")
			}
		}
		return nil
	}
	c := v.capture
	if len(c.captured) != len(v.targets) || len(c.features) != len(v.targets)*Flash.Embed || c.generation > generation {
		return errors.New("dspark support codec: capture shape or generation mismatch")
	}
	if c.published {
		if c.position < 0 || c.position != frontier-1 || c.token < 0 || c.token >= Flash.Vocab || c.generation == math.MaxUint64 || c.generation+1 != generation {
			return errors.New("dspark support codec: published frontier mismatch")
		}
		row, err := v.journal.row(c.position, v.journal.epoch)
		if err != nil || row.token != c.token {
			return errors.New("dspark support codec: published journal mismatch")
		}
	} else if c.position != -1 || c.token != -1 {
		return errors.New("dspark support codec: pending capture")
	}
	for _, captured := range c.captured {
		if captured != c.published {
			return errors.New("dspark support codec: incomplete capture")
		}
	}
	return nil
}

type retainedSupportWriter struct {
	w   io.Writer
	err error
}

type retainedSupportFullWriter struct{ dst io.Writer }

func (w retainedSupportFullWriter) Write(data []byte) (int, error) {
	n, err := w.dst.Write(data)
	if n < 0 || n > len(data) {
		return 0, io.ErrShortWrite
	}
	if n != len(data) && err == nil {
		err = io.ErrShortWrite
	}
	return n, err
}

func (w *retainedSupportWriter) value(value any) {
	if w.err == nil {
		w.err = binary.Write(retainedSupportFullWriter{w.w}, binary.LittleEndian, value)
	}
}

func (w *retainedSupportWriter) floats(values []float32) {
	for _, value := range values {
		w.value(math.Float32bits(value))
	}
}

// writeRetainedDSparkSupport is shared by serialization and the observing count sink.
// It writes no addresses, capacities, inferred device allocations or draft KV.
func writeRetainedDSparkSupport(dst io.Writer, v *dsparkSupportEnvelope, frontier int, generation uint64) error {
	if dst == nil {
		return errors.New("dspark support codec: nil writer")
	}
	if err := v.validateRetained(frontier, generation); err != nil {
		return err
	}
	w := retainedSupportWriter{w: dst}
	w.value(retainedSupportMagic)
	kind := uint8(2) // actual Metal ring representation
	if v.capture != nil {
		kind = 1 // actual CUDA selected row plus history representation
	}
	w.value(kind)
	w.value(v.model)
	w.value(uint32(v.context))
	w.value(uint32(v.block))
	w.value(uint32(len(v.targets)))
	for _, layer := range v.targets {
		w.value(uint32(layer))
	}
	j := v.journal
	w.value(int64(j.frontier))
	w.value(j.epoch)
	w.value(j.prefix)
	w.value(j.basePrefix)
	for _, row := range j.rows {
		w.value(int64(row.position))
		w.value(int64(row.token))
		w.value(row.prefix)
		w.value(row.epoch)
	}
	w.floats(v.history)
	if c := v.capture; c != nil {
		w.value(c.generation)
		w.value(int64(c.position))
		w.value(int64(c.token))
		w.value(c.published)
		w.value(c.captured)
		w.value(c.status)
		w.floats(c.features)
	}
	if w.err != nil {
		return fmt.Errorf("dspark support codec: write: %w", w.err)
	}
	return nil
}

var retainedSupportContainerMagic = [8]byte{'D', 'S', '4', 'S', 'U', 'P', 'S', '1'}

// The container preserves the existing platform core payload byte-for-byte.
// Only CUDA stores a core capture generation; Metal has no such field.
func writeRetainedDSparkSupportContainer(dst io.Writer, core []byte, support *dsparkSupportEnvelope, frontier int, generation uint64, maximum uint64) error {
	if dst == nil || len(core) == 0 {
		return errors.New("dspark support container: nil writer or empty core")
	}
	var encoded bytes.Buffer
	if err := writeRetainedDSparkSupport(&encoded, support, frontier, generation); err != nil {
		return err
	}
	kind := uint8(2)
	header := uint64(25)
	if support.capture != nil {
		kind = 1
		header += 8
	}
	if uint64(len(core)) > maximum || uint64(encoded.Len()) > maximum-uint64(len(core)) || header > maximum-uint64(len(core))-uint64(encoded.Len()) {
		return errors.New("dspark support container: payload exceeds limit")
	}
	w := retainedSupportWriter{w: dst}
	w.value(retainedSupportContainerMagic)
	w.value(kind)
	if kind == 1 {
		w.value(generation)
	}
	w.value(uint64(len(core)))
	w.value(uint64(encoded.Len()))
	w.value(core)
	w.value(encoded.Bytes())
	if w.err != nil {
		return fmt.Errorf("dspark support container: write: %w", w.err)
	}
	return nil
}
