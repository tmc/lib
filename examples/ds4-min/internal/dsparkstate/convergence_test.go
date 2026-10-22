package dsparkstate

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"testing"
)

// This literal writer is retained from the reviewed pre-extraction native
// codec. Its native kind selection does not use the new Platform field.
// Validation is shared: these controls concern exact wire-byte convergence.
var retainedMagic = [8]byte{'D', 'S', '4', 'S', 'U', 'P', '0', '1'}
var retainedContainerMagic = [8]byte{'D', 'S', '4', 'S', 'U', 'P', 'S', '1'}

func retainedNativeContainer(dst io.Writer, core []byte, support *Envelope, frontier int, generation uint64, maximum uint64) error {
	if dst == nil || len(core) == 0 {
		return errors.New("dspark support container: nil writer or empty core")
	}
	var encoded bytes.Buffer
	if err := retainedNativeWrite(&encoded, support, frontier, generation); err != nil {
		return err
	}
	kind := uint8(2)
	header := uint64(25)
	if support.Capture != nil {
		kind = 1
		header += 8
	}
	if uint64(len(core)) > maximum || uint64(encoded.Len()) > maximum-uint64(len(core)) || header > maximum-uint64(len(core))-uint64(encoded.Len()) {
		return errors.New("dspark support container: payload exceeds limit")
	}
	w := retainedWriter{w: dst}
	w.value(retainedContainerMagic)
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

type retainedWriter struct {
	w   io.Writer
	err error
}

type retainedFullWriter struct{ dst io.Writer }

func (w retainedFullWriter) Write(data []byte) (int, error) {
	n, err := w.dst.Write(data)
	if n < 0 || n > len(data) {
		return 0, io.ErrShortWrite
	}
	if n != len(data) && err == nil {
		err = io.ErrShortWrite
	}
	return n, err
}

func (w *retainedWriter) value(value any) {
	if w.err == nil {
		w.err = binary.Write(retainedFullWriter{w.w}, binary.LittleEndian, value)
	}
}

func (w *retainedWriter) floats(values []float32) {
	for _, value := range values {
		w.value(math.Float32bits(value))
	}
}

func retainedNativeWrite(dst io.Writer, v *Envelope, frontier int, generation uint64) error {
	if dst == nil {
		return errors.New("dspark support codec: nil writer")
	}
	if err := v.Validate(frontier, generation); err != nil {
		return err
	}
	w := retainedWriter{w: dst}
	w.value(retainedMagic)
	kind := uint8(2) // actual Metal ring representation
	if v.Capture != nil {
		kind = 1 // actual CUDA selected row plus history representation
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

func TestNativeWriterConvergence(t *testing.T) {
	t.Run("container wire kinds", testContainerWireKinds)
	for _, platform := range []uint8{1, 2} {
		for _, frontier := range []int{0, 2, 134} {
			rows, values, err := Shape(256, 1, 2, 4096)
			if err != nil {
				t.Fatal(err)
			}
			j, err := NewJournal(rows, 7)
			if err != nil {
				t.Fatal(err)
			}
			for p := 0; p < frontier; p++ {
				if err := j.Append(p, p+4, 7); err != nil {
					t.Fatal(err)
				}
			}
			v := &Envelope{Platform: platform, Model: sha256.Sum256([]byte("native frozen identity")), Context: 256, Block: 1, Targets: []int{1, 3}, Journal: j, History: make([]float32, values)}
			for i := range v.History {
				v.History[i] = float32(i%23-11) / 8
			}
			if platform == 1 {
				c := &Capture{Position: -1, Token: -1, Captured: make([]bool, 2), Features: make([]float32, 2*4096)}
				if frontier > 0 {
					c.Generation = 1
					c.Position = frontier - 1
					c.Token = frontier + 3
					c.Published = true
					for i := range c.Captured {
						c.Captured[i] = true
					}
				}
				v.Capture = c
				v.History[len(v.History)-1] = math.Float32frombits(0x7fc01234)
			}
			var old, current bytes.Buffer
			if err := retainedNativeWrite(&old, v, frontier, 2); err != nil {
				t.Fatal(err)
			}
			if err := Write(&current, v, frontier, 2); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(old.Bytes(), current.Bytes()) {
				t.Fatalf("platform %d frontier %d changed native wire", platform, frontier)
			}
			old.Reset()
			current.Reset()
			core := []byte("retained native core bytes")
			if err := retainedNativeContainer(&old, core, v, frontier, 2, 1<<30); err != nil {
				t.Fatal(err)
			}
			if err := WriteContainer(&current, core, v, frontier, 2, 1<<30); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(old.Bytes(), current.Bytes()) {
				t.Fatalf("platform %d frontier %d changed native container", platform, frontier)
			}
		}
	}
}

func testContainerWireKinds(t *testing.T) {
	for _, platform := range []uint8{1, 2, 3} {
		t.Run(fmt.Sprint(platform), func(t *testing.T) {
			rows, values, err := Shape(8, 1, 1, 4096)
			if err != nil {
				t.Fatal(err)
			}
			journal, err := NewJournal(rows, 7)
			if err != nil {
				t.Fatal(err)
			}
			v := &Envelope{Platform: platform, Model: sha256.Sum256([]byte("container identity")),
				Context: 8, Block: 1, Targets: []int{1}, Journal: journal, History: make([]float32, values)}
			if platform == 1 {
				v.Capture = &Capture{Position: -1, Token: -1, Captured: make([]bool, 1), Features: make([]float32, 4096)}
			}
			const generation = uint64(9)
			core := []byte("unchanged platform core")
			var wire, envelope bytes.Buffer
			if err := Write(&envelope, v, 0, generation); err != nil {
				t.Fatal(err)
			}
			if err := WriteContainer(&wire, core, v, 0, generation, 1<<30); err != nil {
				t.Fatal(err)
			}
			gotCore, gotSupport, gotGeneration, err := ReadContainer(bytes.NewReader(wire.Bytes()), platform, 1<<30)
			wantGeneration := uint64(0)
			if platform == 1 {
				wantGeneration = generation
			}
			if err != nil || !bytes.Equal(gotCore, core) || !bytes.Equal(gotSupport, envelope.Bytes()) || gotGeneration != wantGeneration {
				t.Fatalf("container wire-kind roundtrip: platform=%d generation=%d want=%d err=%v", platform, gotGeneration, wantGeneration, err)
			}
			foreign := platform%3 + 1
			if _, _, _, err := ReadContainer(bytes.NewReader(wire.Bytes()), foreign, 1<<30); err == nil {
				t.Fatal("foreign container wire kind accepted")
			}
			for _, size := range []int{9, len(wire.Bytes()) - 1} {
				if _, _, _, err := ReadContainer(bytes.NewReader(wire.Bytes()[:size]), platform, 1<<30); err == nil {
					t.Fatal("truncated container accepted")
				}
			}
		})
	}
}
