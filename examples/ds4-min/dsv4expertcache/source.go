package dsv4expertcache

import (
	"errors"
	"fmt"
	"io"
	"math"
)

// Tensor describes an admitted routed tensor's actual file extent. Stride is
// the expert-axis stride, not an inferred dense dimension.
type Tensor struct {
	Offset, Bytes, Stride uint64
	Experts               int
}

// Triplet contains actual gate, up and down tensor descriptors for one layer.
type Triplet struct{ Gate, Up, Down Tensor }

// Source reads an immutable, already-open admitted model. The caller keeps its
// reader alive and immutable until Source is no longer used; NewSource does not
// hash or reopen it. Identity must come from the model owner's admission gate.
type Source struct {
	reader   io.ReaderAt
	identity Identity
	layers   map[int]Triplet
}

// NewSource validates each extent against the admitted file size. It rejects
// padding or inconsistent expert strides rather than guessing a packing layout.
func NewSource(reader io.ReaderAt, size uint64, identity Identity, layers map[int]Triplet) (*Source, error) {
	if nilCompletion(reader) || size > math.MaxInt64 || identity.Model == ([32]byte{}) || identity.Quant == ([32]byte{}) || identity.Layout == ([32]byte{}) || len(layers) == 0 {
		return nil, errors.New("invalid admitted source")
	}
	s := &Source{reader: reader, identity: identity, layers: make(map[int]Triplet, len(layers))}
	for layer, tr := range layers {
		if layer < 0 {
			return nil, errors.New("invalid source layer")
		}
		for _, t := range []Tensor{tr.Gate, tr.Up, tr.Down} {
			if t.Experts <= 0 || t.Stride == 0 || uint64(t.Experts) > math.MaxUint64/t.Stride || t.Bytes != uint64(t.Experts)*t.Stride || t.Offset > size || t.Bytes > size-t.Offset {
				return nil, fmt.Errorf("invalid layer %d tensor extent", layer)
			}
		}
		if tr.Gate.Experts != tr.Up.Experts || tr.Gate.Experts != tr.Down.Experts {
			return nil, errors.New("expert counts differ")
		}
		s.layers[layer] = tr
	}
	return s, nil
}

// Read reads one component of a qualified expert into an exactly sized buffer.
// Component is 0 for gate, 1 for up and 2 for down. No partial read is accepted.
func (s *Source) Read(key Key, component int, dst []byte) error {
	if s == nil || key.Identity != s.identity {
		return errors.New("source identity mismatch")
	}
	tr, ok := s.layers[key.Layer]
	if !ok || component < 0 || component > 2 {
		return errors.New("invalid expert component")
	}
	t := []Tensor{tr.Gate, tr.Up, tr.Down}[component]
	if key.Expert < 0 || key.Expert >= t.Experts || uint64(len(dst)) != t.Stride {
		return errors.New("invalid expert buffer")
	}
	n, err := s.reader.ReadAt(dst, int64(t.Offset+uint64(key.Expert)*t.Stride))
	if n != len(dst) {
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return fmt.Errorf("read expert component: %w", err)
	}
	if err != nil {
		return fmt.Errorf("read expert component: %w", err)
	}
	return nil
}

// Transfer submits one component from staging storage. Completion is repeatable
// and covers any submitted writes even when Submit returns an error. Submit must
// not retain src after completion. The adapter rejects graph capture until a
// separate graph-lifetime qualification exists.
type Transfer interface {
	Submit(upload Upload, component int, src []byte) (Completion, error)
}
