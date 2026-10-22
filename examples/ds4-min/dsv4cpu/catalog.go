package dsv4cpu

import (
	"errors"
	"fmt"
	"io"
	"math"
)

// Tensor describes a serialized GGUF tensor. Dimensions are fastest first;
// Offset is absolute in the model ReaderAt. Bytes must match the encoding.
type Tensor struct {
	Name          string
	Dimensions    []int
	Encoding      Encoding
	Offset, Bytes int64
}

// Catalog binds immutable GGUF tensor descriptors to their source. It does not
// own the source lifetime. The source must remain immutable until all states
// using the catalog have closed.
type Catalog struct {
	source  io.ReaderAt
	tensors map[string]Tensor
}

// NewCatalog validates and copies descriptors without reading or expanding
// model weights. Duplicate names and overflowing/out-of-file spans are refused.
func NewCatalog(source io.ReaderAt, size int64, tensors []Tensor) (*Catalog, error) {
	if source == nil || size < 0 {
		return nil, errors.New("invalid model source")
	}
	c := &Catalog{source: source, tensors: make(map[string]Tensor, len(tensors))}
	for _, t := range tensors {
		if t.Name == "" || len(t.Dimensions) < 1 || len(t.Dimensions) > 3 || t.Offset < 0 || t.Bytes < 0 || t.Offset > size-t.Bytes {
			return nil, fmt.Errorf("invalid tensor %q extent", t.Name)
		}
		if _, ok := c.tensors[t.Name]; ok {
			return nil, fmt.Errorf("duplicate tensor %q", t.Name)
		}
		rows := 1
		for _, d := range t.Dimensions[1:] {
			if d <= 0 || rows > math.MaxInt/d {
				return nil, fmt.Errorf("tensor %q dimensions overflow", t.Name)
			}
			rows *= d
		}
		if _, err := NewMatrix(source, t.Offset, t.Bytes, rows, t.Dimensions[0], t.Encoding); err != nil {
			return nil, fmt.Errorf("tensor %q: %w", t.Name, err)
		}
		t.Dimensions = append([]int(nil), t.Dimensions...)
		c.tensors[t.Name] = t
	}
	return c, nil
}

// Matrix returns a whole 1D/2D tensor, or one expert slice of a 3D tensor.
// A 3D expert tensor is [input, output, expert] in GGUF dimension order.
func (c *Catalog) Matrix(name string, expert int) (*Matrix, error) {
	if c == nil || c.source == nil {
		return nil, errors.New("invalid tensor catalog")
	}
	t, ok := c.tensors[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", name)
	}
	rows := 1
	if len(t.Dimensions) > 1 {
		rows = t.Dimensions[1]
	}
	offset, size := t.Offset, t.Bytes
	if len(t.Dimensions) == 3 {
		if expert < 0 || expert >= t.Dimensions[2] {
			return nil, fmt.Errorf("tensor %q expert out of range", name)
		}
		size /= int64(t.Dimensions[2])
		offset += int64(expert) * size
	} else if expert != 0 {
		return nil, fmt.Errorf("tensor %q has no expert axis", name)
	}
	return NewMatrix(c.source, offset, size, rows, t.Dimensions[0], t.Encoding)
}
