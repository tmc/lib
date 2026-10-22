package dsv41

import "fmt"

const (
	HeadDim    = 512
	IndexDim   = 128
	WindowRows = 128
)

type ownerCache struct{ kv, index []float32 }

// KVState owns forty raw windows and four cross-layer compressed caches.
// Commit follows layer order for each scalar token. Source readers must supply
// already normalized, rotated and quantized rows; this type does not silently
// substitute a V4 cache or recompute those operations. It is request-local.
type KVState struct {
	context, position, layer int
	raw                      [40][]float32
	owners                   [4]ownerCache
}

// NewKVState creates an empty scalar V4.1 cache. Compressed storage grows only
// for published rows; each raw window has a fixed 128-row capacity.
func NewKVState(context int) (*KVState, error) {
	if context <= 0 || context > 1048576 {
		return nil, fmt.Errorf("invalid KV context")
	}
	s := &KVState{context: context}
	for i := range s.raw {
		s.raw[i] = make([]float32, WindowRows*HeadDim)
	}
	return s, nil
}

// Reset clears all request state without changing its context capacity.
func (s *KVState) Reset() {
	s.position = 0
	s.layer = 0
	for i := range s.raw {
		clear(s.raw[i])
	}
	for i := range s.owners {
		s.owners[i].kv = s.owners[i].kv[:0]
		s.owners[i].index = s.owners[i].index[:0]
	}
}

// Commit writes one layer's raw row and, at source emission positions, its
// compressed KV and index row. Non-source layers must supply nil rows.
// Validation failures leave all buffers and the cursor unchanged.
func (s *KVState) Commit(layer, position int, raw, compressed, index []float32) error {
	l, err := Schedule(layer)
	if err != nil {
		return err
	}
	if s == nil || position != s.position || layer != s.layer || position >= s.context || len(raw) != HeadDim {
		return fmt.Errorf("invalid KV commit position or shape")
	}
	emit := l.PublishKV && (position+1)%l.Ratio == 0
	if emit {
		if len(compressed) != HeadDim || len(index) != IndexDim {
			return fmt.Errorf("missing published KV or index row")
		}
	} else if len(compressed) != 0 || len(index) != 0 {
		return fmt.Errorf("unexpected compressed publication")
	}
	if l.Ratio != 0 {
		n := (position + 1) / l.Ratio
		have := len(s.owners[l.Owner].kv) / HeadDim
		if emit {
			have++
		}
		if have != n {
			return fmt.Errorf("compressed KV owner not ready")
		}
	}
	copy(s.raw[layer][(position%WindowRows)*HeadDim:], raw)
	if emit {
		o := &s.owners[l.Owner]
		o.kv = append(o.kv, compressed...)
		o.index = append(o.index, index...)
	}
	s.layer++
	if s.layer == 40 {
		s.layer = 0
		s.position++
	}
	return nil
}

// Rows returns copies of a layer's currently visible raw and compressed rows
// and its owner's index rows. Raw rows are in chronological order. This method
// can be called immediately after Commit for the attention consumer.
func (s *KVState) Rows(layer, position int) (raw, compressed, index []float32, err error) {
	l, err := Schedule(layer)
	if err != nil {
		return nil, nil, nil, err
	}
	if s == nil || position < 0 || position >= s.context || position > s.position || position == s.position && layer >= s.layer || position < s.position-1 || position == s.position-1 && layer < s.layer {
		return nil, nil, nil, fmt.Errorf("KV rows not committed at position")
	}
	n := min(position+1, WindowRows)
	raw = make([]float32, n*HeadDim)
	for i := 0; i < n; i++ {
		slot := (position + 1 - n + i) % WindowRows
		copy(raw[i*HeadDim:], s.raw[layer][slot*HeadDim:(slot+1)*HeadDim])
	}
	if l.Ratio != 0 {
		count := (position + 1) / l.Ratio
		o := s.owners[l.Owner]
		if len(o.kv) < count*HeadDim {
			return nil, nil, nil, fmt.Errorf("KV owner not ready")
		}
		compressed = append([]float32(nil), o.kv[:count*HeadDim]...)
		index = append([]float32(nil), o.index[:count*IndexDim]...)
	}
	return raw, compressed, index, nil
}
