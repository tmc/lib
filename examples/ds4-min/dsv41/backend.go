package dsv41

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"sync"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// Backend adapts a bound scalar model to independent dsv4 request states.
// The model's immutable ReaderAt remains borrowed: its owner must keep it open
// until Close returns. Backend Close waits for active calls and retires every
// request before returning. Different requests may execute concurrently.
type Backend struct {
	mu     sync.RWMutex
	model  *Model
	states map[*request]bool
}

// NewBackend creates a request owner over an already bound model. Bind permits
// ordinary execution; only BindVerified also permits portable Save and Load.
func NewBackend(model *Model) (*Backend, error) {
	if model == nil || model.catalog == nil {
		return nil, fmt.Errorf("invalid V4.1 model")
	}
	return &Backend{model: model, states: make(map[*request]bool)}, nil
}

type request struct {
	owner *Backend
	core  *State
}
type saved struct {
	owner *Backend
	core  *Checkpoint
}

func (b *Backend) NewState(context int) (dsv4.State, error) {
	if b == nil {
		return nil, dsv4.ErrClosed
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.model == nil {
		return nil, dsv4.ErrClosed
	}
	s, err := b.model.NewState(context)
	if err != nil {
		return nil, err
	}
	r := &request{owner: b, core: s}
	b.states[r] = true
	return r, nil
}

func (b *Backend) Eval(state dsv4.State, token, position int) error {
	if b == nil {
		return dsv4.ErrClosed
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	r, ok := state.(*request)
	if !ok || r == nil || r.owner != b {
		return fmt.Errorf("foreign V4.1 request")
	}
	if b.model == nil || r.core == nil {
		return dsv4.ErrClosed
	}
	if position != r.core.position || token < 0 || token >= 129280 {
		return fmt.Errorf("invalid V4.1 request position or token")
	}
	_, err := r.core.Eval(int32(token))
	return err
}

func (b *Backend) Logits(state dsv4.State) []float32 {
	if b == nil {
		return nil
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	r, ok := state.(*request)
	if !ok || r == nil || r.owner != b || b.model == nil || r.core == nil || !r.core.valid {
		return nil
	}
	return r.core.logits
}

func (b *Backend) Close() error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	for r := range b.states {
		r.core = nil
		delete(b.states, r)
	}
	b.model = nil
	return nil
}

func (r *request) lock() error {
	if r == nil || r.owner == nil {
		return dsv4.ErrClosed
	}
	r.owner.mu.RLock()
	if r.owner.model == nil || r.core == nil {
		r.owner.mu.RUnlock()
		return dsv4.ErrClosed
	}
	return nil
}

func (r *request) Reset() error {
	if err := r.lock(); err != nil {
		return err
	}
	defer r.owner.mu.RUnlock()
	r.core.Reset()
	return nil
}

func (r *request) Snapshot() (dsv4.Snapshot, error) {
	if err := r.lock(); err != nil {
		return nil, err
	}
	defer r.owner.mu.RUnlock()
	c, err := r.core.Checkpoint()
	if err != nil {
		return nil, err
	}
	return &saved{owner: r.owner, core: c}, nil
}

func (r *request) Restore(snapshot dsv4.Snapshot) error {
	if err := r.lock(); err != nil {
		return err
	}
	defer r.owner.mu.RUnlock()
	s, ok := snapshot.(*saved)
	if !ok || s == nil || s.owner != r.owner {
		return fmt.Errorf("foreign V4.1 snapshot")
	}
	if s.core == nil {
		return dsv4.ErrClosed
	}
	return r.core.Restore(s.core)
}

func (r *request) Close() error {
	if r == nil || r.owner == nil {
		return nil
	}
	r.owner.mu.Lock()
	defer r.owner.mu.Unlock()
	r.core = nil
	delete(r.owner.states, r)
	return nil
}

func (s *saved) Close() error {
	if s == nil || s.owner == nil {
		return nil
	}
	s.owner.mu.Lock()
	defer s.owner.mu.Unlock()
	s.core = nil
	return nil
}

// nativeHeader seals the raw upstream payload in a private ds4-min envelope.
// Its checksum detects corruption, not authenticity. It is not a KVC container.
type nativeHeader struct {
	Magic          [8]byte
	Asset, Payload [32]byte
	Context        uint32
	Bytes          uint64
}

var nativeMagic = [8]byte{'D', '4', '1', 'S', 'T', 'A', 'T', '1'}

func (r *request) Save(w io.Writer) error {
	if err := r.lock(); err != nil {
		return err
	}
	defer r.owner.mu.RUnlock()
	if w == nil {
		return fmt.Errorf("missing V4.1 state writer")
	}
	h := sha256.New()
	// Preflight and seal without a payload-sized temporary allocation.
	if err := r.core.WritePayload(h); err != nil {
		return err
	}
	header := nativeHeader{Magic: nativeMagic, Asset: r.owner.model.assetSHA, Context: uint32(r.core.context), Bytes: uint64(payloadBytes(r.core.position))}
	copy(header.Payload[:], h.Sum(nil))
	if err := binary.Write(w, binary.LittleEndian, header); err != nil {
		return fmt.Errorf("write V4.1 envelope: %w", err)
	}
	return r.core.WritePayload(w)
}

func (r *request) Load(reader io.Reader) error {
	if err := r.lock(); err != nil {
		return err
	}
	defer r.owner.mu.RUnlock()
	if reader == nil {
		return fmt.Errorf("missing V4.1 state reader")
	}
	var header nativeHeader
	if err := binary.Read(reader, binary.LittleEndian, &header); err != nil {
		return fmt.Errorf("read V4.1 envelope: %w", err)
	}
	model := r.owner.model
	if header.Magic != nativeMagic || model.assetSHA == ([32]byte{}) || header.Asset != model.assetSHA || header.Context != uint32(r.core.context) || header.Bytes < uint64(payloadBytes(1)) || header.Bytes > uint64(payloadBytes(r.core.context-1)) {
		return fmt.Errorf("invalid V4.1 state envelope")
	}
	next, err := model.NewState(r.core.context)
	if err != nil {
		return err
	}
	h := sha256.New()
	if err := next.restorePayload(io.TeeReader(reader, h), int64(header.Bytes), header.Asset, int(header.Context)); err != nil {
		return err
	}
	var digest [32]byte
	copy(digest[:], h.Sum(nil))
	if digest != header.Payload {
		return fmt.Errorf("V4.1 payload checksum mismatch")
	}
	r.core = next
	return nil
}
