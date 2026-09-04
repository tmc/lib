package cudakvlease

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// PackedRequest describes one stable request row and the query positions
// packed for that request. Positions may be empty when a request has no work in
// this population.
type PackedRequest struct {
	BatchRequest
	Positions     []int
	CausalLengths []int
}

// PackedDeviceMetadata is immutable until its PackedBatch is completed.
type PackedDeviceMetadata struct {
	BatchDeviceMetadata
	TokenRows        cudamem.Slice[int32]
	TokenPositions   cudamem.Slice[int32]
	CausalLengths    cudamem.Slice[int32]
	TokenGenerations cudamem.Slice[uint64]
	TokenCount       int
}

type packedBuffers struct {
	rows        cudamem.Slice[int32]
	positions   cudamem.Slice[int32]
	lengths     cudamem.Slice[int32]
	generations cudamem.Slice[uint64]
}

// PackedMetadata owns bounded, double-buffered token metadata and the stable
// request metadata that owns its page pins.
type PackedMetadata struct {
	mu      sync.Mutex
	admitMu sync.Mutex

	requests  *BatchMetadata
	dev       *cuda.Device
	maxTokens int
	buffers   [2]packedBuffers
	active    int
	busy      bool
	closed    bool
	uploads   uint64
	bytes     uint64
}

// NewPackedMetadata allocates a reusable packed-token metadata owner.
func NewPackedMetadata(adapter *Adapter, dev *cuda.Device, maxRows, maxTokens, maxBlocks int) (*PackedMetadata, error) {
	if adapter == nil || dev == nil || maxRows <= 0 || maxTokens <= 0 || maxBlocks <= 0 {
		return nil, ErrInvalid
	}
	requests, err := NewBatchMetadata(adapter, dev, maxRows, maxBlocks)
	if err != nil {
		return nil, err
	}
	m := &PackedMetadata{requests: requests, dev: dev, maxTokens: maxTokens}
	for i := range m.buffers {
		b := &m.buffers[i]
		if b.rows, err = cudamem.Alloc[int32](dev, maxTokens); err != nil {
			m.free()
			_ = requests.Close()
			return nil, fmt.Errorf("cudakvlease: allocate packed rows: %w", err)
		}
		if b.lengths, err = cudamem.Alloc[int32](dev, maxTokens); err != nil {
			m.free()
			_ = requests.Close()
			return nil, fmt.Errorf("cudakvlease: allocate packed lengths: %w", err)
		}
		if b.positions, err = cudamem.Alloc[int32](dev, maxTokens); err != nil {
			m.free()
			_ = requests.Close()
			return nil, fmt.Errorf("cudakvlease: allocate packed positions: %w", err)
		}
		if b.generations, err = cudamem.Alloc[uint64](dev, maxTokens); err != nil {
			m.free()
			_ = requests.Close()
			return nil, fmt.Errorf("cudakvlease: allocate packed generations: %w", err)
		}
	}
	return m, nil
}

// AllocatedBytes returns the exact persistent double-buffer allocation size.
func (m *PackedMetadata) AllocatedBytes() uint64 {
	if m == nil {
		return 0
	}
	return m.requests.AllocatedBytes() + uint64(2*20*m.maxTokens)
}

// UploadStats returns completed population count and exact copied bytes,
// including stable request metadata.
func (m *PackedMetadata) UploadStats() (uint64, uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	_, requestBytes := m.requests.UploadStats()
	return m.uploads, requestBytes + m.bytes
}

// PackedBatch pins all request pages and owns one packed metadata generation.
type PackedBatch struct {
	mu          sync.Mutex
	owner       *PackedMetadata
	requests    *Batch
	metadata    PackedDeviceMetadata
	allocations map[string][]Allocation
	done        bool
}

// Allocations returns the admitted pages for key while the batch is live.
func (b *PackedBatch) Allocations(key string) ([]Allocation, error) {
	if b == nil || b.owner == nil || key == "" {
		return nil, ErrInvalid
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.done {
		return nil, ErrStale
	}
	allocations, ok := b.allocations[key]
	if !ok {
		return nil, ErrInvalid
	}
	return append([]Allocation(nil), allocations...), nil
}

// Device returns metadata valid until Complete returns successfully.
func (b *PackedBatch) Device() (PackedDeviceMetadata, error) {
	if b == nil || b.owner == nil {
		return PackedDeviceMetadata{}, ErrInvalid
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.done {
		return PackedDeviceMetadata{}, ErrStale
	}
	return b.metadata, nil
}

// Populate atomically publishes stable request and packed-token metadata.
func (m *PackedMetadata) Populate(ctx context.Context, requests []PackedRequest) (*PackedBatch, error) {
	if m == nil || ctx == nil || len(requests) == 0 {
		return nil, ErrInvalid
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, ErrClosed
	}
	if m.busy {
		return nil, errors.New("cudakvlease: packed metadata busy")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	plain := make([]BatchRequest, len(requests))
	for i := range requests {
		plain[i] = requests[i].BatchRequest
	}
	base, err := m.requests.Populate(ctx, plain)
	if err != nil {
		return nil, err
	}
	release := func() { _ = base.Complete(context.Background(), immediateCompletion{}) }
	baseDevice, err := base.Device()
	if err != nil {
		release()
		return nil, err
	}
	rows := make([]int, len(requests))
	for i := range requests {
		rows[i], err = base.Row(requests[i].Key)
		if err != nil {
			release()
			return nil, err
		}
	}
	tokenRows, tokenPositions, causalLengths, generations, err := buildPackedLayout(requests, rows, m.maxTokens)
	if err != nil {
		release()
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		release()
		return nil, err
	}
	next := 1 - m.active
	b := &m.buffers[next]
	err = m.dev.WithContext(func() error {
		if err := b.rows.Slice(0, len(tokenRows)).CopyFromHost(tokenRows); err != nil {
			return err
		}
		if err := b.positions.Slice(0, len(tokenPositions)).CopyFromHost(tokenPositions); err != nil {
			return err
		}
		if err := b.lengths.Slice(0, len(causalLengths)).CopyFromHost(causalLengths); err != nil {
			return err
		}
		return b.generations.Slice(0, len(generations)).CopyFromHost(generations)
	})
	if err != nil {
		release()
		return nil, fmt.Errorf("cudakvlease: populate packed metadata: %w", err)
	}
	if err := ctx.Err(); err != nil {
		release()
		return nil, err
	}
	m.active, m.busy = next, true
	m.uploads++
	m.bytes += uint64(20 * len(tokenRows))
	return &PackedBatch{owner: m, requests: base, metadata: PackedDeviceMetadata{
		BatchDeviceMetadata: baseDevice,
		TokenRows:           b.rows, TokenPositions: b.positions, CausalLengths: b.lengths, TokenGenerations: b.generations,
		TokenCount: len(tokenRows),
	}}, nil
}

func buildPackedLayout(requests []PackedRequest, rows []int, maxTokens int) ([]int32, []int32, []int32, []uint64, error) {
	if len(requests) != len(rows) || maxTokens <= 0 {
		return nil, nil, nil, nil, ErrInvalid
	}
	var tokenRows, positions, lengths []int32
	var generations []uint64
	for i, request := range requests {
		if rows[i] < 0 || request.Generation == 0 || request.Length <= 0 {
			return nil, nil, nil, nil, ErrInvalid
		}
		if len(request.CausalLengths) != 0 && len(request.CausalLengths) != len(request.Positions) {
			return nil, nil, nil, nil, ErrInvalid
		}
		previous := -1
		for j, position := range request.Positions {
			if position < 0 || position >= request.Length || position <= previous || len(tokenRows) == maxTokens {
				return nil, nil, nil, nil, ErrInvalid
			}
			causalLength := position + 1
			if len(request.CausalLengths) != 0 {
				causalLength = request.CausalLengths[j]
			}
			if causalLength <= 0 || causalLength > request.Length {
				return nil, nil, nil, nil, ErrInvalid
			}
			previous = position
			tokenRows = append(tokenRows, int32(rows[i]))
			positions = append(positions, int32(position))
			lengths = append(lengths, int32(causalLength))
			generations = append(generations, request.Generation)
		}
	}
	if len(tokenRows) == 0 {
		return nil, nil, nil, nil, ErrInvalid
	}
	return tokenRows, positions, lengths, generations, nil
}

// Complete waits once for all device consumers, then releases page pins and
// permits both metadata generations to be reused.
func (b *PackedBatch) Complete(ctx context.Context, completion Completion) error {
	if b == nil || b.owner == nil || ctx == nil || completion == nil {
		return ErrInvalid
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.done {
		return ErrStale
	}
	if err := completion.Synchronize(ctx); err != nil {
		return err
	}
	if err := b.requests.Complete(ctx, immediateCompletion{}); err != nil {
		return err
	}
	b.owner.mu.Lock()
	b.owner.busy = false
	b.owner.mu.Unlock()
	b.done = true
	return nil
}

// Close frees both metadata generations. It refuses while a batch is live.
func (m *PackedMetadata) Close() error {
	if m == nil {
		return nil
	}
	m.admitMu.Lock()
	defer m.admitMu.Unlock()
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	if m.busy {
		return errors.New("cudakvlease: packed metadata busy")
	}
	err := errors.Join(m.requests.Close(), m.free())
	if err == nil {
		m.closed = true
	}
	return err
}

func (m *PackedMetadata) free() error {
	var errs []error
	for i := range m.buffers {
		b := &m.buffers[i]
		for _, free := range []func() error{b.rows.Free, b.positions.Free, b.lengths.Free, b.generations.Free} {
			if err := free(); err != nil {
				errs = append(errs, err)
			}
		}
		b.rows = cudamem.Slice[int32]{}
		b.positions = cudamem.Slice[int32]{}
		b.lengths = cudamem.Slice[int32]{}
		b.generations = cudamem.Slice[uint64]{}
	}
	return errors.Join(errs...)
}
