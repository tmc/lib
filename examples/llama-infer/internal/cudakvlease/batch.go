package cudakvlease

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// BatchRequest describes one stable request row in a metadata population.
type BatchRequest struct {
	Key         string
	Generation  uint64
	Allocations []Allocation
	Length      int
	Position    int
}

// BatchDeviceMetadata is one immutable device-visible metadata generation.
type BatchDeviceMetadata struct {
	BlockTables    cudamem.Slice[int32]
	Lengths        cudamem.Slice[int32]
	Positions      cudamem.Slice[int32]
	Rows           cudamem.Slice[int32]
	TokenPositions cudamem.Slice[int32]
	Generations    cudamem.Slice[uint64]
	Count          int
	MaxBlocks      int
}

type batchBuffers struct {
	tables         cudamem.Slice[int32]
	lengths        cudamem.Slice[int32]
	positions      cudamem.Slice[int32]
	rows           cudamem.Slice[int32]
	tokenPositions cudamem.Slice[int32]
	generations    cudamem.Slice[uint64]
}

// BatchMetadata owns two bounded device metadata generations. Populate writes
// the inactive generation completely before publishing it.
type BatchMetadata struct {
	mu sync.Mutex

	adapter   *Adapter
	dev       *cuda.Device
	maxRows   int
	maxBlocks int
	buffers   [2]batchBuffers
	active    int
	rows      map[string]int
	rowGen    map[string]uint64
	busy      bool
	closed    bool
	uploads   uint64
	bytes     uint64
}

// NewBatchMetadata allocates a reusable request-indexed metadata owner.
func NewBatchMetadata(adapter *Adapter, dev *cuda.Device, maxRows, maxBlocks int) (*BatchMetadata, error) {
	if adapter == nil || dev == nil || maxRows <= 0 || maxBlocks <= 0 {
		return nil, ErrInvalid
	}
	m := &BatchMetadata{adapter: adapter, dev: dev, maxRows: maxRows, maxBlocks: maxBlocks, rows: make(map[string]int), rowGen: make(map[string]uint64)}
	for i := range m.buffers {
		b := &m.buffers[i]
		var err error
		if b.tables, err = cudamem.Alloc[int32](dev, maxRows*maxBlocks); err != nil {
			m.free()
			return nil, fmt.Errorf("cudakvlease: allocate batch tables: %w", err)
		}
		if b.lengths, err = cudamem.Alloc[int32](dev, maxRows); err != nil {
			m.free()
			return nil, fmt.Errorf("cudakvlease: allocate batch lengths: %w", err)
		}
		if b.positions, err = cudamem.Alloc[int32](dev, maxRows); err != nil {
			m.free()
			return nil, fmt.Errorf("cudakvlease: allocate batch positions: %w", err)
		}
		if b.rows, err = cudamem.Alloc[int32](dev, maxRows); err != nil {
			m.free()
			return nil, fmt.Errorf("cudakvlease: allocate batch rows: %w", err)
		}
		if b.tokenPositions, err = cudamem.Alloc[int32](dev, maxRows); err != nil {
			m.free()
			return nil, fmt.Errorf("cudakvlease: allocate batch token positions: %w", err)
		}
		if b.generations, err = cudamem.Alloc[uint64](dev, maxRows); err != nil {
			m.free()
			return nil, fmt.Errorf("cudakvlease: allocate batch generations: %w", err)
		}
	}
	return m, nil
}

// AllocatedBytes returns the exact persistent double-buffer allocation size.
func (m *BatchMetadata) AllocatedBytes() uint64 {
	return uint64(2 * (4*m.maxRows*m.maxBlocks + 24*m.maxRows))
}

// UploadStats returns completed population count and exact copied bytes.
func (m *BatchMetadata) UploadStats() (uint64, uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.uploads, m.bytes
}

// Batch pins every page named by one published metadata generation.
type Batch struct {
	mu       sync.Mutex
	owner    *BatchMetadata
	metadata BatchDeviceMetadata
	uses     []*Use
	rows     map[string]int
	done     bool
}

// Row returns the stable row assigned to key in this generation.
func (b *Batch) Row(key string) (int, error) {
	if b == nil || b.owner == nil || key == "" {
		return 0, ErrInvalid
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.done {
		return 0, ErrStale
	}
	row, ok := b.rows[key]
	if !ok {
		return 0, ErrInvalid
	}
	return row, nil
}

// Device returns immutable metadata valid until Complete returns.
func (b *Batch) Device() (BatchDeviceMetadata, error) {
	if b == nil || b.owner == nil {
		return BatchDeviceMetadata{}, ErrInvalid
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.done {
		return BatchDeviceMetadata{}, ErrStale
	}
	return b.metadata, nil
}

// Populate atomically publishes requests and pins all referenced pages.
func (m *BatchMetadata) Populate(ctx context.Context, requests []BatchRequest) (*Batch, error) {
	if ctx == nil || len(requests) == 0 || len(requests) > m.maxRows {
		return nil, ErrInvalid
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, ErrClosed
	}
	if m.busy {
		return nil, errors.New("cudakvlease: batch metadata busy")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	rows, nextRows, nextGen, err := m.assignRows(requests)
	if err != nil {
		return nil, err
	}
	tables := make([]int32, m.maxRows*m.maxBlocks)
	for i := range tables {
		tables[i] = -1
	}
	lengths := make([]int32, m.maxRows)
	positions := make([]int32, m.maxRows)
	rowData := make([]int32, m.maxRows)
	tokenPositions := make([]int32, m.maxRows)
	generations := make([]uint64, m.maxRows)
	uses := make([]*Use, 0)
	release := func() {
		for _, use := range uses {
			_ = use.Complete(context.Background(), immediateCompletion{})
		}
	}
	for i, request := range requests {
		if request.Key == "" || request.Generation == 0 || request.Length <= 0 || request.Position < 0 || len(request.Allocations) == 0 || len(request.Allocations) > m.maxBlocks {
			release()
			return nil, ErrInvalid
		}
		row := rows[i]
		lengths[row], positions[row], rowData[i], generations[row] = int32(request.Length), int32(request.Position), int32(row), request.Generation
		tokenPositions[i] = int32(request.Position)
		for block, allocation := range request.Allocations {
			id, generation := allocation.ID()
			use, err := m.adapter.Bind(id, generation)
			if err != nil {
				release()
				return nil, fmt.Errorf("cudakvlease: bind batch row %d block %d: %w", row, block, err)
			}
			slot, err := use.Slot()
			if err != nil {
				uses = append(uses, use)
				release()
				return nil, err
			}
			uses = append(uses, use)
			tables[row*m.maxBlocks+block] = int32(slot)
		}
	}
	if err := ctx.Err(); err != nil {
		release()
		return nil, err
	}
	next := 1 - m.active
	b := &m.buffers[next]
	err = m.dev.WithContext(func() error {
		for _, copy := range []func() error{
			func() error { return b.tables.CopyFromHost(tables) },
			func() error { return b.lengths.CopyFromHost(lengths) },
			func() error { return b.positions.CopyFromHost(positions) },
			func() error { return b.rows.CopyFromHost(rowData) },
			func() error { return b.tokenPositions.CopyFromHost(tokenPositions) },
			func() error { return b.generations.CopyFromHost(generations) },
		} {
			if err := copy(); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		release()
		return nil, fmt.Errorf("cudakvlease: populate batch metadata: %w", err)
	}
	m.active, m.busy = next, true
	m.rows, m.rowGen = nextRows, nextGen
	uploadBytes := uint64(4*m.maxRows*m.maxBlocks + 24*m.maxRows)
	m.uploads++
	m.bytes += uploadBytes
	return &Batch{owner: m, uses: uses, rows: nextRows, metadata: BatchDeviceMetadata{
		BlockTables: b.tables, Lengths: b.lengths, Positions: b.positions,
		Rows: b.rows, TokenPositions: b.tokenPositions, Generations: b.generations, Count: len(requests), MaxBlocks: m.maxBlocks,
	}}, nil
}

func (m *BatchMetadata) assignRows(requests []BatchRequest) ([]int, map[string]int, map[string]uint64, error) {
	rows := make([]int, len(requests))
	used := make(map[int]bool, len(requests))
	keys := make(map[string]bool, len(requests))
	for i, request := range requests {
		if request.Key == "" || keys[request.Key] {
			return nil, nil, nil, ErrInvalid
		}
		keys[request.Key] = true
		if row, ok := m.rows[request.Key]; ok {
			if request.Generation < m.rowGen[request.Key] {
				return nil, nil, nil, ErrStale
			}
			rows[i], used[row] = row, true
			continue
		}
		row := 0
		for row < m.maxRows && used[row] {
			row++
		}
		if row == m.maxRows {
			return nil, nil, nil, ErrInvalid
		}
		rows[i], used[row] = row, true
	}
	nextRows := make(map[string]int, len(requests))
	nextGen := make(map[string]uint64, len(requests))
	for i, request := range requests {
		nextRows[request.Key], nextGen[request.Key] = rows[i], request.Generation
	}
	return rows, nextRows, nextGen, nil
}

type immediateCompletion struct{}

func (immediateCompletion) Synchronize(context.Context) error { return nil }

// Complete waits for all device consumers before releasing page pins and reuse.
func (b *Batch) Complete(ctx context.Context, completion Completion) error {
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
	for _, use := range b.uses {
		if err := use.Complete(ctx, immediateCompletion{}); err != nil {
			return err
		}
	}
	b.owner.mu.Lock()
	b.owner.busy = false
	b.owner.mu.Unlock()
	b.done = true
	return nil
}

// Close frees both generations. It refuses while a Batch is live.
func (m *BatchMetadata) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	if m.busy {
		return errors.New("cudakvlease: batch metadata busy")
	}
	err := m.free()
	if err == nil {
		m.closed = true
	}
	return err
}

func (m *BatchMetadata) free() error {
	var errs []error
	for i := range m.buffers {
		b := &m.buffers[i]
		if err := b.tables.Free(); err != nil {
			errs = append(errs, err)
		} else {
			b.tables = cudamem.Slice[int32]{}
		}
		if err := b.lengths.Free(); err != nil {
			errs = append(errs, err)
		} else {
			b.lengths = cudamem.Slice[int32]{}
		}
		if err := b.positions.Free(); err != nil {
			errs = append(errs, err)
		} else {
			b.positions = cudamem.Slice[int32]{}
		}
		if err := b.rows.Free(); err != nil {
			errs = append(errs, err)
		} else {
			b.rows = cudamem.Slice[int32]{}
		}
		if err := b.tokenPositions.Free(); err != nil {
			errs = append(errs, err)
		} else {
			b.tokenPositions = cudamem.Slice[int32]{}
		}
		if err := b.generations.Free(); err != nil {
			errs = append(errs, err)
		} else {
			b.generations = cudamem.Slice[uint64]{}
		}
	}
	return errors.Join(errs...)
}
