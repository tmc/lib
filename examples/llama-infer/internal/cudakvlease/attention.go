package cudakvlease

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// CausalAttention describes one packed paged-attention dispatch.
type CausalAttention struct {
	Kernel    driver.Function
	Layer     int
	QHeads    int
	KVHeads   int
	HeadDim   int
	MaxSeqLen int
	Threads   int
	Scale     float32
}

// AttendCausal submits and synchronizes packed paged causal attention. The
// returned status is indexed by stable request row. AttendCausal does not
// complete the batch; the caller retains responsibility for publication,
// rollback, and Complete.
func (b *PackedBatch) AttendCausal(ctx context.Context, stream *cuda.Stream, op CausalAttention, q, out cudamem.Slice[float32]) ([]int32, error) {
	if b == nil || b.owner == nil || ctx == nil || stream == nil {
		return nil, ErrInvalid
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.done {
		return nil, ErrStale
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	owner := b.owner
	adapter := owner.requests.adapter
	if stream.Device() != owner.dev || op.Layer < 0 || op.Layer >= adapter.config.NumLayers ||
		op.KVHeads != adapter.config.KVHeads || op.HeadDim != adapter.config.HeadDim ||
		op.MaxSeqLen <= 0 || op.Threads <= 0 || op.Threads > 1024 || op.Threads&(op.Threads-1) != 0 {
		return nil, ErrInvalid
	}
	config := cudaattn.Config{NumQHeads: op.QHeads, NumKVHeads: op.KVHeads, HeadDim: op.HeadDim, Scale: op.Scale}
	if config.Validate() != nil {
		return nil, ErrInvalid
	}
	want := b.metadata.TokenCount * op.QHeads * op.HeadDim
	if q.Len() != want || out.Len() != want {
		return nil, ErrInvalid
	}
	pages, ok := adapter.pages.(*cudaPages)
	if !ok || pages.dev != owner.dev {
		return nil, ErrNoView
	}
	page, err := pages.Page(op.Layer, 0)
	if err != nil {
		return nil, fmt.Errorf("cudakvlease: resolve attention arena: %w", err)
	}
	shared, err := cudaattn.PagedSharedMemoryBytes(op.MaxSeqLen, op.Threads)
	if err != nil {
		return nil, ErrInvalid
	}
	rows := owner.requests.maxRows
	status := make([]int32, rows)
	var deviceStatus cudamem.Slice[int32]
	err = owner.dev.WithContext(func() error {
		var err error
		deviceStatus, err = cudamem.Alloc[int32](owner.dev, rows)
		if err != nil {
			return err
		}
		if err := deviceStatus.CopyFromHostAsync(status, stream); err != nil {
			return err
		}
		m := b.metadata
		if err := cudalaunch.Kernel(op.Kernel).
			Grid(uint32(op.QHeads), uint32(m.TokenCount)).
			Block(uint32(op.Threads)).SharedMem(uint32(shared)).On(stream).
			Launch(q, page.K, page.V, m.BlockTables, m.Lengths, m.Generations,
				m.TokenRows, m.TokenPositions, m.CausalLengths, m.TokenGenerations, out,
				int32(m.MaxBlocks), int32(adapter.config.BlockSize), int32(op.QHeads),
				int32(op.KVHeads), int32(op.HeadDim), op.Scale, int32(m.TokenCount),
				int32(rows), int32(adapter.config.TotalPages), int32(op.MaxSeqLen), deviceStatus); err != nil {
			return err
		}
		return deviceStatus.CopyToHostAsync(status, stream)
	})
	if err != nil {
		// The status clear may already be queued. Keep pins and temporary memory
		// alive until every submitted operation is terminal.
		syncErr := stream.Synchronize(context.Background())
		freeErr := owner.dev.WithContext(deviceStatus.Free)
		return nil, errors.Join(err, syncErr, freeErr)
	}
	syncErr := stream.Synchronize(ctx)
	if syncErr != nil {
		// Context cancellation does not cancel CUDA work. Finish waiting before
		// releasing the batch lock or its arena pins.
		syncErr = errors.Join(syncErr, stream.Synchronize(context.Background()))
	}
	freeErr := owner.dev.WithContext(deviceStatus.Free)
	if err := errors.Join(syncErr, freeErr); err != nil {
		return nil, err
	}
	return status, nil
}
