package cudakvlease

import (
	"context"
	"errors"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalayer"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// DecoderLayer describes attention followed by an output projection, residual
// addition, and RMS normalization.
type DecoderLayer struct {
	Attention      CausalAttention
	ProjectionNorm driver.Function
	Dim            int
	Threads        int
	Epsilon        float32
}

// RunDecoderLayer executes one packed decoder-layer transaction. It enqueues
// attention and the adjacent projection, residual, and normalization stage on
// one stream and synchronizes once. Returned status is indexed by stable
// request row. The batch remains live; the caller retains completion ownership.
func (b *PackedBatch) RunDecoderLayer(ctx context.Context, stream *cuda.Stream, op DecoderLayer,
	q, attentionOut, residual, projection, normWeight, output, normalized cudamem.Slice[float32]) ([]int32, error) {
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
	config := cudalayer.ProjectionNormConfig{Dim: op.Dim, Threads: op.Threads, Epsilon: op.Epsilon}
	shared, err := config.SharedMemoryBytes()
	if err != nil || op.ProjectionNorm == 0 || op.Dim != op.Attention.QHeads*op.Attention.HeadDim {
		return nil, ErrInvalid
	}
	tokens := b.metadata.TokenCount
	want := tokens * op.Dim
	if attentionOut.Len() != want || residual.Len() != want || projection.Len() != op.Dim*op.Dim ||
		normWeight.Len() != op.Dim || output.Len() != want || normalized.Len() != want {
		return nil, ErrInvalid
	}
	owner := b.owner
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
		if err := b.launchCausalLocked(stream, op.Attention, q, attentionOut, deviceStatus); err != nil {
			return err
		}
		m := b.metadata
		if err := cudalaunch.Kernel(op.ProjectionNorm).
			Grid(uint32(tokens)).Block(uint32(op.Threads)).SharedMem(uint32(shared)).On(stream).
			Launch(attentionOut, residual, projection, normWeight, m.TokenRows, deviceStatus,
				output, normalized, int32(op.Dim), int32(tokens), int32(rows), op.Epsilon); err != nil {
			return err
		}
		return deviceStatus.CopyToHostAsync(status, stream)
	})
	if err != nil {
		// A preceding operation may already be queued. Retain pages and temporary
		// storage until every submitted consumer is terminal.
		syncErr := stream.Synchronize(context.Background())
		freeErr := owner.dev.WithContext(deviceStatus.Free)
		return nil, errors.Join(err, syncErr, freeErr)
	}
	syncErr := stream.Synchronize(ctx)
	if syncErr != nil {
		// Cancellation stops the wait, not CUDA execution.
		syncErr = errors.Join(syncErr, stream.Synchronize(context.Background()))
	}
	freeErr := owner.dev.WithContext(deviceStatus.Free)
	if err := errors.Join(syncErr, freeErr); err != nil {
		return nil, err
	}
	return status, nil
}
