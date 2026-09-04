package cudakvlease

import (
	"context"
	"errors"
	"fmt"

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

// PackedAttentionBlock describes pre-attention projection, paged KV update,
// attention, and post-attention projection. It does not include an MLP block.
type PackedAttentionBlock struct {
	DecoderLayer
	RMSNormQKV driver.Function
	RoPEStore  driver.Function
	RoPETheta  float32
}

// PackedDecoderLayer describes every operation in one Llama decoder layer.
type PackedDecoderLayer struct {
	PackedAttentionBlock
	GateUpSiLU      driver.Function
	DownResidual    driver.Function
	IntermediateDim int
}

type packedMLP struct {
	op                           PackedDecoderLayer
	gateWeight, upWeight         cudamem.Slice[float32]
	downWeight                   cudamem.Slice[float32]
	gate, up, activation, output cudamem.Slice[float32]
}

// RunPackedAttentionBlock executes one packed attention block on one stream.
// The weight matrices are shared by every token row. Scratch storage and the
// batch's page pins must remain live until this call returns; batch completion
// remains the caller's responsibility.
func (b *PackedBatch) RunPackedAttentionBlock(ctx context.Context, stream *cuda.Stream, op PackedAttentionBlock,
	input, inputNorm, inputNormWeight, qWeight, kWeight, vWeight, q, k, v, qRoPE, attentionOut,
	projection, outputNormWeight, output, outputNorm cudamem.Slice[float32]) ([]int32, error) {
	return b.runPackedDecoderLayer(ctx, stream, op, input, inputNorm, inputNormWeight, qWeight, kWeight, vWeight,
		q, k, v, qRoPE, attentionOut, projection, outputNormWeight, output, outputNorm, nil)
}

// RunPackedDecoderLayer executes one complete Llama decoder layer on one
// stream. Weight matrices are shared across token rows. Scratch storage and
// page pins remain live through the sole terminal synchronization; the caller
// retains batch completion ownership.
func (b *PackedBatch) RunPackedDecoderLayer(ctx context.Context, stream *cuda.Stream, op PackedDecoderLayer,
	input, inputNorm, inputNormWeight, qWeight, kWeight, vWeight, q, k, v, qRoPE, attentionOut,
	projection, outputNormWeight, attentionResidual, outputNorm, gateWeight, upWeight, downWeight,
	gate, up, activation, output cudamem.Slice[float32]) ([]int32, error) {
	mlp := &packedMLP{op: op, gateWeight: gateWeight, upWeight: upWeight, downWeight: downWeight,
		gate: gate, up: up, activation: activation, output: output}
	return b.runPackedDecoderLayer(ctx, stream, op.PackedAttentionBlock, input, inputNorm, inputNormWeight,
		qWeight, kWeight, vWeight, q, k, v, qRoPE, attentionOut, projection, outputNormWeight,
		attentionResidual, outputNorm, mlp)
}

func (b *PackedBatch) runPackedDecoderLayer(ctx context.Context, stream *cuda.Stream, op PackedAttentionBlock,
	input, inputNorm, inputNormWeight, qWeight, kWeight, vWeight, q, k, v, qRoPE, attentionOut,
	projection, outputNormWeight, attentionResidual, outputNorm cudamem.Slice[float32], mlp *packedMLP) ([]int32, error) {
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
	attention := op.Attention
	dim := op.Dim
	qDim := attention.QHeads * attention.HeadDim
	kvDim := attention.KVHeads * attention.HeadDim
	tokens := b.metadata.TokenCount
	wantInput, inputOK := product(tokens, dim)
	wantQ, qOK := product(tokens, qDim)
	wantKV, kvOK := product(tokens, kvDim)
	qWeights, qwOK := product(qDim, dim)
	kvWeights, kvwOK := product(kvDim, dim)
	projectionLen, projectionOK := product(dim, dim)
	mlpOK := true
	if mlp != nil {
		intermediate := mlp.op.IntermediateDim
		wantIntermediate, intermediateOK := product(tokens, intermediate)
		intermediateWeights, weightOK := product(intermediate, dim)
		downWeights, downOK := product(dim, intermediate)
		mlpOK = intermediateOK && weightOK && downOK && mlp.op.GateUpSiLU != 0 && mlp.op.DownResidual != 0 &&
			mlp.gateWeight.Len() == intermediateWeights && mlp.upWeight.Len() == intermediateWeights &&
			mlp.downWeight.Len() == downWeights && mlp.gate.Len() == wantIntermediate &&
			mlp.up.Len() == wantIntermediate && mlp.activation.Len() == wantIntermediate && mlp.output.Len() == wantInput
	}
	config := cudalayer.ProjectionNormConfig{Dim: dim, Threads: op.Threads, Epsilon: op.Epsilon}
	shared, sharedErr := config.SharedMemoryBytes()
	if sharedErr != nil || !inputOK || !qOK || !kvOK || !qwOK || !kvwOK || !projectionOK ||
		op.RMSNormQKV == 0 || op.RoPEStore == 0 || op.ProjectionNorm == 0 || !(op.RoPETheta > 0) || !mlpOK ||
		dim != qDim || input.Len() != wantInput || inputNorm.Len() != wantInput || inputNormWeight.Len() != dim ||
		qWeight.Len() != qWeights || kWeight.Len() != kvWeights || vWeight.Len() != kvWeights ||
		q.Len() != wantQ || k.Len() != wantKV || v.Len() != wantKV || qRoPE.Len() != wantQ || attentionOut.Len() != wantQ ||
		projection.Len() != projectionLen || outputNormWeight.Len() != dim || attentionResidual.Len() != wantInput || outputNorm.Len() != wantInput {
		return nil, ErrInvalid
	}
	owner := b.owner
	adapter := owner.requests.adapter
	pages, ok := adapter.pages.(*cudaPages)
	if !ok || pages.dev != owner.dev || stream.Device() != owner.dev || attention.Layer < 0 || attention.Layer >= adapter.config.NumLayers {
		return nil, ErrInvalid
	}
	page, err := pages.Page(attention.Layer, 0)
	if err != nil {
		return nil, fmt.Errorf("cudakvlease: resolve decoder arena: %w", err)
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
		if err := cudalaunch.Kernel(op.RMSNormQKV).Grid(uint32(tokens)).Block(uint32(op.Threads)).SharedMem(uint32(shared)).On(stream).
			Launch(input, inputNormWeight, qWeight, kWeight, vWeight, m.TokenRows, m.Generations, m.TokenGenerations,
				deviceStatus, inputNorm, q, k, v, int32(dim), int32(qDim), int32(kvDim), int32(tokens), int32(rows), op.Epsilon); err != nil {
			return err
		}
		width := max(qDim, kvDim)
		blocks := (width + op.Threads - 1) / op.Threads
		if err := cudalaunch.Kernel(op.RoPEStore).Grid(uint32(blocks), uint32(tokens)).Block(uint32(op.Threads)).On(stream).
			Launch(q, k, v, qRoPE, page.K, page.V, m.BlockTables, m.Lengths, m.Generations, m.TokenRows,
				m.TokenPositions, m.TokenGenerations, int32(tokens), int32(rows), int32(m.MaxBlocks), int32(adapter.config.BlockSize),
				int32(attention.QHeads), int32(attention.KVHeads), int32(attention.HeadDim), int32(adapter.config.TotalPages), op.RoPETheta, deviceStatus); err != nil {
			return err
		}
		if err := b.launchCausalLocked(stream, attention, qRoPE, attentionOut, deviceStatus); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(op.ProjectionNorm).Grid(uint32(tokens)).Block(uint32(op.Threads)).SharedMem(uint32(shared)).On(stream).
			Launch(attentionOut, input, projection, outputNormWeight, m.TokenRows, deviceStatus,
				attentionResidual, outputNorm, int32(dim), int32(tokens), int32(rows), op.Epsilon); err != nil {
			return err
		}
		if mlp != nil {
			intermediate := mlp.op.IntermediateDim
			if err := cudalaunch.Kernel(mlp.op.GateUpSiLU).Grid(uint32(tokens)).Block(uint32(op.Threads)).On(stream).
				Launch(outputNorm, mlp.gateWeight, mlp.upWeight, m.TokenRows, deviceStatus, mlp.gate, mlp.up,
					mlp.activation, int32(dim), int32(intermediate), int32(tokens), int32(rows)); err != nil {
				return err
			}
			if err := cudalaunch.Kernel(mlp.op.DownResidual).Grid(uint32(tokens)).Block(uint32(op.Threads)).On(stream).
				Launch(mlp.activation, attentionResidual, mlp.downWeight, m.TokenRows, deviceStatus, mlp.output,
					int32(dim), int32(intermediate), int32(tokens), int32(rows)); err != nil {
				return err
			}
		}
		return deviceStatus.CopyToHostAsync(status, stream)
	})
	if err != nil {
		syncErr := stream.Synchronize(context.Background())
		freeErr := owner.dev.WithContext(deviceStatus.Free)
		return nil, errors.Join(err, syncErr, freeErr)
	}
	syncErr := stream.Synchronize(ctx)
	if syncErr != nil {
		syncErr = errors.Join(syncErr, stream.Synchronize(context.Background()))
	}
	freeErr := owner.dev.WithContext(deviceStatus.Free)
	if err := errors.Join(syncErr, freeErr); err != nil {
		return nil, err
	}
	return status, nil
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
	want, ok := product(tokens, op.Dim)
	projectionLen, projectionOK := product(op.Dim, op.Dim)
	if !ok || !projectionOK || attentionOut.Len() != want || residual.Len() != want || projection.Len() != projectionLen ||
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

func product(a, b int) (int, bool) {
	if a <= 0 || b <= 0 || a > int(^uint(0)>>1)/b {
		return 0, false
	}
	return a * b, true
}
