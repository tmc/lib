package cudaspec

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/tmc/lib/nvidia/x/cudamtp"
)

// BatchDraftStep describes one admitted synchronous MTP draft-logit production
// step inside ExecuteBatchCycle.
//
// ExecuteBatchCycle validates request bounds, speculation keys, finite inputs,
// and DraftDigest before constructing a BatchDraftStep.
type BatchDraftStep struct {
	StepID       int64
	Key          SpeculationCacheKey
	DraftDigest  [32]byte
	Hidden       []float32
	DraftWeights []float32
	HeadWeights  []float32
	MTP          cudamtp.MTPConfig
}

// DraftLogitsEngine computes MTP draft logits shaped [NumDraftHeads, VocabSize]
// for an admitted BatchDraftStep.
type DraftLogitsEngine interface {
	ProduceDraftLogits(ctx context.Context, step *BatchDraftStep) ([]float32, error)
}

// DraftLogitsFunc adapts a function to DraftLogitsEngine.
type DraftLogitsFunc func(ctx context.Context, step *BatchDraftStep) ([]float32, error)

// ProduceDraftLogits calls f(ctx, step).
func (f DraftLogitsFunc) ProduceDraftLogits(ctx context.Context, step *BatchDraftStep) ([]float32, error) {
	return f(ctx, step)
}

// BatchDraftProducer configures synchronous MTP draft-logit production for
// ExecuteBatchCycle.
//
// Wait must establish that all producer work for the step is quiescent before
// logits are read or producer-owned resources are released, including when
// ProduceDraftLogits returns cancellation or failure. Wait receives no canceled
// context: cancellation alone cannot authorize releasing device buffers.
// Done optionally proves exact-step quiescence independently of a terminal
// error from Wait; a false result retains Owner (and any io.Closer error owner)
// in BatchCycleResult.DraftOwner for explicit caller recovery. A failed owner
// Close also retains a recovery owner; retries release only remaining resources
// and stop at failed prerequisites. The caller must establish quiescence before
// closing an owner retained after an incomplete step.
type BatchDraftProducer struct {
	Engine DraftLogitsEngine
	Wait   func(*BatchDraftStep) error
	Done   func(*BatchDraftStep) bool
	Owner  io.Closer
}

func validDraftProducer(p *BatchDraftProducer) bool {
	if p == nil {
		return true
	}
	if p.Engine == nil || p.Wait == nil {
		return false
	}
	if fn, ok := p.Engine.(DraftLogitsFunc); ok && fn == nil {
		return false
	}
	return true
}

func executeBatchDrafts(ctx context.Context, req BatchCycleRequest, key SpeculationCacheKey, digest [32]byte) ([]int32, *BatchCycleResult, error) {
	if len(req.Hidden) != req.MTP.Dim {
		return nil, nil, errors.New("cudaspec: hidden state length mismatch")
	}
	if req.DraftProducer == nil {
		logits, err := cudamtp.MTPReference(req.Hidden, req.DraftWeights, req.HeadWeights, req.MTP)
		if err != nil {
			return nil, nil, err
		}
		drafts, err := selectGreedyDrafts(logits, req.MTP)
		if err != nil {
			return nil, nil, err
		}
		return drafts, nil, nil
	}

	p := req.DraftProducer
	step := &BatchDraftStep{
		StepID:       1,
		Key:          key,
		DraftDigest:  digest,
		Hidden:       append([]float32(nil), req.Hidden...),
		DraftWeights: append([]float32(nil), req.DraftWeights...),
		HeadWeights:  append([]float32(nil), req.HeadWeights...),
		MTP:          req.MTP,
	}
	result := &BatchCycleResult{Key: key}
	logits, runErr := p.Engine.ProduceDraftLogits(ctx, step)
	var errCloser io.Closer
	if runErr != nil {
		_ = errors.As(runErr, &errCloser)
	}
	owner := retainDraftOwners(p.Owner, errCloser)
	closeOwners := func() error {
		if owner == nil {
			return nil
		}
		err := owner.Close()
		if err != nil {
			result.DraftOwner = owner
		}
		return err
	}
	if err := p.Wait(step); err != nil {
		quiescent := p.Done != nil && p.Done(step)
		if quiescent {
			closeErr := closeOwners()
			return nil, result, errors.Join(runErr, fmt.Errorf("cudaspec: draft producer drain: %w", err), closeErr)
		}
		result.DraftOwner = owner
		return nil, result, errors.Join(runErr, fmt.Errorf("cudaspec: draft producer drain: %w", err))
	}
	if p.Done != nil && !p.Done(step) {
		result.DraftOwner = owner
		return nil, result, errors.Join(runErr, errors.New("cudaspec: draft producer completion not established"))
	}
	closeErr := closeOwners()
	if err := ctx.Err(); err != nil {
		return nil, result, errors.Join(runErr, err, closeErr)
	}
	if runErr != nil {
		return nil, result, errors.Join(runErr, closeErr)
	}
	if closeErr != nil {
		return nil, result, closeErr
	}
	drafts, err := selectGreedyDrafts(logits, req.MTP)
	if err != nil {
		return nil, result, err
	}
	return drafts, nil, nil
}

func selectGreedyDrafts(logits []float32, c cudamtp.MTPConfig) ([]int32, error) {
	if len(logits) != c.NumDraftHeads*c.VocabSize {
		return nil, fmt.Errorf("cudaspec: draft logits length %d, want %d", len(logits), c.NumDraftHeads*c.VocabSize)
	}
	drafts := make([]int32, c.NumDraftHeads)
	for h := range drafts {
		row := logits[h*c.VocabSize : (h+1)*c.VocabSize]
		best := 0
		for j, x := range row {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return nil, errors.New("cudaspec: nonfinite MTP output")
			}
			if x > row[best] {
				best = j
			}
		}
		drafts[h] = int32(best)
	}
	return drafts, nil
}

type joinedCloser struct {
	first  io.Closer
	second io.Closer
}

// Close stops at a failed prerequisite. Successful releases are removed so a
// caller can retry the retained owner without releasing a resource twice.
func (c *joinedCloser) Close() error {
	if c.first != nil {
		if err := c.first.Close(); err != nil {
			return err
		}
		c.first = nil
	}
	if c.second != nil {
		if err := c.second.Close(); err != nil {
			return err
		}
		c.second = nil
	}
	return nil
}

func retainDraftOwners(stepOwner, errCloser io.Closer) io.Closer {
	if errCloser != nil && stepOwner != nil && errCloser != stepOwner {
		return &joinedCloser{first: errCloser, second: stepOwner}
	}
	if errCloser != nil {
		return &joinedCloser{first: errCloser}
	}
	if stepOwner != nil {
		return &joinedCloser{first: stepOwner}
	}
	return nil
}
