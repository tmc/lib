package cudakv

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

var (
	// ErrHostCommandBusy reports a batch refused because another batch on the
	// same consumer is in progress. Nothing was removed from the ring.
	ErrHostCommandBusy = errors.New("cudakv: host command consumer busy")

	// ErrHostCommandUnsupported reports a requested mode the host consumer
	// does not provide. No such request falls back to host execution.
	ErrHostCommandUnsupported = errors.New("cudakv: host command capability unsupported")

	// ErrHostCommandInvalid reports a descriptor removed from the ring and
	// refused without running: an unknown SeqID, nonzero Flags, or a
	// PromptToken outside [0, VocabSize).
	ErrHostCommandInvalid = errors.New("cudakv: invalid host command descriptor")
)

// Host command limits. They bound the copied registry and each batch before
// any allocation or callback.
const (
	maxHostCommandEntries = 1024
	maxHostCommandWords   = 1 << 20
	maxHostCommandBatch   = 1024
)

// HostCommandConfig bounds a copied command registry and each batch.
//
// Mode must be "host-finite"; any other value, including a GPU polling,
// GIN or decode mode, is refused with [ErrHostCommandUnsupported].
// Generation must equal the ring's generation. VocabSize bounds
// PromptToken. MaxEntries and MaxBatch must be in [1, 1024];
// MaxRegistryWords and MaxBatchWords must be in [1, 1<<20].
type HostCommandConfig struct {
	Mode             string
	Generation       uint64
	VocabSize        int32
	MaxEntries       int
	MaxRegistryWords int
	MaxBatch         int
	MaxBatchWords    int
}

// HostCommandInput binds a nonzero SeqID to application input words.
// SeqID is an application identity, not a device publication sequence.
type HostCommandInput struct {
	SeqID uint64
	Words []uint32
}

// HostCommandBatch reports one call to [HostCommandConsumer.ConsumeBatch].
// Consumed counts descriptors removed from the ring, Completed counts callbacks
// that returned nil, and Words counts input words passed to callbacks,
// including a callback that failed. A removed descriptor is never requeued.
type HostCommandBatch struct {
	Consumed  int
	Completed int
	Words     int
}

// A HostCommandConsumer drains a [DeviceCommandRing] in bounded synchronous
// batches, passing each descriptor and its registered input to a caller
// function on the calling goroutine.
//
// It is host batching only. It starts no goroutines, provides no device
// executor, and does not poll device-visible memory. Whatever the callback
// runs, a successful batch is not evidence of persistent GPU execution,
// model decoding or network transfer.
//
// The consumer borrows the ring and never closes it. The zero value is
// unusable; use [NewHostCommandConsumer].
type HostCommandConsumer struct {
	mu     sync.Mutex // held for one batch
	ring   *DeviceCommandRing
	config HostCommandConfig
	inputs map[uint64][]uint32
}

// NewHostCommandConsumer returns a consumer for ring with a copied registry
// of inputs. It checks every limit before allocating; later changes to
// inputs do not affect the consumer. Each entry must have a distinct nonzero
// SeqID and between 1 and MaxBatchWords words, and the entries together at
// most MaxRegistryWords words.
func NewHostCommandConsumer(ring *DeviceCommandRing, config HostCommandConfig, inputs []HostCommandInput) (*HostCommandConsumer, error) {
	if config.Mode != "host-finite" {
		return nil, fmt.Errorf("%w: mode %q", ErrHostCommandUnsupported, config.Mode)
	}
	if ring == nil {
		return nil, errors.New("cudakv: nil command ring")
	}
	if config.VocabSize <= 0 ||
		config.MaxEntries <= 0 || config.MaxEntries > maxHostCommandEntries ||
		config.MaxRegistryWords <= 0 || config.MaxRegistryWords > maxHostCommandWords ||
		config.MaxBatch <= 0 || config.MaxBatch > maxHostCommandBatch ||
		config.MaxBatchWords <= 0 || config.MaxBatchWords > maxHostCommandWords {
		return nil, errors.New("cudakv: invalid host command limits")
	}
	ring.mu.Lock()
	generation := ring.generation
	ring.mu.Unlock()
	if generation == 0 || config.Generation != generation {
		return nil, fmt.Errorf("cudakv: host command generation %d does not match ring generation %d", config.Generation, generation)
	}
	if len(inputs) == 0 || len(inputs) > config.MaxEntries {
		return nil, fmt.Errorf("cudakv: %d host command inputs outside [1, %d]", len(inputs), config.MaxEntries)
	}
	words := 0
	for i, in := range inputs {
		if in.SeqID == 0 {
			return nil, errors.New("cudakv: host command sequence must be nonzero")
		}
		if len(in.Words) == 0 || len(in.Words) > config.MaxBatchWords || len(in.Words) > config.MaxRegistryWords-words {
			return nil, fmt.Errorf("cudakv: host command sequence %d input exceeds word budget", in.SeqID)
		}
		for _, prev := range inputs[:i] {
			if prev.SeqID == in.SeqID {
				return nil, fmt.Errorf("cudakv: duplicate host command sequence %d", in.SeqID)
			}
		}
		words += len(in.Words)
	}
	registry := make(map[uint64][]uint32, len(inputs))
	for _, in := range inputs {
		registry[in.SeqID] = append([]uint32(nil), in.Words...)
	}
	return &HostCommandConsumer{ring: ring, config: config, inputs: registry}, nil
}

// ConsumeBatch removes at most limit descriptors in FIFO order and calls run
// once for each, with a fresh copy of the registered input that run may keep.
// It returns when the ring is empty, limit descriptors were consumed, or the
// next input would exceed MaxBatchWords for this batch; in each of those
// cases the error is nil and later descriptors stay queued.
//
// A descriptor that fails validation is removed and refused with
// [ErrHostCommandInvalid] without calling run. If run returns an error,
// ConsumeBatch returns it wrapped. A concurrent call refuses with
// [ErrHostCommandBusy] rather than waiting.
//
// The context is checked before each removal; a cancelled context leaves the
// next descriptor queued. Once removed, a descriptor is passed to run even if
// the context is cancelled meanwhile; run receives the context and decides.
// Cancellation cannot roll back run's effects, and a native callback must
// retain its resources until the device work it submitted completes.
// If run panics, its descriptor stays consumed and the panic propagates.
func (c *HostCommandConsumer) ConsumeBatch(ctx context.Context, limit int, run func(context.Context, Descriptor, []uint32) error) (HostCommandBatch, error) {
	var batch HostCommandBatch
	if c == nil || c.ring == nil {
		return batch, errors.New("cudakv: host command consumer must be created with NewHostCommandConsumer")
	}
	if ctx == nil || run == nil {
		return batch, errors.New("cudakv: nil host command context or function")
	}
	if limit <= 0 || limit > c.config.MaxBatch {
		return batch, fmt.Errorf("cudakv: host command batch limit %d outside [1, %d]", limit, c.config.MaxBatch)
	}
	if !c.mu.TryLock() {
		return batch, ErrHostCommandBusy
	}
	defer c.mu.Unlock()
	for batch.Consumed < limit {
		if err := ctx.Err(); err != nil {
			return batch, err
		}
		desc, input, ok, err := c.next(c.config.MaxBatchWords - batch.Words)
		if !ok {
			return batch, nil
		}
		batch.Consumed++
		if err != nil {
			return batch, err
		}
		batch.Words += len(input)
		if err := run(ctx, desc, input); err != nil {
			return batch, fmt.Errorf("cudakv: host command sequence %d: %w", desc.SeqID, err)
		}
		batch.Completed++
	}
	return batch, nil
}

// next removes the oldest descriptor unless the ring is empty or its input
// exceeds budget words, reporting ok=false in those cases. An invalid
// descriptor is removed and returned with an error.
func (c *HostCommandConsumer) next(budget int) (desc Descriptor, input []uint32, ok bool, err error) {
	r := c.ring
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.queued == 0 {
		return Descriptor{}, nil, false, nil
	}
	desc = r.ring[r.tail]
	words, known := c.inputs[desc.SeqID]
	valid := known && desc.Flags == 0 && desc.PromptToken >= 0 && desc.PromptToken < c.config.VocabSize
	if valid && len(words) > budget {
		return Descriptor{}, nil, false, nil
	}
	r.ring[r.tail] = Descriptor{}
	r.tail++
	if r.tail == len(r.ring) {
		r.tail = 0
	}
	r.queued--
	if !valid {
		return desc, nil, true, fmt.Errorf("%w: sequence %d flags %#x token %d", ErrHostCommandInvalid, desc.SeqID, desc.Flags, desc.PromptToken)
	}
	return desc, append([]uint32(nil), words...), true, nil
}
