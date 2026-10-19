package cudasearch

import (
	"context"
	"fmt"
	"io"
	"math"
	"unsafe"
)

// outputMemoryOwner owns one returned map. next is used only for legacy
// Outputs calls, whose aliases remain live until the executable closes.
type outputMemoryOwner struct {
	reservation *memoryReservation
	next        *outputMemoryOwner
}

func (o *outputMemoryOwner) Close() error {
	if o != nil {
		o.reservation.release()
	}
	return nil
}

type outputFormat uint8

const (
	outputFloats outputFormat = iota
	outputFloatBytes
	outputBytes
)

type outputDescription struct {
	name   string
	length int // float elements for outputFloats, source bytes otherwise
	format outputFormat
}

// Per-call operations keep tests on the same readback path without loading a
// device library. Production supplies actual buffer lengths and synchronous
// copies. Injected callbacks do not acquire a general memory measurement claim.
type outputReadOps struct {
	run        func(context.Context) error
	length     func(int) (int, error)
	readFloats func(int, []float32) error
	readBytes  func(int, []byte) error
	decode     func(int, []byte) ([]float32, error)
}

func outputMemoryAdd(total *int64, bytes int64) error {
	if bytes < 0 || bytes > math.MaxInt64-*total {
		return ErrMemoryBudgetExceeded
	}
	*total += bytes
	return nil
}

// Sizes follow the search ledger's declared units: value/header sizes, requested
// payload capacities and one key/value slot per map insertion. Runtime map
// buckets, allocator rounding, compiler internals and process RSS are excluded.
func readBuiltInOutputs(ctx context.Context, count int, describe func(int) (outputDescription, error), ops outputReadOps) (values map[string][]float32, owner io.Closer, err error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	metadata, err := memoryProduct(count, int64(unsafe.Sizeof(outputDescription{})))
	if err != nil {
		return nil, nil, err
	}
	if err := outputMemoryAdd(&metadata, int64(unsafe.Sizeof(memoryReservation{})+unsafe.Sizeof(outputReadOps{})+unsafe.Sizeof([]outputDescription(nil)))); err != nil {
		return nil, nil, err
	}
	meta, err := reserveSearchMemory(ctx, metadata)
	if err != nil {
		return nil, nil, err
	}
	defer meta.release()
	descriptions := make([]outputDescription, count)
	retained := int64(unsafe.Sizeof(map[string][]float32(nil)) + unsafe.Sizeof(outputMemoryOwner{}) + unsafe.Sizeof(memoryReservation{}))
	var scratch int64
	for i := range descriptions {
		d, err := describe(i)
		if err != nil {
			return nil, nil, err
		}
		if d.length < 0 || d.name == "" || d.format > outputBytes || d.format == outputFloatBytes && d.length%4 != 0 {
			return nil, nil, ErrMemoryAccountingUnavailable
		}
		for _, previous := range descriptions[:i] {
			if previous.name == d.name {
				return nil, nil, ErrMemoryAccountingUnavailable
			}
		}
		elements := d.length
		if d.format == outputFloatBytes {
			elements /= 4
		}
		payload, err := memoryProduct(elements, 4)
		if err != nil {
			return nil, nil, err
		}
		for _, bytes := range []int64{payload, int64(len(d.name)), int64(unsafe.Sizeof("") + unsafe.Sizeof([]float32(nil)))} {
			if err := outputMemoryAdd(&retained, bytes); err != nil {
				return nil, nil, err
			}
		}
		if d.format != outputFloats {
			if err := outputMemoryAdd(&scratch, int64(d.length)); err != nil {
				return nil, nil, err
			}
			if err := outputMemoryAdd(&scratch, int64(unsafe.Sizeof([]byte(nil)))); err != nil {
				return nil, nil, err
			}
		}
		descriptions[i] = d
	}
	reservation, err := reserveSearchMemory(ctx, retained)
	if err != nil {
		return nil, nil, err
	}
	owned := &outputMemoryOwner{reservation: reservation}
	defer func() {
		if err != nil {
			owned.Close()
		}
	}()
	if scratch > 0 {
		if err = outputMemoryAdd(&scratch, int64(unsafe.Sizeof(memoryReservation{}))); err != nil {
			return nil, nil, err
		}
		temporary, reserveErr := reserveSearchMemory(ctx, scratch)
		if reserveErr != nil {
			return nil, nil, reserveErr
		}
		defer temporary.release()
	}
	if err = ctx.Err(); err != nil {
		return nil, nil, err
	}
	if err = ops.run(ctx); err != nil {
		return nil, nil, err
	}
	values = make(map[string][]float32, count)
	for i, d := range descriptions {
		if err = ctx.Err(); err != nil {
			return nil, nil, err
		}
		var data []float32
		if d.format == outputFloats {
			data = make([]float32, d.length)
			err = ops.readFloats(i, data)
		} else {
			raw := make([]byte, d.length)
			if err = ops.readBytes(i, raw); err == nil {
				if err = ctx.Err(); err == nil {
					data, err = ops.decode(i, raw)
				}
			}
		}
		if err != nil {
			return nil, nil, err
		}
		expected := d.length
		if d.format == outputFloatBytes {
			expected /= 4
		}
		if len(data) != expected || cap(data) != expected {
			return nil, nil, fmt.Errorf("cudasearch: decoded output extent differs")
		}
		values[d.name] = data
	}
	if err = ctx.Err(); err != nil {
		return nil, nil, err
	}
	return values, owned, nil
}

func retainBuiltInOutputs(first **outputMemoryOwner, owner io.Closer) {
	if o, ok := owner.(*outputMemoryOwner); ok && o != nil {
		o.next = *first
		*first = o
	}
}

func closeBuiltInOutputs(first **outputMemoryOwner, closeResources func() error) error {
	if err := closeResources(); err != nil {
		return err
	}
	for *first != nil {
		o := *first
		if err := o.Close(); err != nil {
			return err
		}
		*first = o.next
	}
	return nil
}
