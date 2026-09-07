package kvcache

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"slices"

	"github.com/tmc/lib/openxla"
	"github.com/tmc/lib/x/xla"
)

// Config describes a fixed-capacity cache. BatchShape dimensions, Capacity,
// KeySize and ValueSize must be positive. An empty BatchShape is unbatched.
// ElementType must be F32, F16 or BF16. New copies BatchShape.
type Config struct {
	// BatchShape contains leading batch dimensions. With explicit head counts,
	// it excludes the head axis.
	BatchShape []int64
	// Capacity is the maximum number of cached sequence positions.
	Capacity int64
	// KeySize is the number of features in each query and key.
	KeySize int64
	// ValueSize is the number of features in each value and attention result.
	ValueSize int64
	// QueryHeads and KVHeads select explicit [batch..., heads, sequence, features]
	// tensors. Both zero selects unheaded tensors. Otherwise both must be positive
	// and QueryHeads must be divisible by KVHeads. History stores only KVHeads.
	QueryHeads int64
	KVHeads    int64
	// ElementType is the storage type of inputs, history and results.
	ElementType xla.ElementType
}

// Cache stores a common prefix of key/value history for every batch member.
// Its zero value is unusable; construct one with New.
type Cache struct {
	engine       *xla.Engine
	device       *openxla.Device
	config       Config
	keys, values *xla.Buffer
	plans        map[int64]*xla.Executable
	length       int64
	closed       bool
}

// New initializes zero-filled device history. It borrows engine, which must
// remain open until the cache is closed. History uses the default compilation
// device, which is queried from the initialized buffers.
func New(engine *xla.Engine, config Config) (*Cache, error) {
	if err := validateConfig(config); err != nil {
		return nil, err
	}
	return newCache(engine, config)
}

func newCache(engine *xla.Engine, config Config) (*Cache, error) {
	devices, err := engine.Devices()
	if err != nil {
		return nil, err
	}
	if len(devices) == 0 {
		return nil, fmt.Errorf("kvcache: no addressable devices")
	}
	config.BatchShape = slices.Clone(config.BatchShape)
	c := &Cache{engine: engine, config: config, plans: make(map[int64]*xla.Executable)}
	outputs, err := c.initialize()
	if err != nil {
		return nil, err
	}
	c.keys, c.values = outputs[0], outputs[1]
	c.device, err = c.keys.Device()
	if err != nil {
		return nil, errors.Join(err, c.Close())
	}
	valueDevice, err := c.values.Device()
	if err != nil {
		return nil, errors.Join(err, c.Close())
	}
	if valueDevice != c.device {
		return nil, errors.Join(fmt.Errorf("kvcache: history buffers reside on different devices"), c.Close())
	}
	return c, nil
}

func validateConfig(c Config) error {
	if c.QueryHeads != 0 || c.KVHeads != 0 {
		if c.QueryHeads <= 0 || c.KVHeads <= 0 || c.QueryHeads%c.KVHeads != 0 {
			return fmt.Errorf("kvcache: query heads must be a positive multiple of KV heads")
		}
	}
	width := int64(2)
	switch c.ElementType {
	case xla.F32:
		width = 4
	case xla.F16, xla.BF16:
	default:
		return fmt.Errorf("kvcache: f32, f16 or bf16 elements required")
	}
	for _, spec := range [][2]int64{{c.KeySize, c.KVHeads}, {c.ValueSize, c.KVHeads}, {c.KeySize, c.QueryHeads}, {c.ValueSize, c.QueryHeads}} {
		count := width
		dims := slices.Clone(c.BatchShape)
		if spec[1] > 0 {
			dims = append(dims, spec[1])
		}
		dims = append(dims, c.Capacity, spec[0])
		for _, n := range dims {
			if n <= 0 {
				return fmt.Errorf("kvcache: dimensions must be positive")
			}
			if n > int64(int(^uint(0)>>1))/count {
				return fmt.Errorf("kvcache: byte size overflows int")
			}
			count *= n
		}
	}
	return nil
}

// Device returns the device selected by the cache's initialization graph.
// Input buffers for Append must reside on this device.
func (c *Cache) Device() (*openxla.Device, error) {
	if c == nil || c.closed || c.keys == nil {
		return nil, fmt.Errorf("kvcache: cache is closed")
	}
	return c.device, nil
}

// Len returns the number of cached positions, or zero for a closed cache.
func (c *Cache) Len() int64 {
	if c == nil || c.closed {
		return 0
	}
	return c.length
}

// Reset discards logical history. It retains device buffers and compiled graphs;
// subsequent attention masks exclude stale suffix values.
func (c *Cache) Reset() error {
	if c == nil || c.closed || c.keys == nil {
		return fmt.Errorf("kvcache: cache is closed")
	}
	c.length = 0
	return nil
}

// Append appends k and v and computes causal attention for q. Inputs have shape
// [batch..., chunk, features], or [batch..., heads, chunk, features] when
// explicit head counts are configured. q and k use KeySize. chunk must
// be positive and fit remaining capacity. Inputs are borrowed, and the returned
// buffer belongs to the caller. Inputs must be finite for masked-zero guarantees.
//
// Compilation, validation, execution and cancellation errors preserve history.
// Cancellation during execution waits for completion before releasing outputs.
// Failure to release the position buffer, replaced history, or an evicted
// program closes the cache and returns no output.
func (c *Cache) Append(ctx context.Context, q, k, v *xla.Buffer) (*xla.Buffer, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if c == nil || c.closed || c.keys == nil {
		return nil, fmt.Errorf("kvcache: cache is closed")
	}
	var chunk int64
	for i, input := range []*xla.Buffer{q, k, v} {
		typ, shape, err := input.Shape()
		if err != nil {
			return nil, fmt.Errorf("kvcache: input %d: %w", i, err)
		}
		device, err := input.Device()
		if err != nil {
			return nil, fmt.Errorf("kvcache: input %d device: %w", i, err)
		}
		if device != c.device {
			return nil, fmt.Errorf("kvcache: input %d is on a different device", i)
		}
		features := c.config.KeySize
		heads := c.config.KVHeads
		if i == 0 {
			heads = c.config.QueryHeads
		}
		if i == 2 {
			features = c.config.ValueSize
		}
		expected := c.shape(1, features, heads)
		rank := len(expected)
		if typ != c.config.ElementType || len(shape) != rank {
			return nil, fmt.Errorf("kvcache: input %d shape or element type mismatch", i)
		}
		if i == 0 {
			chunk = shape[rank-2]
		}
		if chunk <= 0 {
			return nil, fmt.Errorf("kvcache: positive chunk length required")
		}
		expected[rank-2] = chunk
		if !slices.Equal(shape, expected) {
			return nil, fmt.Errorf("kvcache: input %d shape or chunk length mismatch", i)
		}
	}
	if chunk > c.config.Capacity-c.length {
		return nil, fmt.Errorf("kvcache: capacity exceeded")
	}
	return c.append(ctx, q, k, v, chunk)
}

func (c *Cache) append(ctx context.Context, q, k, v *xla.Buffer, chunk int64) (*xla.Buffer, error) {
	executable, err := c.program(chunk)
	if err != nil {
		return nil, err
	}
	var data [8]byte
	binary.LittleEndian.PutUint64(data[:], uint64(c.length))
	position, err := c.engine.NewBufferFromHost(data[:], openxla.Buffer_Type_S64, nil, c.device)
	if err != nil {
		return nil, err
	}
	outputs, err := executable.Execute(ctx, []*xla.Buffer{c.keys, c.values, q, k, v, position})
	cleanupErr := position.Close()
	if cleanupErr != nil {
		return nil, errors.Join(err, cleanupErr, closeBuffers(outputs), c.Close())
	}
	if err != nil {
		return nil, err
	}
	if len(outputs) != 3 {
		return nil, errors.Join(fmt.Errorf("kvcache: got %d outputs, want 3", len(outputs)), closeBuffers(outputs))
	}
	return c.commit(outputs, chunk, (*xla.Buffer).Close)
}

func (c *Cache) commit(outputs []*xla.Buffer, chunk int64, release func(*xla.Buffer) error) (*xla.Buffer, error) {
	err := errors.Join(release(c.keys), release(c.values))
	c.keys, c.values = nil, nil
	if err != nil {
		return nil, errors.Join(err, closeBuffers(outputs), c.Close())
	}
	c.keys, c.values = outputs[0], outputs[1]
	c.length += chunk
	return outputs[2], nil
}

// Close releases owned buffers and executables, invalidating the cache even if
// native cleanup reports an error. It does not close the borrowed engine.
func (c *Cache) Close() error {
	if c == nil {
		return nil
	}
	c.closed = true
	c.length = 0
	err := closeBuffers([]*xla.Buffer{c.keys, c.values})
	c.keys, c.values = nil, nil
	for n, program := range c.plans {
		err = errors.Join(err, program.Close())
		delete(c.plans, n)
	}
	return err
}

func closeBuffers(buffers []*xla.Buffer) (err error) {
	for _, b := range buffers {
		err = errors.Join(err, b.Close())
	}
	return err
}
