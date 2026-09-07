package kvcache

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/tmc/lib/x/stablehlo"
	"github.com/tmc/lib/x/stablehlo/builder"
	"github.com/tmc/lib/x/xla"
)

func (c *Cache) shape(length, features, heads int64) []int64 {
	shape := slices.Clone(c.config.BatchShape)
	if heads > 0 {
		shape = append(shape, heads)
	}
	return append(shape, length, features)
}

func (c *Cache) queryTensor(length int64) builder.Type {
	typ := c.tensor(length, c.config.KeySize)
	typ.Shape = c.shape(length, c.config.KeySize, c.config.QueryHeads)
	return typ
}

func (c *Cache) tensor(length, features int64) builder.Type {
	element := "f32"
	if c.config.ElementType == xla.F16 {
		element = "f16"
	}
	if c.config.ElementType == xla.BF16 {
		element = "bf16"
	}
	return builder.Type{Shape: c.shape(length, features, c.config.KVHeads), ElementType: element}
}

func (c *Cache) compile(b *builder.FuncBuilder) (*xla.Executable, error) {
	var program bytes.Buffer
	if err := (&stablehlo.Module{Funcs: []stablehlo.Emitter{b}}).Emit(&program); err != nil {
		return nil, err
	}
	return c.engine.Compile(program.Bytes(), "mlir", nil)
}

func (c *Cache) initialize() ([]*xla.Buffer, error) {
	b := builder.NewFunc("main")
	k, err := b.Constant(0, c.tensor(c.config.Capacity, c.config.KeySize))
	if err != nil {
		return nil, err
	}
	v, err := b.Constant(0, c.tensor(c.config.Capacity, c.config.ValueSize))
	if err != nil {
		return nil, err
	}
	b.Return(k, v)
	executable, err := c.compile(b)
	if err != nil {
		return nil, err
	}
	outputs, runErr := executable.Execute(context.Background(), nil)
	err = errors.Join(runErr, executable.Close())
	if err == nil && len(outputs) != 2 {
		err = fmt.Errorf("kvcache: initialization returned %d outputs, want 2", len(outputs))
	}
	if err != nil {
		return nil, errors.Join(err, closeBuffers(outputs))
	}
	return outputs, nil
}

func (c *Cache) program(chunk int64) (*xla.Executable, error) {
	if p := c.plans[chunk]; p != nil {
		return p, nil
	}
	b, err := c.appendGraph(chunk)
	if err != nil {
		return nil, err
	}
	p, err := c.compile(b)
	if err != nil {
		return nil, err
	}
	// Retain the decode graph and only the most recent prefill shape.
	if chunk != 1 {
		for n, old := range c.plans {
			if n != 1 {
				delete(c.plans, n)
				if err := old.Close(); err != nil {
					return nil, errors.Join(err, p.Close(), c.Close())
				}
			}
		}
	}
	c.plans[chunk] = p
	return p, nil
}

func (c *Cache) appendGraph(chunk int64) (*builder.FuncBuilder, error) {
	b := builder.NewFunc("main", c.tensor(c.config.Capacity, c.config.KeySize), c.tensor(c.config.Capacity, c.config.ValueSize), c.queryTensor(chunk), c.tensor(chunk, c.config.KeySize), c.tensor(chunk, c.config.ValueSize), builder.Type{ElementType: "i64"})
	zero, err := b.IntegerConstant(0, builder.Type{ElementType: "i64"})
	if err != nil {
		return nil, err
	}
	starts := make([]builder.Value, len(b.Arg(0).Type.Shape))
	for i := range starts {
		starts[i] = zero
	}
	starts[len(starts)-2] = b.Arg(5)
	k, err := b.DynamicUpdateSlice(b.Arg(0), b.Arg(3), starts)
	if err != nil {
		return nil, err
	}
	v, err := b.DynamicUpdateSlice(b.Arg(1), b.Arg(4), starts)
	if err != nil {
		return nil, err
	}
	mask, err := b.CausalMaskAt(chunk, c.config.Capacity, b.Arg(5))
	if err != nil {
		return nil, err
	}
	var out builder.Value
	if c.config.QueryHeads > 0 {
		out, err = b.MaskedGroupedQueryAttention(b.Arg(2), k, v, mask)
	} else {
		out, err = b.MaskedScaledDotProductAttention(b.Arg(2), k, v, mask)
	}
	if err != nil {
		return nil, err
	}
	b.Return(k, v, out)
	return b, nil
}
