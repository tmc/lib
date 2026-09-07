package llama

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

func f32(shape ...int64) builder.Type { return builder.Type{Shape: shape, ElementType: "f32"} }
func i64(shape ...int64) builder.Type { return builder.Type{Shape: shape, ElementType: "i64"} }
func (m *Model) historyType() builder.Type {
	c := m.config
	return f32(c.BatchSize, c.KVHeads, c.Capacity, c.HeadSize)
}
func (m *Model) metadataTypes(chunk int64) []builder.Type {
	c := m.config
	return []builder.Type{i64(c.BatchSize, chunk), i64(c.BatchSize), i64(c.BatchSize), i64(c.BatchSize * c.KVHeads * chunk), f32(c.BatchSize, 1, chunk, c.HeadSize/2), f32(c.BatchSize, 1, chunk, c.HeadSize/2)}
}
func (m *Model) compile(b *builder.FuncBuilder) (*xla.Executable, error) {
	var text bytes.Buffer
	if err := (&stablehlo.Module{Funcs: []stablehlo.Emitter{b}}).Emit(&text); err != nil {
		return nil, err
	}
	return m.engine.Compile(text.Bytes(), "mlir", nil)
}
func (m *Model) initialize(ctx context.Context) error {
	b := builder.NewFunc("main")
	var values []builder.Value
	for range 2 * len(m.layers) {
		v, err := b.Constant(0, m.historyType())
		if err != nil {
			return err
		}
		values = append(values, v)
	}
	b.Return(values...)
	p, err := m.compile(b)
	if err != nil {
		return err
	}
	outputs, runErr := p.Execute(ctx, nil)
	if err := errors.Join(runErr, p.Close()); err != nil {
		return errors.Join(err, closeBuffers(outputs))
	}
	if len(outputs) != len(values) {
		return errors.Join(fmt.Errorf("llama: unexpected initialization outputs"), closeBuffers(outputs))
	}
	m.history = outputs
	for _, v := range outputs {
		d, err := v.Device()
		if err != nil {
			return err
		}
		if m.device != nil && d != m.device {
			return fmt.Errorf("llama: history buffers reside on different devices")
		}
		m.device = d
	}
	return nil
}
func (m *Model) program(chunk int64) (*xla.Executable, error) {
	if p := m.plans[chunk]; p != nil {
		return p, nil
	}
	b, err := m.forwardGraph(chunk)
	if err != nil {
		return nil, err
	}
	p, err := m.compile(b)
	if err != nil {
		return nil, fmt.Errorf("llama: compile forward: %w", err)
	}
	if chunk != 1 {
		for n, old := range m.plans {
			if n != 1 {
				delete(m.plans, n)
				if err := old.Close(); err != nil {
					return nil, errors.Join(err, p.Close(), m.Close())
				}
			}
		}
	}
	m.plans[chunk] = p
	return p, nil
}

// graph retains the first construction error. Helpers stop emitting after it,
// so a failed intermediate cannot be used to infer a later tensor shape.
type graph struct {
	b   *builder.FuncBuilder
	err error
}

func (g *graph) take(v builder.Value, err error) builder.Value {
	if g.err == nil {
		g.err = err
	}
	return v
}
func (g *graph) reshape(x builder.Value, typ builder.Type) builder.Value {
	if g.err != nil {
		return builder.Value{}
	}
	return g.take(g.b.Reshape(x, typ))
}
func (g *graph) transpose(x builder.Value, axes []int64, typ builder.Type) builder.Value {
	if g.err != nil {
		return builder.Value{}
	}
	return g.take(g.b.Transpose(x, axes, typ))
}
func (g *graph) add(x, y builder.Value) builder.Value {
	if g.err != nil {
		return builder.Value{}
	}
	return g.take(g.b.Add(x, y))
}
func (g *graph) norm(x builder.Value, param int, eps float32) builder.Value {
	if g.err != nil {
		return builder.Value{}
	}
	return g.take(g.b.RMSNorm(x, g.b.Arg(param), int64(len(x.Type.Shape)-1), eps))
}
func (g *graph) linear(x builder.Value, param, bias int) builder.Value {
	if g.err != nil {
		return builder.Value{}
	}
	w := g.b.Arg(param)
	shape := slices.Clone(x.Type.Shape)
	shape[len(shape)-1] = w.Type.Shape[0]
	x = g.take(g.b.DotGeneral(x, w, nil, nil, []int64{int64(len(shape) - 1)}, []int64{1}, f32(shape...)))
	if bias >= 0 && g.err == nil {
		expanded := g.take(g.b.BroadcastInDim(g.b.Arg(bias), []int64{int64(len(shape) - 1)}, f32(shape...)))
		x = g.add(x, expanded)
	}
	return x
}
func (g *graph) rotate(x, cosine, sine builder.Value) builder.Value {
	if g.err != nil {
		return builder.Value{}
	}
	return g.take(g.b.RotaryEmbedding(x, cosine, sine))
}
func (g *graph) update(old, values, indices builder.Value, c Config, chunk int64) builder.Value {
	if g.err != nil {
		return builder.Value{}
	}
	typ := old.Type
	rows := c.BatchSize * c.KVHeads
	old = g.reshape(old, f32(rows*c.Capacity, c.HeadSize))
	values = g.reshape(values, f32(rows*chunk, c.HeadSize))
	if g.err != nil {
		return builder.Value{}
	}
	out := g.take(g.b.ScatterRows(old, indices, values))
	return g.reshape(out, typ)
}
func (g *graph) attention(q, k, v, mask builder.Value) builder.Value {
	if g.err != nil {
		return builder.Value{}
	}
	return g.take(g.b.MaskedGroupedQueryAttention(q, k, v, mask))
}
func (g *graph) gated(gate, up builder.Value) builder.Value {
	if g.err != nil {
		return builder.Value{}
	}
	return g.take(g.b.SwiGLU(gate, up))
}

func (m *Model) forwardGraph(chunk int64) (*builder.FuncBuilder, error) {
	c := m.config
	types := slices.Clone(m.types)
	for range 2 * len(m.layers) {
		types = append(types, m.historyType())
	}
	meta := len(types)
	types = append(types, m.metadataTypes(chunk)...)
	b := builder.NewFunc("main", types...)
	g := &graph{b: b}
	x := g.take(b.GatherRows(b.Arg(m.embedding), b.Arg(meta)))
	mask := g.take(b.BatchedCausalMask(chunk, c.Capacity, b.Arg(meta+1), b.Arg(meta+2)))
	mask = g.reshape(mask, builder.Type{Shape: []int64{c.BatchSize, 1, chunk, c.Capacity}, ElementType: "i1"})
	cosine, sine := b.Arg(meta+4), b.Arg(meta+5)
	toHeads := func(v builder.Value, heads int64) builder.Value {
		v = g.reshape(v, f32(c.BatchSize, chunk, heads, c.HeadSize))
		return g.transpose(v, []int64{0, 2, 1, 3}, f32(c.BatchSize, heads, chunk, c.HeadSize))
	}
	var history []builder.Value
	for i, p := range m.layers {
		normal := g.norm(x, p.norm, c.NormEpsilon)
		q := g.rotate(toHeads(g.linear(normal, p.q, p.qBias), c.QueryHeads), cosine, sine)
		k := g.rotate(toHeads(g.linear(normal, p.k, p.kBias), c.KVHeads), cosine, sine)
		v := toHeads(g.linear(normal, p.v, p.vBias), c.KVHeads)
		k = g.update(b.Arg(len(m.types)+2*i), k, b.Arg(meta+3), c, chunk)
		v = g.update(b.Arg(len(m.types)+2*i+1), v, b.Arg(meta+3), c, chunk)
		history = append(history, k, v)
		attention := g.attention(q, k, v, mask)
		attention = g.transpose(attention, []int64{0, 2, 1, 3}, f32(c.BatchSize, chunk, c.QueryHeads, c.HeadSize))
		attention = g.reshape(attention, f32(c.BatchSize, chunk, c.QueryHeads*c.HeadSize))
		x = g.add(x, g.linear(attention, p.o, -1))
		normal = g.norm(x, p.post, c.NormEpsilon)
		gate, up := g.linear(normal, p.gate, -1), g.linear(normal, p.up, -1)
		x = g.add(x, g.linear(g.gated(gate, up), p.down, -1))
	}
	logits := g.linear(g.norm(x, m.norm, c.NormEpsilon), m.head, -1)
	if g.err != nil {
		return nil, g.err
	}
	b.Return(append(history, logits)...)
	return b, nil
}
