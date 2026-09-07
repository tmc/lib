package kvcache

import (
	"context"
	"errors"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/tmc/lib/x/xla"
)

func TestNativeBatchCache(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	engine, err := xla.NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := engine.Close(); err != nil {
			t.Error(err)
		}
	})
	for _, tt := range []struct {
		name      string
		typ       xla.ElementType
		batch     []int64
		qh, kh    int64
		tolerance float64
	}{
		{"gqa_f32", xla.F32, []int64{2, 2}, 4, 2, 1e-6},
		{"gqa_f16", xla.F16, []int64{2, 2}, 4, 2, 5e-4},
		{"gqa_bf16", xla.BF16, []int64{2, 2}, 4, 2, 4e-3},
		{"mha", xla.F32, []int64{4}, 4, 4, 1e-6},
		{"mqa", xla.F32, []int64{4}, 4, 1, 1e-6},
		{"unheaded", xla.F32, []int64{2, 2}, 0, 0, 1e-6},
		{"unbatched", xla.F32, nil, 0, 0, 1e-6},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := Config{BatchShape: tt.batch, Capacity: 4, KeySize: 2, ValueSize: 3, QueryHeads: tt.qh, KVHeads: tt.kh, ElementType: tt.typ}
			c, err := New(engine, cfg)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := c.Close(); err != nil {
					t.Error(err)
				}
			})
			nb, qh, kh := batchSize(tt.batch), max(tt.qh, 1), max(tt.kh, 1)
			lengths := make([]int64, nb)
			keys, values := make([]float32, nb*kh*4*2), make([]float32, nb*kh*4*3)
			width := 2
			if tt.typ == xla.F32 {
				width = 4
			}
			download := func(b *xla.Buffer, n int) []float64 {
				t.Helper()
				data := make([]byte, n*width)
				if err := b.ToHost(data); err != nil {
					t.Fatal(err)
				}
				out := make([]float64, n)
				for i := range out {
					out[i] = gqaDecode(t, data[i*width:], tt.typ)
				}
				return out
			}
			checkHistory := func() {
				t.Helper()
				for i, b := range []*xla.Buffer{c.keys, c.values} {
					want := [][]float32{keys, values}[i]
					for j, v := range download(b, len(want)) {
						if v != float64(want[j]) {
							t.Fatalf("history %d row element %d = %g, want %g", i, j, v, want[j])
						}
					}
				}
				if !slices.Equal(c.Lengths(), lengths) {
					t.Fatalf("lengths %v, want %v", c.Lengths(), lengths)
				}
			}
			for step, spec := range []struct {
				chunk       int64
				counts      []int64
				full, reset bool
			}{
				{3, []int64{3, 2, 0, 1}, false, false},
				{1, []int64{1, 1, 1, 1}, true, false},
				{3, []int64{0, 1, 3, 0}, false, false},
				{1, []int64{0, 0, 0, 1}, false, false},
				{3, []int64{0, 0, 0, 0}, false, false},
				{3, []int64{1, 0, 2, 1}, false, true},
			} {
				if spec.reset {
					if err := c.Reset(); err != nil {
						t.Fatal(err)
					}
					clear(lengths)
				}
				counts := spec.counts[:nb]
				chunk := spec.chunk
				data := make([][]float32, 3)
				inputs := make([]*xla.Buffer, 3)
				for i := range data {
					heads, features := kh, int64(2)
					if i == 0 {
						heads = qh
					}
					if i == 2 {
						features = 3
					}
					for batch := int64(0); batch < nb; batch++ {
						for head := int64(0); head < heads; head++ {
							for row := int64(0); row < chunk; row++ {
								for col := int64(0); col < features; col++ {
									v := float32((batch+2*head+row+col+int64(step))%7-3) / 2
									if i == 2 {
										v += float32(batch*8 + head*2)
									}
									if row >= counts[batch] {
										v = 100
									} // Poison padded payloads.
									data[i] = append(data[i], v)
								}
							}
						}
					}
					headsInShape := int64(0)
					if tt.qh > 0 {
						headsInShape = heads
					}
					b, err := engine.NewBufferFromHost(gqaEncode(t, data[i], tt.typ), tt.typ, c.shape(chunk, features, headsInShape), c.device)
					if err != nil {
						t.Fatal(err)
					}
					inputs[i] = b
					t.Cleanup(func() {
						if err := b.Close(); err != nil {
							t.Error(err)
						}
					})
				}
				if step == 2 {
					oldPlans := len(c.plans)
					for _, bad := range [][]int64{nil, {-1}, {4}, {1, 1, 1, 1}} {
						if _, err := c.AppendBatch(context.Background(), inputs[0], inputs[1], inputs[2], bad); err == nil {
							t.Fatalf("accepted counts %v", bad)
						}
						checkHistory()
					}
					ctx, cancel := context.WithCancel(context.Background())
					cancel()
					during, stop := context.WithCancel(context.Background())
					defer stop()
					for _, ctx := range []context.Context{ctx, &cancelOnCompletion{Context: during, cancel: stop}} {
						if _, err := c.AppendBatch(ctx, inputs[0], inputs[1], inputs[2], counts); !errors.Is(err, context.Canceled) {
							t.Fatalf("cancellation: %v", err)
						}
						checkHistory()
					}
					if len(c.plans) != oldPlans {
						t.Fatal("validation/cancellation changed program count")
					}
				}
				oldPlan := c.plans[chunk]
				var out *xla.Buffer
				if spec.full {
					out, err = c.Append(context.Background(), inputs[0], inputs[1], inputs[2])
				} else {
					out, err = c.AppendBatch(context.Background(), inputs[0], inputs[1], inputs[2], counts)
				}
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := out.Close(); err != nil {
						t.Error(err)
					}
				})
				if oldPlan != nil && c.plans[chunk] != oldPlan {
					t.Fatal("counts caused recompilation")
				}
				for batch := int64(0); batch < nb; batch++ {
					for head := int64(0); head < kh; head++ {
						for row := int64(0); row < counts[batch]; row++ {
							for i, history := range [][]float32{keys, values} {
								features := int64(i + 2)
								for col := int64(0); col < features; col++ {
									history[((batch*kh+head)*4+lengths[batch]+row)*features+col] = data[i+1][((batch*kh+head)*chunk+row)*features+col]
								}
							}
						}
					}
				}
				got := download(out, int(nb*qh*chunk*3))
				for batch := int64(0); batch < nb; batch++ {
					for head := int64(0); head < qh; head++ {
						kvhead := head / (qh / kh)
						for row := int64(0); row < chunk; row++ {
							var weights []float64
							if row < counts[batch] {
								weights = make([]float64, lengths[batch]+row+1)
								maximum := math.Inf(-1)
								for past := range weights {
									for col := int64(0); col < 2; col++ {
										weights[past] += float64(data[0][((batch*qh+head)*chunk+row)*2+col]) * float64(keys[((batch*kh+kvhead)*4+int64(past))*2+col]) / math.Sqrt(2)
									}
									maximum = math.Max(maximum, weights[past])
								}
								total := 0.0
								for i, w := range weights {
									weights[i] = math.Exp(w - maximum)
									total += weights[i]
								}
								for i := range weights {
									weights[i] /= total
								}
							}
							for col := int64(0); col < 3; col++ {
								want := 0.0
								for past, w := range weights {
									want += w * float64(values[((batch*kh+kvhead)*4+int64(past))*3+col])
								}
								v := got[((batch*qh+head)*chunk+row)*3+col]
								if math.IsNaN(v) || math.Abs(v-want) > 1e-5+tt.tolerance*math.Abs(want) {
									t.Fatalf("step %d batch %d head %d row %d col %d = %g, want %g", step, batch, head, row, col, v, want)
								}
							}
						}
					}
				}
				for i, n := range counts {
					lengths[i] += n
				}
				checkHistory()
				copy := c.Lengths()
				copy[0] = -1
				if c.Len() != slices.Max(lengths) || !slices.Equal(c.Lengths(), lengths) {
					t.Fatal("length accessors")
				}
				if len(c.plans) > 2 {
					t.Fatal("unbounded program retention")
				}
			}
		})
	}
}
