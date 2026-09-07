package llama

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/tmc/lib/x/xla"
)

func testConfig() Config {
	return Config{BatchSize: 2, Capacity: 5, VocabSize: 7, HiddenSize: 6, IntermediateSize: 7, QueryHeads: 2, KVHeads: 1, HeadSize: 4, NormEpsilon: 1e-5, RopeTheta: 10000}
}

type cancelOnCompletion struct {
	context.Context
	cancel context.CancelFunc
	calls  int
}

func (c *cancelOnCompletion) Err() error {
	c.calls++
	if c.calls == 3 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestNativeModel(t *testing.T) {
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
		name   string
		tied   bool
		qh, kh int64
	}{{"untied", false, 2, 1}, {"tied", true, 2, 1}, {"grouped", false, 4, 2}} {
		t.Run(tt.name, func(t *testing.T) {
			tied := tt.tied
			c := testConfig()
			c.QueryHeads, c.KVHeads = tt.qh, tt.kh
			weights := fixture(c, tied)
			ref := newReference(c, fixture(c, tied))
			m, err := New(context.Background(), engine, c, weights)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := m.Close(); err != nil {
					t.Error(err)
				}
			})
			if (m.head == m.embedding) != tied {
				t.Fatal("incorrect tied parameter ownership")
			}
			// Mutating the host source after New must not alter device weights.
			weights.Embedding[0] += 1000
			snapshot := func() [][]byte {
				t.Helper()
				size, _ := byteSize(4, c.BatchSize, c.KVHeads, c.Capacity, c.HeadSize)
				out := make([][]byte, len(m.history))
				for i, b := range m.history {
					out[i] = make([]byte, size)
					if err := b.ToHost(out[i]); err != nil {
						t.Fatal(err)
					}
				}
				return out
			}
			lengths := []int64{0, 0}
			for step, spec := range []struct {
				chunk       int64
				counts      []int64
				full, reset bool
			}{
				{3, []int64{3, 1}, false, false}, {1, []int64{1, 1}, true, false}, {3, []int64{1, 0}, false, false}, {3, []int64{0, 3}, false, false}, {3, []int64{0, 0}, false, false}, {2, []int64{1, 2}, false, true},
			} {
				if spec.reset {
					if err := m.Reset(); err != nil {
						t.Fatal(err)
					}
					clear(lengths)
					ref = newReference(c, fixture(c, tied))
				}
				ids := make([]int32, c.BatchSize*spec.chunk)
				for batch, n := range spec.counts {
					for row := int64(0); row < spec.chunk; row++ {
						i := int64(batch)*spec.chunk + row
						ids[i] = -999
						if row < n {
							ids[i] = int32((i + int64(step)) % c.VocabSize)
						}
					}
				}
				if step == 2 {
					before := snapshot()
					oldLengths := m.Lengths()
					oldPrograms := len(m.plans)
					bad := slices.Clone(ids)
					bad[0] = int32(c.VocabSize)
					for _, v := range []struct {
						ids    []int32
						counts []int64
					}{{bad, spec.counts}, {ids, []int64{2, 0}}, {ids, []int64{-1, 0}}, {ids, []int64{1}}, {ids[:1], spec.counts}} {
						if _, err := m.Forward(context.Background(), v.ids, v.counts); err == nil {
							t.Fatal("invalid forward accepted")
						}
					}
					ctx, cancel := context.WithCancel(context.Background())
					cancel()
					during, stop := context.WithCancel(context.Background())
					defer stop()
					for _, ctx := range []context.Context{ctx, &cancelOnCompletion{Context: during, cancel: stop}} {
						if _, err := m.Forward(ctx, ids, spec.counts); !errors.Is(err, context.Canceled) {
							t.Fatalf("cancellation: %v", err)
						}
					}
					for i, b := range snapshot() {
						if !slices.Equal(b, before[i]) {
							t.Fatalf("cancel/validation changed layer history %d", i)
						}
					}
					if !slices.Equal(m.Lengths(), oldLengths) || len(m.plans) != oldPrograms {
						t.Fatal("cancel/validation changed state")
					}
				}
				priorHistory, priorLengths := snapshot(), slices.Clone(lengths)
				oldProgram := m.plans[spec.chunk]
				counts := spec.counts
				if spec.full {
					counts = nil
				}
				out, err := m.Forward(context.Background(), ids, counts)
				if err != nil {
					t.Fatal(err)
				}
				data := make([]byte, len(ids)*int(c.VocabSize)*4)
				readErr := out.ToHost(data)
				closeErr := out.Close()
				if err := errors.Join(readErr, closeErr); err != nil {
					t.Fatal(err)
				}
				for batch, n := range spec.counts {
					for row := int64(0); row < n; row++ {
						index := int64(batch)*spec.chunk + row
						want := ref.token(batch, ids[index])
						for v, w := range want {
							got := float64(math.Float32frombits(binary.LittleEndian.Uint32(data[(index*c.VocabSize+int64(v))*4:])))
							if math.IsNaN(got) || math.Abs(got-w) > 2e-5*(1+math.Abs(w)) {
								t.Fatalf("step %d batch %d row %d logit %d = %.9g, want %.9g", step, batch, row, v, got, w)
							}
						}
					}
					lengths[batch] += n
				}
				if !slices.Equal(m.Lengths(), lengths) {
					t.Fatalf("lengths %v, want %v", m.Lengths(), lengths)
				}
				copy := m.Lengths()
				copy[0] = -1
				if m.Lengths()[0] != lengths[0] {
					t.Fatal("lengths alias")
				}
				if oldProgram != nil && m.plans[spec.chunk] != oldProgram {
					t.Fatal("count values caused recompilation")
				}
				if len(m.plans) > 2 {
					t.Fatal("unbounded program retention")
				}
				for i, data := range snapshot() {
					for batch := range lengths {
						for head := int64(0); head < c.KVHeads; head++ {
							for pos := int64(0); pos < c.Capacity; pos++ {
								if pos >= priorLengths[batch] && pos < lengths[batch] {
									continue
								}
								start := ((int64(batch)*c.KVHeads+head)*c.Capacity + pos) * c.HeadSize * 4
								if !slices.Equal(data[start:start+c.HeadSize*4], priorHistory[i][start:start+c.HeadSize*4]) {
									t.Fatalf("step %d changed untouched history %d batch %d position %d", step, i, batch, pos)
								}
							}
						}
					}
					expected := ref.keys
					if i%2 == 1 {
						expected = ref.values
					}
					layer := i / 2
					for batch, n := range lengths {
						for head := int64(0); head < c.KVHeads; head++ {
							for pos := int64(0); pos < n; pos++ {
								for col := int64(0); col < c.HeadSize; col++ {
									index := ((int64(batch)*c.KVHeads+head)*c.Capacity+pos)*c.HeadSize + col
									got := float64(math.Float32frombits(binary.LittleEndian.Uint32(data[index*4:])))
									want := expected[batch][layer][pos][head*c.HeadSize+col]
									if math.IsNaN(got) || math.Abs(got-want) > 2e-5*(1+math.Abs(want)) {
										t.Fatalf("history %d batch %d position %d = %g, want %g", i, batch, pos, got, want)
									}
								}
							}
						}
					}
				}
			}
		})
	}
}
