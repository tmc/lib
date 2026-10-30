package main

import (
	"encoding/json"
	"errors"
	"os"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

type afmoeAttentionTensor struct {
	Data []float32 `json:"data"`
}

type afmoeAttentionFixture struct {
	Attention struct {
		Heads       int                    `json:"query_heads"`
		KVHeads     int                    `json:"kv_heads"`
		HeadDim     int                    `json:"head_dim"`
		Query       [][]float32            `json:"query"`
		Keys        [][][]float32          `json:"keys"`
		Values      [][][]float32          `json:"values"`
		Pos         int                    `json:"query_position"`
		Gate        afmoeAttentionTensor   `json:"gate"`
		QNormWeight afmoeAttentionTensor   `json:"q_norm_weight"`
		KNormWeight afmoeAttentionTensor   `json:"k_norm_weight"`
		Local       afmoeAttentionExpected `json:"local"`
		Global      afmoeAttentionExpected `json:"global"`
	} `json:"attention"`
	Window struct {
		Pos     int                      `json:"query_position"`
		Window  int                      `json:"window"`
		Count   int                      `json:"key_count"`
		Query   [][]float32              `json:"query"`
		Markers map[string][][]float32   `json:"value_markers"`
		Local   afmoeAttentionWindowWant `json:"local"`
		Global  afmoeAttentionWindowWant `json:"global"`
	} `json:"window_boundary"`
	RoPE struct {
		Input    afmoeAttentionTensor `json:"input"`
		Pos      int                  `json:"position"`
		Theta    float32              `json:"theta"`
		Expected afmoeAttentionTensor `json:"expected"`
	} `json:"rope_half_split"`
}

type afmoeAttentionExpected struct {
	Query   [][]float32   `json:"normalized_query"`
	Keys    [][][]float32 `json:"normalized_keys"`
	Ungated []float32     `json:"ungated"`
	Gated   []float32     `json:"gated"`
}

type afmoeAttentionWindowWant struct {
	Start   int                  `json:"first_allowed_index"`
	Count   int                  `json:"allowed_count"`
	Ungated afmoeAttentionTensor `json:"ungated"`
}

func readAfmoeAttentionFixture(t *testing.T) afmoeAttentionFixture {
	t.Helper()
	b, err := os.ReadFile("testdata/afmoe-mini/reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var f afmoeAttentionFixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func flattenAfmoeAttention(x [][]float32) []float32 {
	var flat []float32
	for _, row := range x {
		flat = append(flat, row...)
	}
	return flat
}

func flattenAfmoeAttentionCache(x [][][]float32) []float32 {
	var flat []float32
	for _, token := range x {
		flat = append(flat, flattenAfmoeAttention(token)...)
	}
	return flat
}

func checkAfmoeAttentionValues(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("value[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}

func TestAfmoeAttentionReference(t *testing.T) {
	f := readAfmoeAttentionFixture(t).Attention
	s := afmoeAttentionShape{f.Heads, f.KVHeads, f.HeadDim}
	for _, tc := range []struct {
		name  string
		local bool
		want  afmoeAttentionExpected
	}{
		{"local", true, f.Local},
		{"global", false, f.Global},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var q []float32
			for _, head := range f.Query {
				n, err := afmoeRMSNorm(head, f.QNormWeight.Data, 1e-5)
				if err != nil {
					t.Fatal(err)
				}
				q = append(q, n...)
			}
			var keys []float32
			for pos, token := range f.Keys {
				var k []float32
				for _, head := range token {
					n, err := afmoeRMSNorm(head, f.KNormWeight.Data, 1e-5)
					if err != nil {
						t.Fatal(err)
					}
					k = append(k, n...)
				}
				if tc.local {
					// Q is rotated only once, at the query position. A scratch
					// query lets each cached K rotate at its original position.
					if err := afmoeApplyRoPE(make([]float32, len(q)), k, s, pos, 10000); err != nil {
						t.Fatal(err)
					}
				}
				keys = append(keys, k...)
			}
			window := 0
			if tc.local {
				window = 2048
				if err := afmoeApplyRoPE(q, make([]float32, s.KVHeads*s.HeadDim), s, f.Pos, 10000); err != nil {
					t.Fatal(err)
				}
			}
			checkAfmoeAttentionValues(t, q, flattenAfmoeAttention(tc.want.Query))
			checkAfmoeAttentionValues(t, keys, flattenAfmoeAttentionCache(tc.want.Keys))
			out, err := afmoeAttentionDecode(q, keys, flattenAfmoeAttentionCache(f.Values), s, f.Pos, window)
			if err != nil {
				t.Fatal(err)
			}
			checkAfmoeAttentionValues(t, out, tc.want.Ungated)
			if err := afmoeGateAttention(out, f.Gate.Data); err != nil {
				t.Fatal(err)
			}
			checkAfmoeAttentionValues(t, out, tc.want.Gated)
		})
	}
}

func TestAfmoeAttentionWindowBoundary(t *testing.T) {
	f := readAfmoeAttentionFixture(t).Window
	s := afmoeAttentionShape{4, 2, 2}
	keys, values := make([]float32, f.Count*4), make([]float32, f.Count*4)
	for pos, rows := range f.Markers {
		i, err := strconv.Atoi(pos)
		if err != nil {
			t.Fatal(err)
		}
		copy(values[i*4:], flattenAfmoeAttention(rows))
	}
	for _, tc := range []struct {
		name   string
		window int
		want   afmoeAttentionWindowWant
	}{
		{"local", f.Window, f.Local},
		{"global", 0, f.Global},
	} {
		t.Run(tc.name, func(t *testing.T) {
			start := afmoeAttentionStart(f.Pos, tc.window)
			if start != tc.want.Start || f.Pos-start+1 != tc.want.Count {
				t.Fatalf("start/count = %d/%d, want %d/%d", start, f.Pos-start+1, tc.want.Start, tc.want.Count)
			}
			out, err := afmoeAttentionDecode(flattenAfmoeAttention(f.Query), keys, values, s, f.Pos, tc.window)
			if err != nil {
				t.Fatal(err)
			}
			checkAfmoeAttentionValues(t, out, tc.want.Ungated.Data)
		})
	}
}

func TestAfmoeRoPEHalfSplit(t *testing.T) {
	f := readAfmoeAttentionFixture(t).RoPE
	q, k := append([]float32(nil), f.Input.Data...), append([]float32(nil), f.Input.Data...)
	if err := afmoeApplyRoPE(q, k, afmoeAttentionShape{1, 1, 4}, f.Pos, f.Theta); err != nil {
		t.Fatal(err)
	}
	checkAfmoeAttentionValues(t, q, f.Expected.Data)
	checkAfmoeAttentionValues(t, k, f.Expected.Data)
}

func TestAfmoeAttentionRejectsInvalidShapes(t *testing.T) {
	for _, tc := range []struct {
		name string
		s    afmoeAttentionShape
		q    []float32
		k    []float32
		v    []float32
		pos  int
		win  int
	}{
		{"zero kv heads", afmoeAttentionShape{4, 0, 2}, nil, nil, nil, 0, 0},
		{"uneven groups", afmoeAttentionShape{3, 2, 2}, nil, nil, nil, 0, 0},
		{"odd head width", afmoeAttentionShape{1, 1, 3}, nil, nil, nil, 0, 0},
		{"short query", afmoeAttentionShape{1, 1, 2}, []float32{1}, make([]float32, 2), make([]float32, 2), 0, 0},
		{"cache missing position", afmoeAttentionShape{1, 1, 2}, make([]float32, 2), make([]float32, 2), make([]float32, 2), 1, 0},
		{"unequal cache lengths", afmoeAttentionShape{1, 1, 2}, make([]float32, 2), make([]float32, 2), make([]float32, 4), 0, 0},
		{"negative position", afmoeAttentionShape{1, 1, 2}, make([]float32, 2), make([]float32, 2), make([]float32, 2), -1, 0},
		{"negative window", afmoeAttentionShape{1, 1, 2}, make([]float32, 2), make([]float32, 2), make([]float32, 2), 0, -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := afmoeAttentionDecode(tc.q, tc.k, tc.v, tc.s, tc.pos, tc.win); err == nil {
				t.Fatal("accepted invalid attention input")
			}
		})
	}
	if err := afmoeGateAttention([]float32{1}, nil); err == nil {
		t.Fatal("accepted mismatched gate")
	}
	if err := afmoeApplyRoPE(make([]float32, 2), make([]float32, 2), afmoeAttentionShape{1, 1, 2}, 0, 0); err == nil {
		t.Fatal("accepted zero theta")
	}
}

func TestAfmoeAttentionModuleLookupCleanup(t *testing.T) {
	lookupErr, unloadErr := errors.New("lookup failed"), errors.New("unload failed")
	for _, failUnload := range []bool{false, true} {
		t.Run(strconv.FormatBool(failUnload), func(t *testing.T) {
			lookups, closes := 0, 0
			m, err := lookupAfmoeAttentionModule(driver.Module(11), func(string) (driver.Function, error) {
				lookups++
				if lookups == 2 {
					return 0, lookupErr
				}
				return driver.Function(12), nil
			}, func(*driver.Module) error {
				closes++
				if failUnload {
					return unloadErr
				}
				return nil
			})
			if !errors.Is(err, lookupErr) || lookups != 2 || closes != 1 {
				t.Fatalf("lookup cleanup = %v, calls %d/%d", err, lookups, closes)
			}
			if failUnload {
				if !errors.Is(err, unloadErr) || m == nil || m.mod != driver.Module(11) {
					t.Fatalf("lost failed-unload owner: module %v, err %v", m, err)
				}
				if err := m.close(func(*driver.Module) error { closes++; return nil }); err != nil || m.mod != 0 {
					t.Fatalf("retry close = %v, module %v", err, m)
				}
				if err := m.close(func(*driver.Module) error { t.Fatal("double close"); return nil }); err != nil {
					t.Fatal(err)
				}
			} else if m != nil {
				t.Fatalf("retained successfully unloaded module: %v", m)
			}
		})
	}
}
