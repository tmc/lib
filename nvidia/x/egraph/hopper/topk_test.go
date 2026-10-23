package hopper

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// moe8Router is Moe8's biased sigmoid top-8 over 256 experts
// (moe8_decode_layer/reference.py:51-53, 418-424, 462-463).
func moe8Router() RowTopK {
	return RowTopK{
		Name: "moe8_route", D: 256, K: 8, X: F32, Bias: F32,
		Score:     MustLambda("(sigmoid (mul $0 0.1767766952966369))"),
		Key:       MustLambda("(add $0 $1)"),
		Normalize: true, Eps: 1e-15, Scale: 1,
	}
}

func TestRowTopKEmulated(t *testing.T) {
	tests := []struct {
		name string
		k    RowTopK
		rows int
	}{
		{"moe8_route", moe8Router(), 11},
		{"plain_bf16", RowTopK{Name: "top4", D: 64, K: 4, X: BF16}, 9},
		{"odd_per", RowTopK{Name: "top3", D: 96, K: 3, X: F32, Bias: BF16, Key: MustLambda("(sub $0 $1)")}, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := rand.New(rand.NewPCG(21, 22))
			x := make([]float32, tt.rows*tt.k.D)
			for i := range x {
				x[i] = tt.k.X.Round(float32(r.NormFloat64() * 4))
			}
			// A repeated value in row 0 exercises ties (lower column first).
			x[5], x[40] = 30, 30
			var bias []float32
			bufs := []emuBuf{{"x", tt.k.X.cType(), typed{tt.k.X, f64s(x)}.bytes()}}
			if tt.k.Bias != 0 {
				bias = make([]float32, tt.k.D)
				for i := range bias {
					bias[i] = tt.k.Bias.Round(float32(r.NormFloat64() * 0.05))
				}
				bias[5], bias[40] = 0, 0
				bufs = append(bufs, emuBuf{"bias", tt.k.Bias.cType(), typed{tt.k.Bias, f64s(bias)}.bytes()})
			}
			bufs = append(bufs,
				emuBuf{"ids", "int32_t", make([]byte, 4*tt.rows*tt.k.K)},
				emuBuf{"weights", "float", make([]byte, 4*tt.rows*tt.k.K)})
			src, err := tt.k.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			grid, block := tt.k.Grid(tt.rows)
			args := "x, ids, weights"
			if tt.k.Bias != 0 {
				args = "x, bias, ids, weights"
			}
			out := runEmu(t, src, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
				grid[0], block[0], tt.k.Name, tt.rows, args), bufs...)
			wantIDs, wantW, err := tt.k.Reference(tt.rows, x, bias)
			if err != nil {
				t.Fatal(err)
			}
			gotIDs := bytesI32(out["ids"])
			gotW := bytesF32(out["weights"])
			for i := range wantIDs {
				if gotIDs[i] != wantIDs[i] {
					t.Fatalf("ids[%d] = %d, want %d", i, gotIDs[i], wantIDs[i])
				}
				if d := math.Abs(float64(gotW[i] - wantW[i])); d > 1e-6*math.Abs(float64(wantW[i]))+1e-7 {
					t.Fatalf("weights[%d] = %g, want %g", i, gotW[i], wantW[i])
				}
			}
		})
	}
}

// TestMoe8RouterFormula checks Reference against reference.py's formula for
// one row: the biased scores pick, the unbiased scores weigh.
func TestMoe8RouterFormula(t *testing.T) {
	k := moe8Router()
	r := rand.New(rand.NewPCG(23, 24))
	x := make([]float32, 256)
	bias := make([]float32, 256)
	scores := make([]float64, 256)
	for i := range x {
		x[i] = float32(r.NormFloat64() * 4)
		bias[i] = float32(r.NormFloat64() * 0.1)
		scores[i] = float64(float32(1 / (1 + math.Exp(-float64(x[i])*0.1767766952966369))))
	}
	ids, w, err := k.Reference(1, x, bias)
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	for _, id := range ids {
		sum += scores[id]
	}
	for s, id := range ids {
		key := float32(scores[id]) + bias[id]
		for c := range 256 {
			if !contains(ids, int32(c)) && float32(scores[c])+bias[c] > key {
				t.Fatalf("slot %d picked %d but %d has a larger biased score", s, id, c)
			}
		}
		if want := scores[id] / (sum + 1e-15); math.Abs(float64(w[s])-want) > 1e-6 {
			t.Errorf("weight %d = %g, want %g", s, w[s], want)
		}
	}
}

func contains(s []int32, v int32) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func f64s(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, f := range v {
		out[i] = float64(f)
	}
	return out
}

func bytesI32(b []byte) []int32 {
	out := make([]int32, len(b)/4)
	for i := range out {
		out[i] = int32(uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24)
	}
	return out
}

func TestRowTopKGolden(t *testing.T) {
	src, err := moe8Router().Source()
	if err != nil {
		t.Fatal(err)
	}
	checkGolden(t, "moe8_route.cu", src)
}

func ExampleRowTopK() {
	k := RowTopK{
		Name: "route", D: 256, K: 8, X: F32, Bias: F32,
		Score:     MustLambda("(sigmoid (mul $0 0.1767766952966369))"),
		Key:       MustLambda("(add $0 $1)"),
		Normalize: true, Eps: 1e-15,
	}
	x := make([]float32, 256)
	x[7], x[200] = 9, 8
	ids, w, _ := k.Reference(1, x, make([]float32, 256))
	fmt.Println(ids[:3])
	fmt.Printf("%.4f\n", w[0])
	// Output:
	// [7 200 0]
	// 0.1792
}
