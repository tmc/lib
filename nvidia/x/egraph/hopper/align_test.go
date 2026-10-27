package hopper

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestExpertAlignEmulated(t *testing.T) {
	tests := []struct {
		name   string
		a      ExpertAlign
		tokens int
		ls     int32
	}{
		{"decode", ExpertAlign{Name: "align_decode", K: 8, Experts: 256, Local: 32, Warps: 4}, 5, 64},
		{"chunks", ExpertAlign{Name: "align_chunks", K: 8, Experts: 256, Local: 32, Warps: 2}, 41, 224},
		{"small", ExpertAlign{Name: "align_small", K: 2, Experts: 8, Local: 3, Warps: 1}, 50, 6},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := rand.New(rand.NewPCG(51, uint64(tt.tokens)))
			ids := make([]int32, tt.tokens*tt.a.K)
			for tok := range tt.tokens {
				// Distinct experts per token, biased toward the local range.
				perm := r.Perm(tt.a.Experts)
				slices.SortStableFunc(perm, func(x, y int) int {
					lx := (x-int(tt.ls)+tt.a.Experts)%tt.a.Experts < 2*tt.a.Local
					ly := (y-int(tt.ls)+tt.a.Experts)%tt.a.Experts < 2*tt.a.Local
					switch {
					case lx && !ly:
						return -1
					case ly && !lx:
						return 1
					}
					return 0
				})
				for k := range tt.a.K {
					ids[tok*tt.a.K+k] = int32(perm[k])
				}
			}
			src, err := tt.a.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			n := len(ids)
			res := runEmu(t, src, fmt.Sprintf("eg_emu_launch(dim3(1), dim3(%d), 0, %s, %d, ids, local_start, offsets, rows, slots)",
				32*tt.a.Warps, tt.a.Name, tt.tokens),
				emuBuf{"ids", "int32_t", i32Bytes(ids)},
				emuBuf{"local_start", "int32_t", i32Bytes([]int32{tt.ls})},
				emuBuf{"offsets", "int32_t", poison(4 * (tt.a.Local + 1))},
				emuBuf{"rows", "int32_t", poison(4 * n)},
				emuBuf{"slots", "int32_t", poison(4 * n)})
			offsets, rows, slots, err := tt.a.Reference(ids, tt.ls)
			if err != nil {
				t.Fatal(err)
			}
			if len(rows) == 0 {
				t.Fatal("test routes nothing to local experts")
			}
			if got := bytesI32(res["offsets"]); !slices.Equal(got, offsets) {
				t.Errorf("offsets = %v, want %v", got, offsets)
			}
			if got := bytesI32(res["rows"])[:len(rows)]; !slices.Equal(got, rows) {
				t.Errorf("rows = %v, want %v", got, rows)
			}
			if got := bytesI32(res["slots"]); !slices.Equal(got, slots) {
				t.Errorf("slots = %v, want %v", got, slots)
			}
		})
	}
}

// TestExpertAlignReference checks the layout's invariants directly.
func TestExpertAlignReference(t *testing.T) {
	a := ExpertAlign{Name: "a", K: 2, Experts: 8, Local: 3}
	ids := []int32{7, 0, 1, 2, 6, 5, 0, 7}
	offsets, rows, slots, err := a.Reference(ids, 6) // local experts are 6, 7, 0
	if err != nil {
		t.Fatal(err)
	}
	if want := []int32{0, 1, 3, 5}; !slices.Equal(offsets, want) {
		t.Errorf("offsets = %v, want %v", offsets, want)
	}
	if want := []int32{2, 0, 3, 0, 3}; !slices.Equal(rows, want) {
		t.Errorf("rows = %v, want %v", rows, want)
	}
	if want := []int32{1, 3, -1, -1, 0, -1, 4, 2}; !slices.Equal(slots, want) {
		t.Errorf("slots = %v, want %v", slots, want)
	}
}

func TestExpertAlignGolden(t *testing.T) {
	src, err := ExpertAlign{Name: "moe8_align", K: 8, Experts: 256, Local: 32}.Source()
	if err != nil {
		t.Fatal(err)
	}
	checkGolden(t, "moe8_align.cu", src)
}

func ExampleExpertAlign() {
	a := ExpertAlign{Name: "align", K: 2, Experts: 4, Local: 2}
	offsets, rows, slots, _ := a.Reference([]int32{1, 2, 0, 1, 3, 0}, 0)
	fmt.Println(offsets, rows, slots)
	// Output: [0 2 4] [1 2 0 1] [2 -1 0 3 -1 1]
}
