package egraph_test

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

type q81Block struct {
	d, sum uint16
	qs     [32]int8
}

func q81Value(x []q81Block, index int) float64 {
	b := x[index/32]
	return float64(halfToFloat32(b.d)) * float64(b.qs[index%32])
}

func q80Q81Reference(weights []byte, x []q81Block, rows, cols int) []float64 {
	const blockBytes = 34
	const blockSize = 32
	y := make([]float64, rows)
	for row := 0; row < rows; row++ {
		for b := 0; b < cols/blockSize; b++ {
			off := (row*cols/blockSize + b) * blockBytes
			d := float64(halfToFloat32(uint16(weights[off]) | uint16(weights[off+1])<<8))
			for i := 0; i < blockSize; i++ {
				y[row] += d * float64(int8(weights[off+2+i])) * q81Value(x, b*blockSize+i)
			}
		}
	}
	return y
}

func q2KQ81Reference(weights []byte, x []q81Block, rows, cols int) []float64 {
	blocksPerRow := cols / q2kSuperBlock
	out := make([]float64, rows)
	for row := 0; row < rows; row++ {
		for b := 0; b < blocksPerRow; b++ {
			off := (row*blocksPerRow + b) * q2kBlockBytes
			values := dequantizeQ2K(weights[off : off+q2kBlockBytes])
			for i, value := range values {
				out[row] += float64(value) * q81Value(x, b*q2kSuperBlock+i)
			}
		}
	}
	return out
}

func selectedQ2KQ81Reference(weights []byte, x []q81Block, ids []int, experts, rows, cols int) []float64 {
	const blockBytes = q2kBlockBytes
	blocksPerRow := cols / q2kSuperBlock
	perExpert := rows * blocksPerRow * blockBytes
	out := make([]float64, len(ids)*rows)
	for slot, expert := range ids {
		if expert < 0 || expert >= experts {
			for row := 0; row < rows; row++ {
				out[slot*rows+row] = math.NaN()
			}
			continue
		}
		start := expert * perExpert
		rowValues := q2KQ81Reference(weights[start:start+perExpert], x[slot*cols/32:(slot+1)*cols/32], rows, cols)
		copy(out[slot*rows:(slot+1)*rows], rowValues)
	}
	return out
}

func iq2XXSQ81Reference(weights []byte, x []q81Block, rows, cols int) []float64 {
	const blockBytes = iq2xxsBlockBytes
	blocksPerRow := cols / iq2xxsSuperBlock
	out := make([]float64, rows)
	w := make([]float32, cols)
	for row := 0; row < rows; row++ {
		off := row * blocksPerRow * blockBytes
		if err := egraph.DequantizeIQ2XXS(w, weights[off:off+blocksPerRow*blockBytes]); err != nil {
			panic(err)
		}
		for i, value := range w {
			out[row] += float64(value) * q81Value(x, i)
		}
	}
	return out
}

func TestQ8_1CPUReferencesUseFloat64AndExpertIndirection(t *testing.T) {
	const cols, rows = 512, 2
	weights := q2kTestBlocks(rows * cols / q2kSuperBlock)
	x := make([]q81Block, cols/32)
	for i := range x {
		x[i].d = 0x3800 // 0.5
		x[i].sum = 0x3c00
		for j := range x[i].qs {
			x[i].qs[j] = int8((i+j)%11 - 5)
		}
	}
	got := q2KQ81Reference(weights, x, rows, cols)
	if len(got) != rows || math.IsNaN(got[0]) || math.IsInf(got[0], 0) {
		t.Fatalf("q2_k q8_1 reference returned %#v", got)
	}
	other := append([]byte(nil), weights...)
	for i := range other {
		other[i] ^= 0x11
	}
	selected := selectedQ2KQ81Reference(append(weights, other...), x, []int{1, -1}, 2, rows, cols)
	if !math.IsNaN(selected[rows]) {
		t.Fatalf("invalid expert id did not produce NaN: %#v", selected)
	}
	if selected[0] == got[0] {
		t.Fatal("selected expert reference did not select a distinct expert")
	}
	q80 := make([]byte, rows*(cols/32)*34)
	for i := range q80 {
		q80[i] = byte(i*17 + 3)
	}
	q80got := q80Q81Reference(q80, x, rows, cols)
	if len(q80got) != rows || math.IsNaN(q80got[0]) || math.IsInf(q80got[0], 0) {
		t.Fatalf("q8_0 q8_1 reference returned %#v", q80got)
	}
	iq2got := iq2XXSQ81Reference(iq2xxsTestBlocks(rows*cols/iq2xxsSuperBlock), x, rows, cols)
	if len(iq2got) != rows || math.IsNaN(iq2got[0]) || math.IsInf(iq2got[0], 0) {
		t.Fatalf("iq2_xxs q8_1 reference returned %#v", iq2got)
	}
}
