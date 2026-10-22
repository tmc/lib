package main

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// This scalar oracle decodes serialized operands and accumulates in float64.
// It shares no production lookup table, warp mapping, or matrix instruction.
// Activation rounding is covered separately by the literal C byte gate.
func mxfp4ScalarProducts(weights, activation []byte, cols, rows, experts, slots int, ids []int32) ([]float64, error) {
	if cols <= 0 || cols%256 != 0 || rows <= 0 || experts <= 0 || slots <= 0 || len(ids) == 0 || len(ids)%slots != 0 {
		return nil, fmt.Errorf("invalid scalar geometry")
	}
	inputRows := len(ids) / slots
	product := func(factors ...int) (int, error) {
		n := 1
		for _, factor := range factors {
			if factor <= 0 || n > math.MaxInt/factor {
				return 0, fmt.Errorf("scalar extent overflow")
			}
			n *= factor
		}
		return n, nil
	}
	weightBytes, err := product(experts, rows, cols/32, 17)
	if err != nil {
		return nil, err
	}
	activationBytes, err := product(inputRows, cols/256, 144)
	if err != nil {
		return nil, err
	}
	decodedBytes, err := product(inputRows, cols, 8)
	if err != nil {
		return nil, err
	}
	outputBytes, err := product(len(ids), rows, 8)
	if err != nil {
		return nil, err
	}
	if len(weights) != weightBytes || len(activation) != activationBytes {
		return nil, fmt.Errorf("incomplete scalar operands")
	}
	for _, id := range ids {
		if id < 0 || int(id) >= experts {
			return nil, fmt.Errorf("invalid scalar expert")
		}
	}
	value := func(code byte) float64 {
		table := [...]float64{0, .5, 1, 1.5, 2, 3, 4, 6}
		v := table[code&7]
		if code&8 != 0 {
			v = -v
		}
		return v
	}
	scale := func(e byte) float64 {
		if e == 0 {
			return math.Ldexp(1, -127)
		}
		return math.Ldexp(1, int(e)-127)
	}
	x := make([]float64, decodedBytes/8)
	for r := 0; r < inputRows; r++ {
		for k := 0; k < cols; k++ {
			block := activation[((k/256)*inputRows+r)*144:]
			group, lane := k%256/32, k%32
			e := block[(group/2)*4+group%2]
			if e == 255 {
				return nil, fmt.Errorf("nonfinite scalar activation scale")
			}
			code := block[16+group*16+lane%16] >> (4 * (lane / 16))
			x[r*cols+k] = value(code) * scale(e)
		}
	}
	out := make([]float64, outputBytes/8)
	for pair, id := range ids {
		input := x[(pair/slots)*cols:]
		for row := 0; row < rows; row++ {
			sum := 0.0
			for k := 0; k < cols; k += 32 {
				w := weights[((int(id)*rows+row)*(cols/32)+k/32)*17:]
				if w[0] == 255 {
					return nil, fmt.Errorf("nonfinite scalar scale")
				}
				dot := 0.0
				for j := 0; j < 32; j++ {
					code := w[1+j%16] >> (4 * (j / 16))
					dot += value(code) * input[k+j]
				}
				sum += scale(w[0]) * dot
			}
			out[pair*rows+row] = sum
		}
	}
	return out, nil
}

func TestMXFP4ScalarExtentOverflow(t *testing.T) {
	for _, tc := range []struct {
		name                       string
		cols, rows, experts, pairs int
	}{
		{"weight-experts", 256, 1, math.MaxInt, 1},
		{"weight-rows", 256, math.MaxInt, 1, 1},
		{"activation", math.MaxInt &^ 255, 1, 1, 8},
		{"decoded-allocation", math.MaxInt &^ 255, 1, 1, 1},
		{"output-allocation", 256, math.MaxInt / 136, 1, 32},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := mxfp4ScalarProducts(nil, nil, tc.cols, tc.rows, tc.experts, 1, make([]int32, tc.pairs))
			if err == nil || !strings.Contains(err.Error(), "extent overflow") {
				t.Fatalf("expected overflow refusal, got %v", err)
			}
		})
	}
}

func TestMXFP4ScalarSerializedProducts(t *testing.T) {
	// Two activation rows and two experts have deliberately different values.
	// Every nibble is one; E8M0 scale127 is one.
	weights := make([]byte, 2*8*17)
	for b := 0; b < len(weights)/17; b++ {
		weights[b*17] = 127
		code := byte(0x22)
		if b >= 8 {
			code = 0x33
		}
		for j := 1; j < 17; j++ {
			weights[b*17+j] = code
		}
	}
	activation := make([]byte, 2*144)
	for r := 0; r < 2; r++ {
		for g := 0; g < 8; g++ {
			activation[r*144+(g/2)*4+g%2] = 127
		}
		for j := 16; j < 144; j++ {
			activation[r*144+j] = byte(0x22 + r*0x22)
		}
	}
	for _, tc := range []struct {
		name  string
		ids   []int32
		slots int
		want  []float64
	}{
		{"paired", []int32{1, 0, 0, 1}, 2, []float64{384, 256, 512, 768}},
		{"down", []int32{1, 0}, 1, []float64{384, 512}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := mxfp4ScalarProducts(weights, activation, 256, 1, 2, tc.slots, tc.ids)
			if err != nil {
				t.Fatal(err)
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Fatalf("value%d: got %g want %g", i, got[i], tc.want[i])
				}
			}
		})
	}
	if _, err := mxfp4ScalarProducts(weights, activation[:len(activation)-1], 256, 1, 2, 1, []int32{1, 0}); err == nil {
		t.Fatal("accepted truncated activation")
	}
	if _, err := mxfp4ScalarProducts(weights, activation, 256, 1, 2, 1, []int32{-1, 0}); err == nil {
		t.Fatal("accepted invalid expert")
	}
	t.Run("signed-unequal-nibbles", func(t *testing.T) {
		data := append([]byte(nil), activation...)
		for j := 16; j < 144; j++ {
			data[j] = 0x32
			data[144+j] = 0xc4
		}
		got, err := mxfp4ScalarProducts(weights, data, 256, 1, 2, 1, []int32{1, 0})
		if err != nil || len(got) != 2 || got[0] != 480 || got[1] != 0 {
			t.Fatalf("got %v err %v", got, err)
		}
		data[0] = 255
		if _, err := mxfp4ScalarProducts(weights, data, 256, 1, 2, 1, []int32{1, 0}); err == nil {
			t.Fatal("accepted nonfinite activation scale")
		}
	})
	t.Run("zero-scale", func(t *testing.T) {
		data := append([]byte(nil), weights...)
		for b := 0; b < len(data)/17; b++ {
			data[b*17] = 0
		}
		got, err := mxfp4ScalarProducts(data, activation, 256, 1, 2, 1, []int32{1, 0})
		want := []float64{384 * math.Ldexp(1, -127), 512 * math.Ldexp(1, -127)}
		if err != nil || len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
			t.Fatalf("got %v err %v", got, err)
		}
	})
}
