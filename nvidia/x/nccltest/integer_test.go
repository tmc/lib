package nccltest

import (
	"encoding/binary"
	"fmt"
	"testing"
)

func ExampleCheckBytes() {
	c := Case{Operation: AllReduce, DataType: Uint64, Bytes: 8}
	data := make([]byte, 8)
	binary.NativeEndian.PutUint64(data, 0xe220a8397b1dcdaf)
	compared, wrong, err := CheckBytes(c, 1, 0, data)
	fmt.Println(compared, wrong, err)
	// Output: 1 0 <nil>
}

func TestIntegerReductions(t *testing.T) {
	// Fixed three-rank answers computed with arbitrary-precision integer
	// arithmetic, wrapping to the element width before division for average.
	for _, tt := range []struct {
		d    DataType
		want [6]uint64
	}{
		{Int8, [6]uint64{0xf3, 0xf5, 0xaf, 0x4f, 0xfc, 0xe8}},
		{Uint8, [6]uint64{0xf3, 0xf5, 0x4f, 0xf5, 0x51, 0xe8}},
		{Int32, [6]uint64{0x9ce078f3, 0x296759f5, 0x8009454f, 0x7b1dcdaf, 0xdef57da7, 0x3e99dee8}},
		{Uint32, [6]uint64{0x9ce078f3, 0x296759f5, 0x7b1dcdaf, 0xa1b965f5, 0x344ad2fb, 0x3e99dee8}},
		{Int64, [6]uint64{0x575da3bc9ce078f3, 0x08292c17296759f5, 0xe220a8397b1dcdaf, 0x6e789e6aa1b965f5, 0x1d1f3694344ad2fb, 0xc5d642273e99dee8}},
		{Uint64, [6]uint64{0x575da3bc9ce078f3, 0x08292c17296759f5, 0x06c45d188009454f, 0xe220a8397b1dcdaf, 0x1d1f3694344ad2fb, 0xc5d642273e99dee8}},
	} {
		t.Run(string(tt.d), func(t *testing.T) {
			for i, reduction := range []Reduction{Sum, Product, Minimum, Maximum, Average, PreMulSum} {
				c := Case{Operation: AllReduce, DataType: tt.d, Bytes: tt.d.Size(), Reduction: reduction}
				data := make([]byte, c.Bytes)
				putInteger(data, tt.d, tt.want[i])
				n, wrong, err := CheckBytes(c, 3, 0, data)
				if err != nil || n != 1 || wrong != 0 {
					t.Fatalf("%s: count=%d wrong=%d err=%v", reduction, n, wrong, err)
				}
				data[0] ^= 1
				_, wrong, err = CheckBytes(c, 3, 0, data)
				if err != nil || wrong != 1 {
					t.Fatalf("%s accepted bit corruption: wrong=%d err=%v", reduction, wrong, err)
				}
			}
		})
	}
}

func TestIntegerEncoding(t *testing.T) {
	for _, d := range []DataType{Int8, Uint8, Int32, Uint32, Int64, Uint64} {
		c := Case{Operation: AllReduce, DataType: d, Bytes: d.Size()}
		data := inputBuffer(c, 1, 0, 1)
		const first = uint64(0xe220a8397b1dcdaf)
		var full [8]byte
		binary.NativeEndian.PutUint64(full[:], first)
		var want []byte
		switch d.Size() {
		case 1:
			want = []byte{0xaf}
		case 4:
			want = make([]byte, 4)
			binary.NativeEndian.PutUint32(want, 0x7b1dcdaf)
		case 8:
			want = full[:]
		}
		if string(data) != string(want) {
			t.Fatalf("%s bytes=%x want=%x", d, data, want)
		}
		if got := getInteger(data, d); got != first&d.mask() {
			t.Fatalf("%s value=%x", d, got)
		}
		putInteger(data, d, ^uint64(0))
		for _, b := range data {
			if b != 0xff {
				t.Fatalf("%s truncated store=%x", d, data)
			}
		}
	}
}

func TestIntegerCheckerPreservesLowBits(t *testing.T) {
	c := Case{Operation: AllReduce, DataType: Uint64, Bytes: 8}
	const want = uint64(0xe220a8397b1dcdaf)
	if float64(want) != float64(want^1) {
		t.Fatal("control does not exercise float64 precision loss")
	}
	data := make([]byte, 8)
	binary.NativeEndian.PutUint64(data, want^1)
	if n, wrong, err := CheckBytes(c, 1, 0, data); err != nil || n != 1 || wrong != 1 {
		t.Fatalf("low-bit corruption: count=%d wrong=%d err=%v", n, wrong, err)
	}
}

func TestIntegerPoison(t *testing.T) {
	for _, d := range []DataType{Int8, Uint8, Int32, Uint32, Int64, Uint64} {
		for _, inPlace := range []bool{false, true} {
			c := Case{Operation: ReduceScatter, DataType: d, Bytes: 256, InPlace: inPlace}
			l, err := c.Layout(4)
			if err != nil {
				t.Fatal(err)
			}
			for rank := 0; rank < 4; rank++ {
				data := make([]byte, l.SendCount*d.Size())
				poisonIntegers(c, l, 4, rank, data)
				offset := 0
				if inPlace {
					offset = rank * l.RecvOffset * d.Size()
				}
				n, wrong, err := CheckBytes(c, 4, rank, data[offset:offset+l.RecvCount*d.Size()])
				if err != nil || n != l.RecvCount || wrong != n {
					t.Fatalf("%s rank=%d inPlace=%v count=%d wrong=%d err=%v", d, rank, inPlace, n, wrong, err)
				}
			}
		}
	}
}

func TestCheckBytesLength(t *testing.T) {
	for _, d := range []DataType{Int8, Uint64, Float32, Float64} {
		c := Case{Operation: AllReduce, DataType: d, Bytes: 16}
		for _, size := range []int{0, 15, 17} {
			if _, _, err := CheckBytes(c, 1, 0, make([]byte, size)); err == nil {
				t.Fatalf("%s accepted %d bytes", d, size)
			}
		}
	}
}

func TestIntegerRootAndRouting(t *testing.T) {
	// Fixed four-rank samples for a 256-byte request. These distinguish
	// datatype-scaled element offsets from byte offsets and use root three.
	for _, tt := range []struct {
		d    DataType
		want [7]uint64
	}{
		{Uint8, [7]uint64{0xf5, 0x77, 0x01, 0xd5, 0xed, 0xd0, 0xe0}},
		{Uint32, [7]uint64{0xa1b965f5, 0xf89e39e5, 0xef953637, 0x36ee7c87, 0x724c81ed, 0xa1adcf46, 0x0f2cfae0}},
		{Uint64, [7]uint64{0x6e789e6aa1b965f5, 0x145a11cac985644b, 0x6e73e372e2338acb, 0x973d0136e421c8db, 0xf88bb8a8724c81ed, 0xe2abc6f10d0705a2, 0x4fe95c650f2cfae0}},
	} {
		for _, sample := range []struct {
			op        Operation
			wantIndex int
			nextBlock bool
		}{
			{AllGather, 0, true}, {Gather, 0, true}, {Hypercube, 0, true},
			{AllToAll, 1, true}, {AllToAllV, 2, false}, {Scatter, 3, false},
			{Broadcast, 4, false}, {SendRecv, 0, false},
			{ReduceScatter, 5, false}, {Reduce, 6, false}, {AllReduce, 6, false},
		} {
			c := Case{Operation: sample.op, DataType: tt.d, Bytes: 256, Root: 3}
			l, err := c.Layout(4)
			if err != nil {
				t.Fatal(err)
			}
			rank, index := 2, 0
			if sample.op == Gather || sample.op == Reduce {
				rank = c.Root
			}
			if sample.nextBlock {
				index = l.ParamCount
			}
			if got := expectedInteger(c, l, 4, rank, index); got != tt.want[sample.wantIndex] {
				t.Fatalf("%s %s rank=%d index=%d: got=%x want=%x", tt.d, sample.op, rank, index, got, tt.want[sample.wantIndex])
			}
		}
	}
}

func TestGatherRankPermutation(t *testing.T) {
	for _, d := range []DataType{Int8, Uint8, Int32, Uint32, Int64, Uint64, Float16, BFloat16, Float8E4M3, Float8E5M2} {
		c := Case{Operation: Gather, DataType: d, Bytes: 1024, Root: 3}
		l, err := c.Layout(4)
		if err != nil {
			t.Fatal(err)
		}
		var data []byte
		for rank := 0; rank < 4; rank++ {
			data = append(data, inputBuffer(c, 4, rank, l.SendCount)...)
		}
		if n, wrong, err := CheckBytes(c, 4, 3, data); err != nil || n != l.RecvCount || wrong != 0 {
			t.Fatalf("%s correct gather: count=%d wrong=%d err=%v", d, n, wrong, err)
		}
		block := l.ParamCount * d.Size()
		for i := 0; i < block; i++ {
			data[i], data[3*block+i] = data[3*block+i], data[i]
		}
		if _, wrong, err := CheckBytes(c, 4, 3, data); err != nil || wrong == 0 {
			t.Fatalf("%s accepted rank permutation: wrong=%d err=%v", d, wrong, err)
		}
		if n, wrong, err := CheckBytes(c, 4, 1, nil); err != nil || n != 0 || wrong != 0 {
			t.Fatalf("%s non-root output: count=%d wrong=%d err=%v", d, n, wrong, err)
		}
	}
}
