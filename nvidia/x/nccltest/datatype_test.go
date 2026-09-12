package nccltest

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"
	"time"
)

func ExampleDataType_Size() {
	fmt.Println(Float32.Size(), Float64.Size())
	// Output: 4 8
}

func ExampleCheck64() {
	c := Case{Operation: AllReduce, DataType: Float64, Bytes: 8}
	compared, wrong, err := Check64(c, 1, 0, []float64{7 + 1.0/(1<<32)})
	fmt.Println(compared, wrong, err)
	// Output: 1 0 <nil>
}

func TestFloat64Layout(t *testing.T) {
	for _, tt := range []struct {
		op   Operation
		want Layout
	}{
		{AllReduce, Layout{32, 32, 32, 0, 0}},
		{AllGather, Layout{10, 30, 10, 10, 0}},
		{ReduceScatter, Layout{30, 10, 10, 0, 10}},
		{AllToAll, Layout{30, 30, 10, 0, 0}},
		{AllToAllV, Layout{30, 30, 10, 0, 0}},
	} {
		t.Run(string(tt.op), func(t *testing.T) {
			c := Case{Operation: tt.op, DataType: Float64, Bytes: 260}
			got, err := c.Layout(3)
			if err != nil || got != tt.want {
				t.Fatalf("got %+v, %v; want %+v", got, err, tt.want)
			}
		})
	}
	for _, c := range []Case{
		{Operation: AllReduce, DataType: "unknown", Bytes: 64},
		{Operation: AllReduce, DataType: Float64, Bytes: 7},
		{Operation: AllGather, DataType: Float64, Bytes: 16},
	} {
		if _, err := c.Layout(3); err == nil {
			t.Fatalf("accepted %+v", c)
		}
	}
	c := Case{Operation: AllGather, DataType: Float64, Bytes: 260}
	alg, bus, err := c.Bandwidth(3, time.Microsecond)
	if err != nil || math.Abs(alg-.24) > 1e-15 || math.Abs(bus-.16) > 1e-15 {
		t.Fatalf("bandwidth = %g, %g, %v", alg, bus, err)
	}
}

func TestFloat64Reductions(t *testing.T) {
	// Fixed rank-zero-element payloads, including their low double bits.
	const eps = 1.0 / (1 << 32)
	const a, b, c = 7 + eps, 12 + 2*eps, 13 + 3*eps
	for _, tt := range []struct {
		reduction Reduction
		want      float64
	}{
		{Sum, a + b + c}, {Product, a * b * c}, {Minimum, a},
		{Maximum, c}, {Average, (a + b + c) / 3}, {PreMulSum, a + 2*b + c},
	} {
		t.Run(string(tt.reduction), func(t *testing.T) {
			c := Case{Operation: AllReduce, DataType: Float64, Bytes: 8, Reduction: tt.reduction}
			count, wrong, err := Check64(c, 3, 0, []float64{tt.want})
			if err != nil || count != 1 || wrong != 0 {
				t.Fatalf("correct: %d, %d, %v", count, wrong, err)
			}
			_, wrong, err = Check64(c, 3, 0, []float64{float64(float32(tt.want))})
			if err != nil || wrong != 1 {
				t.Fatalf("narrowed double accepted: wrong=%d err=%v", wrong, err)
			}
		})
	}
}

func TestFloatBuffers(t *testing.T) {
	for _, d := range []DataType{Float32, Float64} {
		t.Run(string(d), func(t *testing.T) {
			c := Case{Operation: AllReduce, DataType: d, Bytes: 32}
			count := c.Bytes / d.Size()
			data := inputBuffer(c, 1, 0, count)
			if len(data) != 32 {
				t.Fatalf("buffer bytes=%d", len(data))
			}
			if d == Float64 {
				if bits := binary.NativeEndian.Uint64(data); bits != 0x401c000000040000 {
					t.Fatalf("first double bits=%016x", bits)
				}
			} else if bits := binary.NativeEndian.Uint32(data); bits != 0x40e00000 {
				t.Fatalf("first float bits=%08x", bits)
			}
			n, wrong, err := CheckBytes(c, 1, 0, data)
			if err != nil || n != count || wrong != 0 {
				t.Fatalf("encoded: %d, %d, %v", n, wrong, err)
			}
			n, wrong, err = CheckBytes(c, 1, 0, poisonBuffer(c, count))
			if err != nil || n != count || wrong != count {
				t.Fatalf("poison: %d, %d, %v", n, wrong, err)
			}
		})
	}
}

func TestCheckerDatatypeMismatch(t *testing.T) {
	if _, _, err := Check(Case{Operation: AllReduce, DataType: Float64, Bytes: 8}, 1, 0, []float32{7}); err == nil {
		t.Fatal("Check accepted double case")
	}
	if _, _, err := Check64(Case{Operation: AllReduce, Bytes: 8}, 1, 0, []float64{7, 2}); err == nil {
		t.Fatal("Check64 accepted default float case")
	}
}

func TestBoundedProduct(t *testing.T) {
	// The first four inputs are 7, 12, 13, 7; the remaining 1,020 ranks
	// contribute 548 negative and 472 positive signs at element zero.
	// A zero output must fail even at the maximum supported rank count.
	for _, d := range []DataType{Float32, Float64} {
		c := Case{Operation: AllReduce, DataType: d, Bytes: d.Size(), Reduction: Product}
		data := make([]byte, d.Size())
		if d == Float32 {
			binary.NativeEndian.PutUint32(data, math.Float32bits(7644))
		} else {
			const eps = 1.0 / (1 << 32)
			binary.NativeEndian.PutUint64(data, math.Float64bits((7+eps)*(12+2*eps)*(13+3*eps)*(7+4*eps)))
		}
		if n, wrong, err := CheckBytes(c, 1024, 0, data); err != nil || n != 1 || wrong != 0 {
			t.Fatalf("%s bounded product: count=%d wrong=%d err=%v", d, n, wrong, err)
		}
		if n, wrong, err := CheckBytes(c, 1024, 0, make([]byte, d.Size())); err != nil || n != 1 || wrong != 1 {
			t.Fatalf("%s accepted zero product: count=%d wrong=%d err=%v", d, n, wrong, err)
		}
	}
}

func TestProductInputEncoding(t *testing.T) {
	for _, d := range []DataType{Float32, Float64} {
		c := Case{Operation: AllReduce, DataType: d, Reduction: Product}
		for _, rank := range []int{4, 13, 1023} {
			data := inputBuffer(c, 1024, rank, 64)
			for i := 0; i < 64; i++ {
				var value float64
				if d == Float32 {
					value = float64(math.Float32frombits(binary.NativeEndian.Uint32(data[i*4:])))
				} else {
					value = math.Float64frombits(binary.NativeEndian.Uint64(data[i*8:]))
				}
				if math.Abs(value) != 1 {
					t.Fatalf("%s rank %d element %d: factor %g", d, rank, i, value)
				}
			}
		}
	}
}

func TestFloat64RankRouting(t *testing.T) {
	c := Case{Operation: AllGather, DataType: Float64, Bytes: 96}
	l, err := c.Layout(3)
	if err != nil {
		t.Fatal(err)
	}
	// Element four comes from rank one, element zero.
	if got := expectedFloat[float64](c, l, 3, 2, 4); got != 12+2.0/(1<<32) {
		t.Fatalf("rank routing = %.17g", got)
	}
	for _, size := range []int{4, 8} {
		for rank := 0; rank < 7; rank++ {
			send, recv := 0, 0
			for peer := 0; peer < 7; peer++ {
				send += peerElements(20, 7, rank, peer, size)
				recv += peerElements(20, 7, peer, rank, size)
			}
			if send != recv {
				t.Fatalf("size=%d rank=%d send=%d recv=%d", size, rank, send, recv)
			}
		}
	}
}
