package nccltest

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func ExampleCheckBytes_half() {
	c := Case{Operation: AllReduce, DataType: Float16, Bytes: 2}
	data := make([]byte, 2)
	binary.NativeEndian.PutUint16(data, 0xbc00) // The seeded rank-zero input is -1.
	count, wrong, err := CheckBytes(c, 1, 0, data)
	fmt.Println(count, wrong, err)
	// Output: 1 0 <nil>
}

func TestSmallAverageScale(t *testing.T) {
	// CUDA host-conversion fixture results for float32(1/ranks).
	for _, tt := range []struct {
		ranks int
		bits  [4]uint16
	}{
		{1, [4]uint16{0x3c00, 0x3f80, 0x38, 0x3c}},
		{2, [4]uint16{0x3800, 0x3f00, 0x30, 0x38}},
		{3, [4]uint16{0x3555, 0x3eab, 0x2b, 0x35}},
		{7, [4]uint16{0x3092, 0x3e12, 0x21, 0x31}},
		{255, [4]uint16{0x1c04, 0x3b81, 0x02, 0x1c}},
		{512, [4]uint16{0x1800, 0x3b00, 0x01, 0x18}},
		{513, [4]uint16{0x17fc, 0x3b00, 0x01, 0x18}},
		{1023, [4]uint16{0x1401, 0x3a80, 0x01, 0x14}},
		{1024, [4]uint16{0x1400, 0x3a80, 0x00, 0x14}},
	} {
		for i, d := range []DataType{Float16, BFloat16, Float8E4M3, Float8E5M2} {
			f, _ := d.smallFormat()
			if got := f.encode(smallAverageScale(d, tt.ranks)); got != tt.bits[i] {
				t.Fatalf("%s ranks=%d: bits=%x want=%x", d, tt.ranks, got, tt.bits[i])
			}
		}
	}
}

func TestSmallChecker(t *testing.T) {
	for _, d := range []DataType{Float16, BFloat16, Float8E4M3, Float8E5M2} {
		c := Case{Operation: AllReduce, DataType: d, Bytes: 64}
		count := c.Bytes / d.Size()
		data := inputBuffer(c, 1, 0, count)
		if n, wrong, err := CheckBytes(c, 1, 0, data); err != nil || n != count || wrong != 0 {
			t.Fatalf("%s input: %d %d %v", d, n, wrong, err)
		}
		data[0] ^= 1
		if _, wrong, err := CheckBytes(c, 1, 0, data); err != nil || wrong != 1 {
			t.Fatalf("%s corruption: %d %v", d, wrong, err)
		}
		if n, wrong, err := CheckBytes(c, 1, 0, poisonBuffer(c, count)); err != nil || n != count || wrong != count {
			t.Fatalf("%s poison: %d %d %v", d, n, wrong, err)
		}
		c.Reduction = Average
		l, _ := c.Layout(3)
		// At element one the two active ranks are both negative.
		if got := expectedSmall(c, l, 3, 0, 1); got != -2*smallAverageScale(d, 3) {
			t.Fatalf("%s average=%g", d, got)
		}
	}
}

func TestSmallReductionTrees(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, d := range []DataType{Float16, BFloat16, Float8E4M3, Float8E5M2} {
		f, _ := d.smallFormat()
		for _, reduction := range []Reduction{Sum, Product, Minimum, Maximum, Average, PreMulSum} {
			for _, ranks := range []int{1, 2, 3, 7, 32, 255, 512, 1024} {
				c := Case{Operation: AllReduce, DataType: d, Bytes: 64 * d.Size(), Reduction: reduction}
				l, _ := c.Layout(ranks)
				for _, index := range []int{0, 1, 33} {
					want := expectedSmall(c, l, ranks, 0, index)
					for tree := 0; tree < 3; tree++ {
						values := make([]float64, ranks)
						for rank := range values {
							x := smallInput(c, ranks, rank, index)
							if reduction == Average {
								x *= smallAverageScale(d, ranks)
							}
							if reduction == PreMulSum {
								x *= float64(1 + rank%2)
							}
							values[rank] = f.decode(f.encode(x))
							if values[rank] != x {
								t.Fatalf("%s %s pre-op is inexact", d, reduction)
							}
						}
						for len(values) > 1 {
							rng.Shuffle(len(values), func(i, j int) { values[i], values[j] = values[j], values[i] })
							next := values[:0]
							for i := 0; i < len(values); i += 2 {
								x := values[i]
								if i+1 < len(values) {
									y := values[i+1]
									switch reduction {
									case Product:
										x *= y
									case Minimum:
										x = math.Min(x, y)
									case Maximum:
										x = math.Max(x, y)
									default:
										x += y
									}
								}
								if rounded := f.decode(f.encode(x)); rounded != x {
									t.Fatalf("%s %s ranks=%d inexact intermediate %g -> %g", d, reduction, ranks, x, rounded)
								}
								next = append(next, x)
							}
							values = next
						}
						if values[0] != want {
							t.Fatalf("%s %s ranks=%d tree=%d got=%g want=%g", d, reduction, ranks, tree, values[0], want)
						}
					}
				}
			}
		}
	}
}

// Assemble receive buffers from sender byte slices, without using the checker’s
// sourceElement routing. This catches rank and element/byte offset mistakes.
func TestSmallCopyRouting(t *testing.T) {
	const ranks = 4
	for _, d := range []DataType{Float16, BFloat16, Float8E4M3, Float8E5M2} {
		for _, op := range []Operation{AllGather, Gather, Hypercube, AllToAll, Scatter, Broadcast, SendRecv} {
			t.Run(string(d)+"/"+string(op), func(t *testing.T) {
				c := Case{Operation: op, DataType: d, Bytes: 1024, Root: 3}
				l, err := c.Layout(ranks)
				if err != nil {
					t.Fatal(err)
				}
				var send [ranks][]byte
				for rank := range send {
					send[rank] = inputBuffer(c, ranks, rank, l.SendCount)
				}
				block := l.ParamCount * d.Size()
				for rank := 0; rank < ranks; rank++ {
					var got []byte
					switch op {
					case AllGather, Gather, Hypercube:
						if op != Gather || rank == c.Root {
							for source := range send {
								got = append(got, send[source]...)
							}
						}
					case AllToAll:
						for source := range send {
							got = append(got, send[source][rank*block:(rank+1)*block]...)
						}
					case Scatter:
						got = append(got, send[c.Root][rank*block:(rank+1)*block]...)
					case Broadcast:
						got = append(got, send[c.Root]...)
					case SendRecv:
						got = append(got, send[(rank+ranks-1)%ranks]...)
					}
					n, wrong, err := CheckBytes(c, ranks, rank, got)
					if err != nil || wrong != 0 || n != len(got)/d.Size() {
						t.Fatalf("rank=%d count=%d wrong=%d err=%v", rank, n, wrong, err)
					}
					if len(got) == 0 {
						continue
					}
					// Select distinct values: repeated FP8 encodings are legitimate.
					last := d.Size()
					for last < len(got) && getSmall(got[last:], d) == getSmall(got, d) {
						last += d.Size()
					}
					if last == len(got) {
						t.Fatal("routing control has no distinct values")
					}
					for i := 0; i < d.Size(); i++ {
						got[i], got[last+i] = got[last+i], got[i]
					}
					if _, wrong, err := CheckBytes(c, ranks, rank, got); err != nil || wrong == 0 {
						t.Fatalf("rank=%d accepted permutation: wrong=%d err=%v", rank, wrong, err)
					}
				}
			})
		}
	}
}

func TestSmallVariableRouting(t *testing.T) {
	// For four ranks and 1024 requested bytes, upstream spread-one traffic
	// assigns these 16-byte-aligned sizes by cyclic source/destination distance.
	bytesByDistance := [...]int{0, 160, 336, 512}
	for _, d := range []DataType{Float16, BFloat16, Float8E4M3, Float8E5M2} {
		c := Case{Operation: AllToAllV, DataType: d, Bytes: 1024}
		l, err := c.Layout(4)
		if err != nil {
			t.Fatal(err)
		}
		if l.SendCount*d.Size() != 1008 || l.RecvCount*d.Size() != 1008 {
			t.Fatalf("%s unexpected layout: %+v", d, l)
		}
		var received [4][]byte
		for source := 0; source < 4; source++ {
			data := inputBuffer(c, 4, source, l.SendCount)
			offset := 0
			for destination := 0; destination < 4; destination++ {
				n := bytesByDistance[(destination-source+4)%4]
				received[destination] = append(received[destination], data[offset:offset+n]...)
				offset += n
			}
			if offset != len(data) {
				t.Fatal("traffic does not consume sender buffer")
			}
		}
		for rank, data := range received {
			if n, wrong, err := CheckBytes(c, 4, rank, data); err != nil || n != l.RecvCount || wrong != 0 {
				t.Fatalf("%s rank=%d count=%d wrong=%d err=%v", d, rank, n, wrong, err)
			}
			if _, wrong, err := CheckBytes(c, 4, (rank+1)%4, data); err != nil || wrong == 0 {
				t.Fatalf("%s accepted wrong destination: wrong=%d err=%v", d, wrong, err)
			}
		}
	}
}
