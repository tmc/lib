package bitopt

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// The NVFP4 decode in moe.cu: eight E2M1 codes per word, code 2r+h
// (register r, half h) is sign<<3 | magnitude, and register r of the
// decode is the bf16x2 pair (h 0, h 1) with the magnitude at bits 6-8 and
// the sign at bit 15 of each half.

// Bit positions of the old packing: register r's magnitude at oldMag[r] and
// its sign at oldSign[r], plus 16 for the high half.
var (
	oldMag  = [4]int{6, 3, 0, 9}
	oldSign = [4]int{15, 12, 14, 13}
)

// dequant11 is moe.cu's decode of the old packing, 11 instructions.
func dequant11(q uint32) [4]uint32 {
	const m, mag, sgn = 0x81C081C0, 0x01C001C0, 0x80008000
	return [4]uint32{
		q & m,
		q << 3 & m,
		q<<6&mag | q<<1&sgn,
		q>>3&mag | q<<2&sgn,
	}
}

// codeFields returns the decode's output registers for code word x (code j
// at bits 4j..4j+3): the bits Solution.Pack places.
func codeFields(x uint32) []uint32 {
	f := make([]uint32, 4)
	for r := range 4 {
		for h := range 2 {
			c := x >> (4 * (2*r + h)) & 15
			f[r] |= (c&7)<<(6+16*h) | (c>>3)<<(15+16*h)
		}
	}
	return f
}

func packOld(x uint32) uint32 {
	var q uint32
	for r := range 4 {
		for h := range 2 {
			c := x >> (4 * (2*r + h)) & 15
			q |= (c&7)<<(oldMag[r]+16*h) | (c>>3)<<(oldSign[r]+16*h)
		}
	}
	return q
}

// A packTable packs a code word by bytes: for a packing linear in the bits
// of the code word, pack(x) is the OR over bytes i of t[i][x>>8i&0xff].
type packTable [4][256]uint32

func newPackTable(pack func(uint32) uint32) *packTable {
	t := new(packTable)
	for i := range 4 {
		for v := range 256 {
			t[i][v] = pack(uint32(v) << (8 * i))
		}
	}
	return t
}

func (t *packTable) pack(x uint32) uint32 {
	return t[0][x&0xff] | t[1][x>>8&0xff] | t[2][x>>16&0xff] | t[3][x>>24]
}

// TestNVFP4Exhaustive checks Solve's 9-instruction program against moe.cu's
// dequant11 for all 2^32 code words: the program decodes its packing of
// every word to the registers dequant11 decodes from the old packing.
func TestNVFP4Exhaustive(t *testing.T) {
	sol, err := Solve(nvfp4, 9)
	if err != nil {
		t.Fatal(err)
	}
	// The unit code words decide both maps (each output bit is one input
	// bit), so check them in -short mode too.
	newPack := func(x uint32) uint32 { return sol.Pack(codeFields(x))[0] }
	for e := range 32 {
		x := uint32(1) << e
		got := sol.Eval([]uint32{newPack(x)})
		if want := dequant11(packOld(x)); [4]uint32(got) != want {
			t.Fatalf("code bit %d: got %#08x, want %#08x", e, got, want)
		}
	}
	if testing.Short() {
		t.Skip("skipping 2^32 words in -short mode")
	}
	told, tnew := newPackTable(packOld), newPackTable(newPack)
	var bad atomic.Uint64
	var first atomic.Uint64
	first.Store(1 << 32)
	n := uint64(runtime.GOMAXPROCS(0))
	var wg sync.WaitGroup
	for w := range n {
		wg.Go(func() {
			q := make([]uint32, 1)
			v := make([]uint32, len(sol.Moves))
			for x := w; x < 1<<32; x += n {
				q[0] = tnew.pack(uint32(x))
				for i, m := range sol.Moves {
					v[i] = m.eval(q)
				}
				want := dequant11(told.pack(uint32(x)))
				for k, ps := range sol.Parts {
					var o uint32
					for _, p := range ps {
						o |= v[p.Move] & p.Mask
					}
					if o != want[k] {
						if bad.Add(1) == 1 {
							first.Store(x)
						}
						break
					}
				}
			}
		})
	}
	wg.Wait()
	if bad.Load() != 0 {
		t.Fatalf("%d code words differ, first %#08x", bad.Load(), first.Load())
	}
}
