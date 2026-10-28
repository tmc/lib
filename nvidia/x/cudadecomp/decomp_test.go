package cudadecomp

import (
	"bytes"
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// weights returns n BF16 values drawn from a zero-mean distribution with
// standard deviation 0.02, the scale of trained transformer weights.
func weights(n int, laplace bool, seed uint64) []uint16 {
	r := rand.New(rand.NewPCG(seed, 1))
	w := make([]uint16, n)
	for i := range w {
		var x float64
		if laplace {
			x = r.ExpFloat64() * 0.02 / math.Sqrt2
			if r.IntN(2) == 0 {
				x = -x
			}
		} else {
			x = r.NormFloat64() * 0.02
		}
		w[i] = uint16(math.Float32bits(float32(x)) >> 16)
	}
	return w
}

func TestLZ4RoundTrip(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 7))
	random := make([]byte, 70000)
	for i := range random {
		random[i] = byte(r.Uint32())
	}
	lowEntropy := make([]byte, 70000)
	for i := range lowEntropy {
		lowEntropy[i] = "abcd"[r.IntN(4)]
	}
	tests := []struct {
		name string
		data []byte
	}{
		{"empty", nil},
		{"one", []byte{1}},
		{"short", []byte("hello, world")},
		{"thirteen", bytes.Repeat([]byte{9}, 13)},
		{"run", bytes.Repeat([]byte{0}, 100000)},
		{"period 3", bytes.Repeat([]byte("xyz"), 30000)},
		{"long literals", random[:300]},
		{"random", random},
		{"low entropy", lowEntropy},
		{"text", bytes.Repeat([]byte("the quick brown fox jumps over the lazy dog. "), 2000)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			enc := lz4Compress(nil, tt.data)
			got := make([]byte, len(tt.data))
			if err := lz4Decompress(got, enc); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, tt.data) {
				t.Fatal("round trip mismatch")
			}
			if len(tt.data) > 1000 && tt.name == "run" && len(enc) > len(tt.data)/100 {
				t.Errorf("run of %d bytes compressed to %d", len(tt.data), len(enc))
			}
		})
	}
}

// TestLZ4KnownBlock decodes a block written by hand from the format
// description, so the decoder is not checked only against this package's
// encoder.
func TestLZ4KnownBlock(t *testing.T) {
	// "abcd" then a 10-byte match at offset 4, then literal "XYZ12".
	block := []byte{0x46, 'a', 'b', 'c', 'd', 0x04, 0x00, 0x50, 'X', 'Y', 'Z', '1', '2'}
	want := []byte("abcdabcdabcdabXYZ12")
	got := make([]byte, len(want))
	if err := lz4Decompress(got, block); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestLZ4Malformed(t *testing.T) {
	tests := []struct {
		name  string
		block []byte
		size  int
	}{
		{"empty", nil, 0},
		{"truncated literals", []byte{0x50, 'a', 'b'}, 5},
		{"zero offset", []byte{0x10, 'a', 0, 0, 0x00}, 6},
		{"offset before start", []byte{0x10, 'a', 2, 0, 0x00}, 6},
		{"match past end", []byte{0x1f, 'a', 1, 0, 200, 0x00}, 10},
		{"short output", []byte{0x30, 'a', 'b', 'c'}, 4},
		{"long output", []byte{0x30, 'a', 'b', 'c'}, 2},
		{"length overflow", append([]byte{0xf0}, bytes.Repeat([]byte{255}, 10)...), 100},
		{"missing offset", []byte{0x10, 'a', 1}, 6},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := lz4Decompress(make([]byte, tt.size), tt.block); err == nil {
				t.Error("lz4Decompress accepted malformed block")
			}
		})
	}
}

func FuzzLZ4Decompress(f *testing.F) {
	f.Add(lz4Compress(nil, []byte("abcabcabcabcabcabcabc")), 21)
	f.Add([]byte{0x46, 'a', 'b', 'c', 'd', 0x04, 0x00, 0x50, 'X', 'Y', 'Z', '1', '2'}, 19)
	f.Fuzz(func(t *testing.T, block []byte, size int) {
		if size < 0 || size > 1<<20 {
			return
		}
		lz4Decompress(make([]byte, size), block)
	})
}

func FuzzLZ4RoundTrip(f *testing.F) {
	f.Add([]byte("abcabcabcabcabcabcabcabcabc"))
	f.Fuzz(func(t *testing.T, data []byte) {
		got := make([]byte, len(data))
		if err := lz4Decompress(got, lz4Compress(nil, data)); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, data) {
			t.Fatal("mismatch")
		}
	})
}

func TestBF16RoundTrip(t *testing.T) {
	tests := []struct {
		name     string
		w        []uint16
		codec    Codec
		minRatio float64
	}{
		// The ratios are those this encoder reaches on these
		// distributions, less a margin; see TestBF16Ratio.
		{"gaussian deflate", weights(1<<18, false, 1), CodecDeflate, 1.35},
		{"laplace deflate", weights(1<<18, true, 2), CodecDeflate, 1.35},
		{"gaussian lz4", weights(1<<18, false, 3), CodecLZ4, 1.1},
		{"stored", weights(1000, false, 4), CodecStored, 1.0},
		{"odd tail", weights(DefaultChunkSize+3, false, 5), CodecLZ4, 1.0},
		{"zeros", make([]uint16, 100000), CodecLZ4, 50},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ct, err := CompressBF16("w", tt.w, tt.codec)
			if err != nil {
				t.Fatal(err)
			}
			got, err := DecompressBF16(context.Background(), ct)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, tt.w) {
				t.Fatal("round trip changed weights")
			}
			if r := ct.Ratio(); r < tt.minRatio {
				t.Errorf("ratio %.3f < %.2f", r, tt.minRatio)
			}
			for i, c := range ct.Chunks {
				if c.DstOffset < uint64(len(tt.w)) && c.DstOffset+uint64(c.DstSize) > uint64(len(tt.w)) {
					t.Errorf("chunk %d spans the planes", i)
				}
			}
		})
	}
}

// TestBF16Ratio reports what byte-plane splitting gains on Gaussian
// weights over compressing the interleaved bytes.
func TestBF16Ratio(t *testing.T) {
	w := weights(1<<18, false, 9)
	for _, codec := range []Codec{CodecLZ4, CodecDeflate} {
		split, err := CompressBF16("w", w, codec)
		if err != nil {
			t.Fatal(err)
		}
		inter := make([]byte, 2*len(w))
		for i, v := range w {
			inter[2*i], inter[2*i+1] = byte(v), byte(v>>8)
		}
		flat, err := Compress("w", inter, codec, DefaultChunkSize)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%v: split %.3f, interleaved %.3f", codec, split.Ratio(), flat.Ratio())
		if split.Ratio() < flat.Ratio() {
			t.Errorf("%v: split ratio %.3f below interleaved %.3f", codec, split.Ratio(), flat.Ratio())
		}
	}
}

func TestSplitBytePlanes(t *testing.T) {
	w := []uint16{0x3f80, 0xbf00, 0x0001, 0xff7f}
	p := SplitBytePlanes(w)
	want := []byte{0x3f, 0xbf, 0x00, 0xff, 0x80, 0x00, 0x01, 0x7f}
	if !bytes.Equal(p, want) {
		t.Errorf("SplitBytePlanes = % x, want % x", p, want)
	}
	if got := MergeBytePlanes(p); !slices.Equal(got, w) {
		t.Errorf("MergeBytePlanes = %x, want %x", got, w)
	}
}

// TestCorruption flips one payload byte, or one checksum bit, in every
// chunk in turn, and requires ErrChunkCorrupted each time.
func TestCorruption(t *testing.T) {
	w := weights(3*DefaultChunkSize/2, false, 11)
	for _, codec := range []Codec{CodecStored, CodecLZ4, CodecDeflate} {
		ct, err := CompressBF16("w", w, codec)
		if err != nil {
			t.Fatal(err)
		}
		for i, c := range ct.Chunks {
			for _, where := range []string{"payload", "crc"} {
				bad := ct
				bad.Payload = slices.Clone(ct.Payload)
				bad.Chunks = slices.Clone(ct.Chunks)
				if where == "payload" {
					bad.Payload[c.SrcOffset+uint64(c.SrcSize)/2] ^= 0x10
				} else {
					bad.Chunks[i].CRC32 ^= 1
				}
				got, err := DecompressBF16(context.Background(), bad)
				if !errors.Is(err, ErrChunkCorrupted) {
					t.Errorf("%v chunk %d %s: err = %v, want ErrChunkCorrupted", codec, i, where, err)
				}
				if got != nil {
					t.Errorf("%v chunk %d %s: returned weights", codec, i, where)
				}
			}
		}
	}
}

func TestDecompressStructural(t *testing.T) {
	ct, err := CompressBF16("w", weights(DefaultChunkSize, false, 12), CodecLZ4)
	if err != nil {
		t.Fatal(err)
	}
	mutants := []struct {
		name string
		mut  func(*Tensor)
	}{
		{"dropped chunk", func(t *Tensor) { t.Chunks = t.Chunks[1:] }},
		{"payload range", func(t *Tensor) { t.Chunks[0].SrcSize = uint32(len(t.Payload)) + 1 }},
		{"destination range", func(t *Tensor) { t.Chunks[1].DstOffset = uint64(t.Size) }},
		{"codec", func(t *Tensor) { t.Chunks[0].Codec = 9 }},
	}
	for _, m := range mutants {
		t.Run(m.name, func(t *testing.T) {
			bad := ct
			bad.Chunks = slices.Clone(ct.Chunks)
			m.mut(&bad)
			if _, err := bad.Decompress(context.Background()); !errors.Is(err, ErrChunkCorrupted) {
				t.Errorf("err = %v, want ErrChunkCorrupted", err)
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ct.Decompress(ctx); !errors.Is(err, context.Canceled) {
		t.Errorf("canceled: err = %v", err)
	}
	if _, err := DecompressBF16(context.Background(), Tensor{Layout: LayoutBytes}); err == nil {
		t.Error("DecompressBF16 accepted byte layout")
	}
}

func TestCompressErrors(t *testing.T) {
	if _, err := Compress("x", []byte{1}, CodecLZ4, 0); err == nil {
		t.Error("chunk size 0 accepted")
	}
	if _, err := Compress("x", []byte{1}, 7, 16); err == nil {
		t.Error("codec 7 accepted")
	}
}
