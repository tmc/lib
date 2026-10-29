package cudadecomp

import (
	"bytes"
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestBatchParams(t *testing.T) {
	ct, err := CompressBF16("w", weights(2*DefaultChunkSize, false, 21), CodecLZ4)
	if err != nil {
		t.Fatal(err)
	}
	const src, dst, act = 0x10000000, 0x20000000, 0x30000000
	params, copies, err := ct.BatchParams(src, dst, act)
	if err != nil {
		t.Fatal(err)
	}
	if len(params)+len(copies) != len(ct.Chunks) {
		t.Fatalf("%d params + %d copies, want %d chunks", len(params), len(copies), len(ct.Chunks))
	}
	if len(params) == 0 || len(copies) == 0 {
		t.Fatalf("want both compressed and stored chunks: %d params, %d copies", len(params), len(copies))
	}
	p, c := 0, 0
	covered := 0
	for _, ch := range ct.Chunks {
		covered += int(ch.DstSize)
		if ch.Codec == CodecStored {
			want := Copy{Src: src + uintptr(ch.SrcOffset), Dst: dst + uintptr(ch.DstOffset), Size: int(ch.DstSize)}
			if copies[c] != want {
				t.Errorf("copy %d = %+v, want %+v", c, copies[c], want)
			}
			c++
			continue
		}
		want := driver.MemDecompressParams{
			SrcNumBytes: uint(ch.SrcSize),
			DstNumBytes: uint(ch.DstSize),
			DstActBytes: act + uintptr(8*p),
			Src:         src + uintptr(ch.SrcOffset),
			Dst:         dst + uintptr(ch.DstOffset),
			Algo:        driver.MemDecompressAlgorithmLZ4,
		}
		if params[p] != want {
			t.Errorf("param %d = %+v, want %+v", p, params[p], want)
		}
		p++
	}
	if covered != ct.Size {
		t.Errorf("chunks cover %d of %d bytes", covered, ct.Size)
	}

	bad := ct
	bad.Chunks = slices.Clone(ct.Chunks)
	bad.Chunks[0].SrcOffset = uint64(len(ct.Payload))
	if _, _, err := bad.BatchParams(src, dst, act); !errors.Is(err, ErrChunkCorrupted) {
		t.Errorf("out-of-range chunk: err = %v", err)
	}
}

func TestCheckDevice(t *testing.T) {
	ct, err := CompressBF16("w", weights(DefaultChunkSize, false, 22), CodecDeflate)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		mask   driver.MemDecompressAlgorithm
		maxLen int
		ok     bool
	}{
		{"deflate", driver.MemDecompressAlgorithmDeflate, 1 << 20, true},
		{"all", driver.MemDecompressAlgorithmDeflate | driver.MemDecompressAlgorithmLZ4, 1 << 20, true},
		{"lz4 only", driver.MemDecompressAlgorithmLZ4, 1 << 20, false},
		{"unsupported", driver.MemDecompressUnsupported, 1 << 20, false},
		{"too long", driver.MemDecompressAlgorithmDeflate, 1024, false},
	}
	for _, tt := range tests {
		if err := ct.CheckDevice(tt.mask, tt.maxLen); (err == nil) != tt.ok {
			t.Errorf("%s: CheckDevice = %v, want ok=%v", tt.name, err, tt.ok)
		}
	}
}

func TestReadOps(t *testing.T) {
	ct, err := CompressBF16("w", weights(3*DefaultChunkSize/2, false, 23), CodecLZ4)
	if err != nil {
		t.Fatal(err)
	}
	ops, slot := ct.ReadOps(1<<20, 2)
	if slot%ReadAlign != 0 {
		t.Errorf("slot size %d not aligned", slot)
	}
	for i, op := range ops {
		c := ct.Chunks[i]
		if op.FileOffset != 1<<20+int64(c.SrcOffset) || op.Size != int(c.SrcSize) || op.BufOffset != int64(i%2)*slot {
			t.Errorf("op %d = %+v", i, op)
		}
		if int64(op.Size) > slot {
			t.Errorf("op %d: %d bytes exceed slot %d", i, op.Size, slot)
		}
	}
}

func TestSchedule(t *testing.T) {
	for _, n := range []int{1, 2, 3, 10} {
		if err := VerifySchedule(Schedule(n), n, 2); err != nil {
			t.Errorf("n=%d: %v", n, err)
		}
	}
	if Schedule(0) != nil {
		t.Error("Schedule(0) not empty")
	}
	steps := Schedule(4)
	mutants := []struct {
		name  string
		steps []Step
		slots int
	}{
		{"one slot", steps, 1},
		{"decompress with read", []Step{
			{0, 0, -1}, {1, 1, 0}, {2, 2, 1}, {3, 3, 2}, {-1, -1, 3},
		}, 2},
		{"consume with decompress", func() []Step {
			s := slices.Clone(steps)
			s[2].Consume, s[1].Consume = -1, 0
			return s
		}(), 2},
		{"chunk dropped", steps[:len(steps)-1], 2},
		{"chunk twice", append(slices.Clone(steps), Step{Read: 0, Decompress: -1, Consume: -1}), 2},
		// Reading two chunks ahead lets chunk 2 overwrite slot 0 while
		// chunk 0 is still waiting for decompression.
		{"read ahead", []Step{
			{0, -1, -1}, {1, -1, -1}, {2, 0, -1}, {3, 1, 0}, {-1, 2, 1}, {-1, 3, 2}, {-1, -1, 3},
		}, 2},
	}
	for _, m := range mutants {
		if err := VerifySchedule(m.steps, 4, m.slots); err == nil {
			t.Errorf("%s: VerifySchedule accepted mutant", m.name)
		}
	}
}

func TestRun(t *testing.T) {
	w := weights(5*DefaultChunkSize/2, true, 24)
	ct, err := CompressBF16("w", w, CodecLZ4)
	if err != nil {
		t.Fatal(err)
	}
	const base = 4096
	file := append(make([]byte, base), ct.Payload...)
	out := make([]byte, ct.Size)
	next := 0
	err = Run(context.Background(), bytes.NewReader(file), base, ct, func(c Chunk, b []byte) error {
		if c != ct.Chunks[next] {
			t.Errorf("chunk %d delivered out of order", next)
		}
		next++
		copy(out[c.DstOffset:], b)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(MergeBytePlanes(out), w) {
		t.Fatal("pipeline output differs from input")
	}

	file[base+int(ct.Chunks[2].SrcOffset)+5] ^= 0xff
	err = Run(context.Background(), bytes.NewReader(file), base, ct, func(Chunk, []byte) error { return nil })
	if !errors.Is(err, ErrChunkCorrupted) {
		t.Errorf("corrupt chunk: err = %v", err)
	}

	stop := errors.New("stop")
	err = Run(context.Background(), bytes.NewReader(file), base, ct, func(Chunk, []byte) error { return stop })
	if !errors.Is(err, stop) {
		t.Errorf("consumer error: err = %v", err)
	}
}
