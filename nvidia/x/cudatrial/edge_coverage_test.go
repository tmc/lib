package cudatrial

import (
	"bytes"
	"testing"
)

func TestEdgeBitmap_PerThreadHashXOR(t *testing.T) {
	m, err := NewThreadEdgeMap(32, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, step := range []struct {
		thread int
		block  uint64
		want   int
	}{
		{0, 5, 21}, {1, 5, 21}, {0, 7, 21}, {1, 9, 27}, {0, 7, 20},
	} {
		i, err := m.Record(step.thread, step.block)
		if err != nil || i != step.want {
			t.Fatalf("record(%d,%d) = %d,%v; want %d", step.thread, step.block, i, err, step.want)
		}
	}
	b := m.Snapshot()
	if b.counts[21] != 3 || b.counts[27] != 1 || b.counts[20] != 1 {
		t.Fatal(b.counts)
	}
	before := m.Snapshot().Bytes()
	if _, err := m.Record(2, 1); err == nil {
		t.Fatal("invalid thread accepted")
	}
	if !bytes.Equal(before, m.Snapshot().Bytes()) {
		t.Fatal("invalid record changed map")
	}
	m.Reset()
	i, _ := m.Record(0, 7)
	if i != 23 || m.Snapshot().counts[27] != 0 {
		t.Fatal("reset retained predecessor or counts")
	}
	if b.counts[27] != 1 {
		t.Fatal("snapshot aliased reset")
	}
}

func TestEdgeBitmapQuantization(t *testing.T) {
	b, _ := NewEdgeBitmap(256)
	for i := range b.counts {
		b.counts[i] = byte(i)
	}
	q := b.Quantized()
	for _, c := range []struct{ n, want byte }{{0, 0}, {1, 1}, {2, 2}, {3, 2}, {4, 4}, {7, 4}, {8, 8}, {15, 8}, {16, 16}, {31, 16}, {32, 32}, {63, 32}, {64, 64}, {127, 64}, {128, 128}, {255, 128}} {
		if q[c.n] != c.want {
			t.Errorf("%d: got %d want %d", c.n, q[c.n], c.want)
		}
	}
	q[3] = 0
	if b.Bytes()[3] != 3 {
		t.Fatal("quantization aliases raw counters")
	}
}

func TestEdgeBitmapMergeAndSaturation(t *testing.T) {
	m, _ := NewThreadEdgeMap(32, 1)
	for i := 0; i < 300; i++ {
		m.Record(0, 0)
	}
	a := m.Snapshot()
	if a.counts[16] != 255 {
		t.Fatal("counter wrapped")
	}
	b, _ := NewEdgeBitmap(32)
	b.counts[17] = 4
	if err := b.Merge(a); err != nil {
		t.Fatal(err)
	}
	if b.counts[16] != 255 || b.counts[17] != 4 {
		t.Fatal("merge did not preserve union")
	}
	before := b.Bytes()
	c, _ := NewEdgeBitmap(34)
	if err := b.Merge(c); err == nil {
		t.Fatal("mismatched map accepted")
	}
	if !bytes.Equal(before, b.Bytes()) {
		t.Fatal("failed merge changed receiver")
	}
	for _, size := range []int{-1, 0, 1, 3, 16<<20 + 2} {
		if _, err := NewEdgeBitmap(size); err == nil {
			t.Fatalf("size %d accepted", size)
		}
	}
}

func FuzzThreadEdgeMap(f *testing.F) {
	f.Add([]byte{1, 5, 0, 7, 1, 9})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 4096 {
			return
		}
		m, _ := NewThreadEdgeMap(32, 2)
		var previous [2]uint64
		var want [32]byte
		for i := 0; i+1 < len(data); i += 2 {
			thread := int(data[i] % 2)
			block := uint64(data[i+1])
			edge := 16 + int((previous[thread]^block)%16)
			if want[edge] < 255 {
				want[edge]++
			}
			previous[thread] = block >> 1
			got, err := m.Record(thread, block)
			if err != nil || got != edge {
				t.Fatal(got, err, edge)
			}
		}
		if !bytes.Equal(m.Snapshot().Bytes(), want[:]) {
			t.Fatal("edge map mismatch")
		}
	})
}
