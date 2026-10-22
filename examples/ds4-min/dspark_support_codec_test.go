package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"reflect"
	"testing"
)

type supportCodecCounter struct{ n uint64 }

func (w *supportCodecCounter) Write(p []byte) (int, error) {
	w.n += uint64(len(p))
	return len(p), nil
}

type supportCodecShortWriter struct{}

func (supportCodecShortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

func TestSupportCodecCountAndShortWrite(t *testing.T) {
	for _, cuda := range []bool{false, true} {
		v, frontier, generation := supportCodecFixture(t, cuda, true)
		var counted supportCodecCounter
		var encoded bytes.Buffer
		core := []byte{1, 2, 3}
		if err := writeDSparkSupportContainer(&counted, core, v, frontier, generation, 1<<30); err != nil {
			t.Fatal(err)
		}
		if err := writeDSparkSupportContainer(&encoded, core, v, frontier, generation, 1<<30); err != nil {
			t.Fatal(err)
		}
		if counted.n != uint64(encoded.Len()) {
			t.Fatal("count and write differ")
		}
		if err := writeDSparkSupportContainer(supportCodecShortWriter{}, core, v, frontier, generation, 1<<30); !errors.Is(err, io.ErrShortWrite) {
			t.Fatalf("short write: %v", err)
		}
		gotCore, gotSupport, gotGeneration, err := readDSparkSupportContainer(&encoded, cuda, 1<<30)
		if err != nil || !bytes.Equal(core, gotCore) || (cuda && gotGeneration != generation) {
			t.Fatalf("container round trip: %v", err)
		}
		if _, err := readDSparkSupport(gotSupport, v, frontier, generation); err != nil {
			t.Fatal(err)
		}
	}
}

func supportCodecFixture(t *testing.T, cuda, wrapped bool) (*dsparkSupportEnvelope, int, uint64) {
	t.Helper()
	context, block := 3, 1
	if wrapped {
		context = Flash.SlidingWindow + 4
	}
	rows, values, err := dsparkFeatureJournalShape(context, block, 1, Flash.Embed)
	if err != nil {
		t.Fatal(err)
	}
	j, err := newDSparkFeatureJournal(rows, 1)
	if err != nil {
		t.Fatal(err)
	}
	frontier := 2
	if wrapped {
		frontier = rows + 1
	}
	for i := 0; i < frontier; i++ {
		if err := j.append(i, i+1, 1); err != nil {
			t.Fatal(err)
		}
	}
	v := &dsparkSupportEnvelope{model: [32]byte{1}, context: context, block: block, targets: []int{0}, journal: j, history: make([]float32, values)}
	for i := range v.history {
		v.history[i] = float32(i%11) / 3
	}
	if cuda {
		v.capture = &dsparkSupportCapture{generation: 7, position: frontier - 1, token: frontier, published: true, captured: []bool{true}, features: make([]float32, Flash.Embed)}
	}
	return v, frontier, 8
}

func TestSupportCodecRoundTrip(t *testing.T) {
	for _, test := range []struct {
		name          string
		cuda, wrapped bool
	}{{"metal", false, false}, {"metal-wrapped", false, true}, {"cuda", true, false}, {"cuda-wrapped", true, true}} {
		t.Run(test.name, func(t *testing.T) {
			v, frontier, generation := supportCodecFixture(t, test.cuda, test.wrapped)
			var encoded bytes.Buffer
			if err := writeDSparkSupport(&encoded, v, frontier, generation); err != nil {
				t.Fatal(err)
			}
			got, err := readDSparkSupport(encoded.Bytes(), v, frontier, generation)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(v, got) {
				t.Fatal("support history or capture metadata lost")
			}
			got.history[0]++
			got.targets[0]++
			got.journal.rows[0].token++
			if got.history[0] == v.history[0] || got.targets[0] == v.targets[0] || got.journal.rows[0].token == v.journal.rows[0].token {
				t.Fatal("decoded support aliases source")
			}
		})
	}
}

func TestSupportCodecMutations(t *testing.T) {
	v, frontier, generation := supportCodecFixture(t, true, false)
	var encoded bytes.Buffer
	if err := writeDSparkSupport(&encoded, v, frontier, generation); err != nil {
		t.Fatal(err)
	}
	commonBytes := 133 + 4*len(v.targets) + 56*len(v.journal.rows) + 4*len(v.history)
	for _, test := range []struct {
		name   string
		mutate func([]byte) []byte
	}{
		{"missing", func(b []byte) []byte { return nil }},
		{"truncated", func(b []byte) []byte { return b[:len(b)-1] }},
		{"trailing", func(b []byte) []byte { return append(b, 0) }},
		{"version", func(b []byte) []byte { b[7]++; return b }},
		{"platform", func(b []byte) []byte { b[8] = 2; return b }},
		{"model", func(b []byte) []byte { b[9]++; return b }},
		{"geometry", func(b []byte) []byte { b[41]++; return b }},
		{"selected-layer", func(b []byte) []byte { b[53]++; return b }},
		{"epoch", func(b []byte) []byte { b[65] = 0; return b }},
		{"prefix", func(b []byte) []byte { b[73]++; return b }},
		{"token", func(b []byte) []byte { binary.LittleEndian.PutUint64(b[145:], uint64(Flash.Vocab)); return b }},
		{"generation", func(b []byte) []byte { binary.LittleEndian.PutUint64(b[commonBytes:], 9); return b }},
		{"publication", func(b []byte) []byte { b[commonBytes+24] = 0; return b }},
		{"capture", func(b []byte) []byte { b[commonBytes+25] = 0; return b }},
		{"noncanonical-bool", func(b []byte) []byte { b[commonBytes+24] = 2; return b }},
	} {
		t.Run(test.name, func(t *testing.T) {
			data := test.mutate(append([]byte(nil), encoded.Bytes()...))
			if _, err := readDSparkSupport(data, v, frontier, generation); err == nil {
				t.Fatal("invalid support envelope accepted")
			}
		})
	}
}

func TestSupportCodecPlatformPadding(t *testing.T) {
	v, frontier, generation := supportCodecFixture(t, true, false)
	// CUDA actual snapshot validator does not impose Metal's whole-ring finite
	// rule. Preserve an unused physical slot's exact numeric bits.
	index := (len(v.journal.rows) - 1) * Flash.Embed
	v.history[index] = math.Float32frombits(0x7fc01234)
	var encoded bytes.Buffer
	if err := writeDSparkSupport(&encoded, v, frontier, generation); err != nil {
		t.Fatal(err)
	}
	got, err := readDSparkSupport(encoded.Bytes(), v, frontier, generation)
	if err != nil {
		t.Fatal(err)
	}
	if math.Float32bits(got.history[index]) != 0x7fc01234 {
		t.Fatal("CUDA unused padding bits changed")
	}
	v.capture = nil
	if err := writeDSparkSupport(&bytes.Buffer{}, v, frontier, generation); err == nil {
		t.Fatal("nonfinite Metal support ring accepted")
	}
}
