package main

import (
	"bytes"
	"errors"
	"io"
	"testing"
)

func TestNativeSupportAdapterOriginalBytes(t *testing.T) {
	for _, cuda := range []bool{false, true} {
		for _, wrapped := range []bool{false, true} {
			v, frontier, generation := supportCodecFixture(t, cuda, wrapped)
			core := []byte{9, 2, 7, 0}
			var before, after bytes.Buffer
			if err := writeRetainedDSparkSupportContainer(&before, core, v, frontier, generation, 1<<30); err != nil {
				t.Fatal(err)
			}
			if err := writeDSparkSupportContainer(&after, core, v, frontier, generation, 1<<30); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(before.Bytes(), after.Bytes()) {
				t.Fatal("native adapter bytes differ from original writer")
			}
			gotCore, support, gotGeneration, err := readDSparkSupportContainer(bytes.NewReader(before.Bytes()), cuda, 1<<30)
			if err != nil || !bytes.Equal(gotCore, core) || (cuda && gotGeneration != generation) || (!cuda && gotGeneration != 0) {
				t.Fatal("native adapter container core or generation changed", err)
			}
			if _, err := readDSparkSupport(support, v, frontier, generation); err != nil {
				t.Fatal(err)
			}
			if err := writeDSparkSupportContainer(nil, core, v, frontier, generation, 1<<30); err == nil {
				t.Fatal("nil native container writer admitted")
			}
			if err := writeDSparkSupportContainer(supportCodecShortWriter{}, core, v, frontier, generation, 1<<30); !errors.Is(err, io.ErrShortWrite) {
				t.Fatal("native container short writer not rejected", err)
			}
			if _, _, _, err := readDSparkSupportContainer(bytes.NewReader(before.Bytes()), !cuda, 1<<30); err == nil {
				t.Fatal("foreign native container kind admitted")
			}
		}
	}
}

func TestNativeSupportAdapterCaptureAndExtent(t *testing.T) {
	v, frontier, generation := supportCodecFixture(t, true, false)
	var original bytes.Buffer
	if err := writeRetainedDSparkSupport(&original, v, frontier, generation); err != nil {
		t.Fatal(err)
	}
	common := 133 + 4*len(v.targets) + 56*len(v.journal.rows) + 4*len(v.history)
	for _, tc := range []struct {
		name   string
		change func([]byte) []byte
	}{
		{"truncated", func(b []byte) []byte { return b[:len(b)-1] }},
		{"trailing", func(b []byte) []byte { return append(b, 0) }},
		{"boolean", func(b []byte) []byte { b[common+24] = 2; return b }},
		{"generation", func(b []byte) []byte { b[common]++; return b }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data := tc.change(bytes.Clone(original.Bytes()))
			if _, err := readDSparkSupport(data, v, frontier, generation); err == nil {
				t.Fatal("invalid native adapter capture or extent admitted")
			}
		})
	}
}
