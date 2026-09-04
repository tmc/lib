package cudaprof_test

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"io"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// profileFields decodes the varint and length-delimited fields used by pprof.
func profileFields(t *testing.T, data []byte) (map[int][]uint64, map[int][][]byte) {
	t.Helper()
	ints := make(map[int][]uint64)
	blobs := make(map[int][][]byte)
	read := func() uint64 {
		v, n := binary.Uvarint(data)
		if n <= 0 {
			t.Fatal("invalid protobuf varint")
		}
		data = data[n:]
		return v
	}
	for len(data) > 0 {
		tag := read()
		field := int(tag >> 3)
		switch tag & 7 {
		case 0:
			ints[field] = append(ints[field], read())
		case 2:
			n := read()
			if n > uint64(len(data)) {
				t.Fatal("truncated protobuf field")
			}
			blobs[field] = append(blobs[field], data[:n])
			data = data[n:]
		default:
			t.Fatalf("unexpected wire type %d", tag&7)
		}
	}
	return ints, blobs
}

func TestProfileSampleSchema(t *testing.T) {
	b := cudaprof.NewBuilder()
	b.AddDetailedSample("kernel", "kernel.cu", 1, 0, cudaprof.SampleValues{
		Samples: 1, VRAMAllocBytes: 2, DRAMReadBytes: 3, DRAMWriteBytes: 4,
		SharedMemBytes: 5, TensorWarps: 6, StallScoreboard: 7, StallWait: 8, GPUNanos: 9,
	}, nil)
	var buf bytes.Buffer
	if err := b.WriteGzip(&buf); err != nil {
		t.Fatal(err)
	}
	r, err := gzip.NewReader(&buf)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	data, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	_, fields := profileFields(t, data)
	want := [][2]string{{"samples", "count"}, {"vram_alloc", "bytes"},
		{"dram_read", "bytes"}, {"dram_write", "bytes"}, {"shared_mem", "bytes"},
		{"tensor_warps", "count"}, {"stall_scoreboard", "count"},
		{"stall_wait", "count"}, {"gpu_time", "nanoseconds"}}
	var got [][2]string
	for _, data := range fields[1] {
		v, _ := profileFields(t, data)
		got = append(got, [2]string{string(fields[6][v[1][0]]), string(fields[6][v[2][0]])})
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sample types = %v, want %v", got, want)
	}
	if len(fields[2]) != 1 {
		t.Fatalf("got %d samples, want 1", len(fields[2]))
	}
	v, _ := profileFields(t, fields[2][0])
	if !reflect.DeepEqual(v[2], []uint64{1, 2, 3, 4, 5, 6, 7, 8, 9}) {
		t.Fatalf("sample values = %v", v[2])
	}
}
