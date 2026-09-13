package cudaprof_test

import (
	"bytes"
	"compress/gzip"
	"context"
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

func TestSelectedSampleTypes(t *testing.T) {
	for _, tt := range []struct {
		name       string
		names      []string
		wantNames  [][2]string
		wantValues []uint64
		wantErr    bool
	}{
		{"time", []string{"gpu_time"}, [][2]string{{"gpu_time", "nanoseconds"}}, []uint64{23}, false},
		{"reordered", []string{"gpu_time", "dram_read"}, [][2]string{{"gpu_time", "nanoseconds"}, {"dram_read", "bytes"}}, []uint64{23, 3}, false},
		{"unknown", []string{"dram_read", "missing"}, [][2]string{{"gpu_time", "nanoseconds"}}, []uint64{23}, true},
		{"duplicate", []string{"dram_read", "dram_read"}, [][2]string{{"gpu_time", "nanoseconds"}}, []uint64{23}, true},
		{"reset", nil, [][2]string{
			{"samples", "count"}, {"pc_samples", "count"}, {"vram_alloc", "bytes"}, {"dram_read", "bytes"},
			{"dram_write", "bytes"}, {"l2_sectors_read", "count"}, {"l2_sectors_miss", "count"},
			{"shared_mem", "bytes"}, {"bank_conflicts", "count"}, {"tensor_warps", "count"},
			{"active_warps", "count"}, {"eligible_warps", "count"}, {"issued_warps", "count"},
			{"inst_executed", "count"}, {"reg_spill", "bytes"}, {"interconnect_tx", "bytes"},
			{"interconnect_rx", "bytes"}, {"stall_scoreboard", "count"}, {"stall_wait", "count"},
			{"stall_math", "count"}, {"stall_mem", "count"}, {"launch_delay", "nanoseconds"},
			{"queue_delay", "nanoseconds"}, {"gpu_time", "nanoseconds"},
		}, []uint64{1, 24, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			s := cudaprof.NewSession(&buf)
			if err := s.SetSampleTypes("gpu_time"); err != nil {
				t.Fatal(err)
			}
			s.RecordDetailedSample(context.Background(), "kernel", "kernel.cu", 1, 0, cudaprof.SampleValues{
				Samples: 1, PCSamples: 24, VRAMAllocBytes: 2, DRAMReadBytes: 3, DRAMWriteBytes: 4,
				L2SectorsRead: 5, L2SectorsMiss: 6, SharedMemBytes: 7, BankConflicts: 8,
				TensorWarps: 9, ActiveWarps: 10, EligibleWarps: 11, IssuedWarps: 12,
				InstExecuted: 13, RegSpillBytes: 14, InterconnectTxBytes: 15, InterconnectRxBytes: 16,
				StallScoreboard: 17, StallWait: 18, StallMath: 19, StallMem: 20,
				LaunchDelayNanos: 21, QueueDelayNanos: 22, GPUNanos: 23,
			})
			if err := s.SetSampleTypes(tt.names...); (err != nil) != tt.wantErr {
				t.Fatalf("SetSampleTypes = %v, want error %v", err, tt.wantErr)
			}
			if err := s.Close(); err != nil {
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
			var got [][2]string
			for _, data := range fields[1] {
				v, _ := profileFields(t, data)
				got = append(got, [2]string{string(fields[6][v[1][0]]), string(fields[6][v[2][0]])})
			}
			if !reflect.DeepEqual(got, tt.wantNames) {
				t.Fatalf("types = %v, want %v", got, tt.wantNames)
			}
			if len(fields[2]) != 1 {
				t.Fatalf("got %d samples, want 1", len(fields[2]))
			}
			v, _ := profileFields(t, fields[2][0])
			if !reflect.DeepEqual(v[2], tt.wantValues) {
				t.Fatalf("values = %v, want %v", v[2], tt.wantValues)
			}
		})
	}
}

func TestProfileSampleSchema(t *testing.T) {
	b := cudaprof.NewBuilder()
	b.AddDetailedSample("kernel", "kernel.cu", 1, 0, cudaprof.SampleValues{
		Samples: 1, PCSamples: 24, VRAMAllocBytes: 2, DRAMReadBytes: 3, DRAMWriteBytes: 4,
		L2SectorsRead: 5, L2SectorsMiss: 6, SharedMemBytes: 7, BankConflicts: 8,
		TensorWarps: 9, ActiveWarps: 10, EligibleWarps: 11, IssuedWarps: 12,
		InstExecuted: 13, RegSpillBytes: 14, InterconnectTxBytes: 15, InterconnectRxBytes: 16,
		StallScoreboard: 17, StallWait: 18, StallMath: 19, StallMem: 20,
		LaunchDelayNanos: 21, QueueDelayNanos: 22, GPUNanos: 23,
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
	want := [][2]string{
		{"samples", "count"}, {"pc_samples", "count"}, {"vram_alloc", "bytes"}, {"dram_read", "bytes"},
		{"dram_write", "bytes"}, {"l2_sectors_read", "count"}, {"l2_sectors_miss", "count"},
		{"shared_mem", "bytes"}, {"bank_conflicts", "count"}, {"tensor_warps", "count"},
		{"active_warps", "count"}, {"eligible_warps", "count"}, {"issued_warps", "count"},
		{"inst_executed", "count"}, {"reg_spill", "bytes"}, {"interconnect_tx", "bytes"},
		{"interconnect_rx", "bytes"}, {"stall_scoreboard", "count"}, {"stall_wait", "count"},
		{"stall_math", "count"}, {"stall_mem", "count"}, {"launch_delay", "nanoseconds"},
		{"queue_delay", "nanoseconds"}, {"gpu_time", "nanoseconds"},
	}
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
	if !reflect.DeepEqual(v[2], []uint64{1, 24, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}) {
		t.Fatalf("sample values = %v", v[2])
	}
}

func TestSourceOnlyFrames(t *testing.T) {
	var buf bytes.Buffer
	s := cudaprof.NewSession(&buf)
	s.RecordStackSample(context.Background(), cudaprof.SampleValues{GPUNanos: 10}, []cudaprof.Frame{
		{FunctionName: "kernel", FileName: "kernel.sass", Address: cudaprof.DefaultMappingStart + 0x1000},
		{FunctionName: "caller", FileName: "main.go", Line: 10},
	})
	if err := s.Close(); err != nil {
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
	if len(fields[4]) != 2 {
		t.Fatalf("locations = %d", len(fields[4]))
	}
	leaf, _ := profileFields(t, fields[4][0])
	caller, _ := profileFields(t, fields[4][1])
	if !reflect.DeepEqual(leaf[2], []uint64{1}) || len(caller[2]) != 0 {
		t.Fatalf("mapping IDs: leaf=%v caller=%v", leaf[2], caller[2])
	}
}

func TestRecordSASSMetric(t *testing.T) {
	for _, tt := range []struct {
		name            string
		metric          cudaprof.SASSMetric
		wantTensorWarps uint64
		wantSamples     uint64
	}{
		{
			name: "non-tensor scalar fma",
			metric: cudaprof.SASSMetric{
				InstructionPC: cudaprof.DefaultMappingStart + 0x100,
				Opcode:        "FFMA",
				StallReason:   "stall_wait",
				IssuedWarps:   4,
				ExecutedWarps: 4,
			},
			wantTensorWarps: 0,
			wantSamples:     4,
		},
		{
			name: "tensor hmma instruction",
			metric: cudaprof.SASSMetric{
				InstructionPC: cudaprof.DefaultMappingStart + 0x200,
				Opcode:        "HMMA.16816.F32",
				IssuedWarps:   8,
				ExecutedWarps: 8,
			},
			wantTensorWarps: 8,
			wantSamples:     8,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			s := cudaprof.NewSession(&buf)
			s.RecordSASSMetric(context.Background(), "kernel", "kernel.cu", 1, tt.metric)
			if err := s.Close(); err != nil {
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
			if len(fields[2]) != 1 {
				t.Fatalf("got %d samples, want 1", len(fields[2]))
			}
			v, _ := profileFields(t, fields[2][0])
			// profileSampleTypes order: samples(0), pc_samples(1), vram_alloc(2),
			// dram_read(3), dram_write(4), l2_sectors_read(5), l2_sectors_miss(6),
			// shared_mem(7), bank_conflicts(8), tensor_warps(9), ...
			if v[2][0] != tt.wantSamples {
				t.Errorf("samples = %d, want %d", v[2][0], tt.wantSamples)
			}
			if v[2][9] != tt.wantTensorWarps {
				t.Errorf("tensor_warps = %d, want %d", v[2][9], tt.wantTensorWarps)
			}
		})
	}
}
