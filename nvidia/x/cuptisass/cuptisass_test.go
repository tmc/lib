package cuptisass

import (
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestVersionQualification(t *testing.T) {
	tests := []struct {
		name    string
		version uint32
		want    bool
	}{
		{name: "CUDA 12.6 API version", version: 24, want: true},
		{name: "CUDA 12.8 API version", version: 26, want: true},
		{name: "CUDA 12.9 API version", version: 28, want: true},
		{name: "CUDA 13.0 API version", version: 130001, want: true},
		{name: "CUDA 13.1 API version", version: 130101, want: true},
		{name: "CUDA 13.2 header version", version: 130200, want: true},
		{name: "CUDA 13.2 runtime version", version: 130202, want: true},
		{name: "CUDA 13.3 API version", version: 130301, want: true},
		{name: "CUDA 13.4 API version", version: 130400, want: true},
		{name: "zero version", version: 0, want: false},
		{name: "unmeasured 130100", version: 130100, want: false},
		{name: "unmeasured 130201", version: 130201, want: false},
		{name: "unmeasured 130500", version: 130500, want: false},
		{name: "unmeasured CUDA 14", version: 140000, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := qualifiedVersion(tt.version); got != tt.want {
				t.Errorf("qualifiedVersion(%d) = %v, want %v", tt.version, got, tt.want)
			}
		})
	}
}

func TestOutputGranularity(t *testing.T) {
	tests := []struct {
		gran OutputGranularity
		want string
	}{
		{GranularityGPU, "GPU"},
		{GranularitySM, "SM"},
		{GranularitySMSP, "SMSP"},
		{OutputGranularity(99), "Unknown"},
	}

	for _, tt := range tests {
		if got := tt.gran.String(); got != tt.want {
			t.Errorf("OutputGranularity(%d).String() = %q, want %q", tt.gran, got, tt.want)
		}
	}
}

func TestNewSessionValidation(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr bool
	}{
		{
			name:    "empty metrics",
			cfg:     Config{DeviceIndex: 0, Metrics: nil},
			wantErr: true,
		},
		{
			name: "invalid granularity",
			cfg: Config{
				DeviceIndex: 0,
				Metrics: []MetricConfig{
					{MetricID: 100, Granularity: OutputGranularity(5)},
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewSession(0, tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Fatalf("NewSession() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCensusOracle(t *testing.T) {
	dis := &cudasass.Disassembly{
		Functions: []*cudasass.Function{
			{
				Name: "_Z8VectorAddPKiS0_Pii",
				Instructions: []cudasass.Instruction{
					{Address: 0x0, Opcode: "MOV"},
					{Address: 0x10, Opcode: "LDG.E"},
					{Address: 0x20, Opcode: "IADD3"},
					{Address: 0x30, Opcode: "STG.E"},
					{Address: 0x40, Opcode: "RET"},
				},
			},
			{
				Name: "_Z13VectorMultiplyPKiS0_Pii",
				Instructions: []cudasass.Instruction{
					{Address: 0x0, Opcode: "MOV"},
					{Address: 0x10, Opcode: "IMAD"},
					{Address: 0x20, Opcode: "RET"},
				},
			},
		},
	}

	census := InstructionCensus(dis)
	if got, want := census["_Z8VectorAddPKiS0_Pii"], 5; got != want {
		t.Errorf("census VectorAdd = %d, want %d", got, want)
	}
	if got, want := census["_Z13VectorMultiplyPKiS0_Pii"], 3; got != want {
		t.Errorf("census VectorMultiply = %d, want %d", got, want)
	}

	offsets, err := InstructionOffsets(dis, "_Z8VectorAddPKiS0_Pii")
	if err != nil {
		t.Fatalf("InstructionOffsets error: %v", err)
	}
	if len(offsets) != 5 || offsets[0] != 0x0 || offsets[4] != 0x40 {
		t.Errorf("unexpected offsets: %v", offsets)
	}

	validRecords := []Record{
		{FunctionName: "_Z8VectorAddPKiS0_Pii", PCOffset: 0x0},
		{FunctionName: "_Z8VectorAddPKiS0_Pii", PCOffset: 0x10},
		{FunctionName: "_Z8VectorAddPKiS0_Pii", PCOffset: 0x20},
		{FunctionName: "_Z8VectorAddPKiS0_Pii", PCOffset: 0x30},
		{FunctionName: "_Z8VectorAddPKiS0_Pii", PCOffset: 0x40},
	}
	if err := VerifyCensus(validRecords, dis, "_Z8VectorAddPKiS0_Pii"); err != nil {
		t.Errorf("VerifyCensus expected success on valid records, got: %v", err)
	}

	invalidRecords := []Record{
		{FunctionName: "_Z8VectorAddPKiS0_Pii", PCOffset: 0x0},
		{FunctionName: "_Z8VectorAddPKiS0_Pii", PCOffset: 0x999}, // invalid PC offset
	}
	if err := VerifyCensus(invalidRecords, dis, "_Z8VectorAddPKiS0_Pii"); err == nil {
		t.Error("VerifyCensus expected error for invalid PC offset, got nil")
	}

	if err := VerifyCensus(validRecords, dis, "NonExistentKernel"); err == nil {
		t.Error("VerifyCensus expected error for missing kernel, got nil")
	}
}

func TestReadCString(t *testing.T) {
	if got := readCString(0); got != "" {
		t.Errorf("readCString(0) = %q, want empty", got)
	}

	buf := []byte("hello_sass\x00")
	ptr := uintptr(unsafe.Pointer(&buf[0]))
	if got, want := readCString(ptr), "hello_sass"; got != want {
		t.Errorf("readCString = %q, want %q", got, want)
	}
}
