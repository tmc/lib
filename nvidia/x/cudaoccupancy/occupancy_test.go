package cudaoccupancy_test

import (
	_ "embed"
	"encoding/hex"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

//go:embed testdata/kernels.cubin.hex
var testCubinHex string

func loadTestCubin(t *testing.T) []byte {
	t.Helper()
	clean := strings.TrimSpace(testCubinHex)
	cubin, err := hex.DecodeString(clean)
	if err != nil {
		t.Fatalf("decode cubin hex: %v", err)
	}
	return cubin
}

func TestInspectCubin(t *testing.T) {
	cubin := loadTestCubin(t)
	kernels, err := cudaoccupancy.InspectCubin(cubin)
	if err != nil {
		t.Fatalf("InspectCubin: %v", err)
	}

	wantKernels := map[string]struct {
		registers int
		sharedMem int
		const0    int
	}{
		"compute":                 {registers: 8, sharedMem: 1024, const0: 912},
		"vector_add":              {registers: 12, sharedMem: 1024, const0: 924},
		"ds4_swiglu":              {registers: 16, sharedMem: 1024, const0: 924},
		"ds4_rmsnorm":             {registers: 30, sharedMem: 1168, const0: 932},
		"ds4_smem_stage_matvec":   {registers: 33, sharedMem: 1152, const0: 928},
		"ds4_moe_expert_dispatch": {registers: 38, sharedMem: 1024, const0: 940},
		"ds4_iq2_scalar":          {registers: 27, sharedMem: 1024, const0: 928},
		"ds4_iq2_simd":            {registers: 18, sharedMem: 1024, const0: 928},
		"ds4_q8_0_matvec":         {registers: 40, sharedMem: 1024, const0: 928},
	}

	found := make(map[string]cudaoccupancy.KernelResource)
	for _, k := range kernels {
		found[k.Name] = k
	}

	for name, want := range wantKernels {
		k, ok := found[name]
		if !ok {
			t.Errorf("kernel %q not found in CUBIN", name)
			continue
		}
		if k.Registers != want.registers {
			t.Errorf("kernel %q registers = %d, want %d", name, k.Registers, want.registers)
		}
		if k.StaticSharedMem != want.sharedMem {
			t.Errorf("kernel %q sharedMem = %d, want %d", name, k.StaticSharedMem, want.sharedMem)
		}
		if want.const0 > 0 && k.ConstantBankSize != want.const0 {
			t.Errorf("kernel %q constantBankSize = %d, want %d", name, k.ConstantBankSize, want.const0)
		}
	}
}

func TestCalculateOccupancySweepGB10(t *testing.T) {
	// Ground truth values measured directly against NVIDIA GB10 (sm_121) hardware
	// via cuOccupancyMaxActiveBlocksPerMultiprocessor.
	tests := []struct {
		name         string
		blockSize    int
		dynSmem      int
		wantBlocks   int
		wantWarps    int
		wantOcc      float64
		wantLimiting string
	}{
		{name: "block_32", blockSize: 32, wantBlocks: 24, wantWarps: 24, wantOcc: 0.50, wantLimiting: cudaoccupancy.LimitBlocks},
		{name: "block_64", blockSize: 64, wantBlocks: 24, wantWarps: 48, wantOcc: 1.00, wantLimiting: cudaoccupancy.LimitNone},
		{name: "block_128", blockSize: 128, wantBlocks: 12, wantWarps: 48, wantOcc: 1.00, wantLimiting: cudaoccupancy.LimitNone},
		{name: "block_256", blockSize: 256, wantBlocks: 6, wantWarps: 48, wantOcc: 1.00, wantLimiting: cudaoccupancy.LimitNone},
		{name: "block_512", blockSize: 512, wantBlocks: 3, wantWarps: 48, wantOcc: 1.00, wantLimiting: cudaoccupancy.LimitNone},
		{name: "block_1024", blockSize: 1024, wantBlocks: 1, wantWarps: 32, wantOcc: 32.0 / 48.0, wantLimiting: cudaoccupancy.LimitBlockSize},
	}

	kernel := cudaoccupancy.KernelResource{
		Name:            "ds4_q8_0_matvec",
		Registers:       40,
		StaticSharedMem: 1024,
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rep, err := kernel.Calculate("sm_121", tt.blockSize, tt.dynSmem)
			if err != nil {
				t.Fatalf("Calculate: %v", err)
			}
			if rep.ActiveBlocks != tt.wantBlocks {
				t.Errorf("activeBlocks = %d, want %d", rep.ActiveBlocks, tt.wantBlocks)
			}
			if rep.TheoreticalWarps != tt.wantWarps {
				t.Errorf("theoreticalWarps = %d, want %d", rep.TheoreticalWarps, tt.wantWarps)
			}
			if math.Abs(rep.Occupancy-tt.wantOcc) > 1e-4 {
				t.Errorf("occupancy = %.4f, want %.4f", rep.Occupancy, tt.wantOcc)
			}
			if rep.LimitingFactor != tt.wantLimiting {
				t.Errorf("limitingFactor = %q, want %q", rep.LimitingFactor, tt.wantLimiting)
			}
		})
	}
}

func TestSharedMemoryLimiting(t *testing.T) {
	// Dynamic shared memory sweep on ds4_q8_0_matvec (blockSize 256, 8 warps/block)
	// Measured ground truth on GB10.
	tests := []struct {
		dynSmem      int
		wantBlocks   int
		wantWarps    int
		wantLimiting string
	}{
		{dynSmem: 0, wantBlocks: 6, wantWarps: 48, wantLimiting: cudaoccupancy.LimitNone},
		{dynSmem: 4096, wantBlocks: 6, wantWarps: 48, wantLimiting: cudaoccupancy.LimitNone},
		{dynSmem: 8192, wantBlocks: 6, wantWarps: 48, wantLimiting: cudaoccupancy.LimitNone},
		{dynSmem: 16384, wantBlocks: 5, wantWarps: 40, wantLimiting: cudaoccupancy.LimitSharedMemory},
		{dynSmem: 24576, wantBlocks: 4, wantWarps: 32, wantLimiting: cudaoccupancy.LimitSharedMemory},
		{dynSmem: 32768, wantBlocks: 3, wantWarps: 24, wantLimiting: cudaoccupancy.LimitSharedMemory},
		{dynSmem: 40960, wantBlocks: 2, wantWarps: 16, wantLimiting: cudaoccupancy.LimitSharedMemory},
		{dynSmem: 48000, wantBlocks: 2, wantWarps: 16, wantLimiting: cudaoccupancy.LimitSharedMemory},
	}

	kernel := cudaoccupancy.KernelResource{
		Name:            "ds4_q8_0_matvec",
		Registers:       40,
		StaticSharedMem: 1024,
	}

	for _, tt := range tests {
		rep, err := kernel.Calculate("sm_121", 256, tt.dynSmem)
		if err != nil {
			t.Fatalf("dynSmem %d: %v", tt.dynSmem, err)
		}
		if rep.ActiveBlocks != tt.wantBlocks {
			t.Errorf("dynSmem %d: activeBlocks = %d, want %d", tt.dynSmem, rep.ActiveBlocks, tt.wantBlocks)
		}
		if rep.TheoreticalWarps != tt.wantWarps {
			t.Errorf("dynSmem %d: theoreticalWarps = %d, want %d", tt.dynSmem, rep.TheoreticalWarps, tt.wantWarps)
		}
		if rep.LimitingFactor != tt.wantLimiting {
			t.Errorf("dynSmem %d: limitingFactor = %q, want %q", tt.dynSmem, rep.LimitingFactor, tt.wantLimiting)
		}
	}
}

func TestRegisterLimiting(t *testing.T) {
	// A kernel with 128 registers per thread
	heavyRegs := cudaoccupancy.KernelResource{
		Name:      "heavy_reg_kernel",
		Registers: 128,
	}

	// On Hopper SM90 (64 warps/SM, 65536 registers):
	// block size 256 (8 warps): 8 * (32 * 128) = 32768 registers/block.
	// 65536 / 32768 = 2 blocks = 16 warps (out of 64). Occupancy = 25%.
	rep, err := heavyRegs.Calculate("sm_90", 256, 0)
	if err != nil {
		t.Fatalf("Calculate: %v", err)
	}
	if rep.ActiveBlocks != 2 {
		t.Errorf("activeBlocks = %d, want 2", rep.ActiveBlocks)
	}
	if rep.TheoreticalWarps != 16 {
		t.Errorf("theoreticalWarps = %d, want 16", rep.TheoreticalWarps)
	}
	if rep.LimitingFactor != cudaoccupancy.LimitRegisters {
		t.Errorf("limitingFactor = %q, want %q", rep.LimitingFactor, cudaoccupancy.LimitRegisters)
	}
}

func TestBlockSizeValidation(t *testing.T) {
	kernel := cudaoccupancy.KernelResource{Registers: 32}

	for _, invalidSize := range []int{0, -1, 1025, 2048} {
		rep, err := kernel.Calculate("sm_90", invalidSize, 0)
		if err != nil {
			t.Fatalf("size %d: %v", invalidSize, err)
		}
		if rep.TheoreticalWarps != 0 || rep.Occupancy != 0.0 {
			t.Errorf("size %d: expected 0 occupancy, got %v", invalidSize, rep)
		}
		if rep.LimitingFactor != cudaoccupancy.LimitBlockSize {
			t.Errorf("size %d: expected LimitBlockSize, got %q", invalidSize, rep.LimitingFactor)
		}
	}
}

func TestLookupDevice(t *testing.T) {
	tests := []struct {
		input    string
		wantArch string
		wantErr  bool
	}{
		{input: "sm_90", wantArch: "sm_90"},
		{input: "SM_90", wantArch: "sm_90"},
		{input: "sm90", wantArch: "sm_90"},
		{input: "sm_121", wantArch: "sm_121"},
		{input: "sm_121a", wantArch: "sm_121"},
		{input: "sm_80", wantArch: "sm_80"},
		{input: "sm_70", wantArch: "sm_70"},
		{input: "invalid_arch", wantErr: true},
	}

	for _, tt := range tests {
		dev, err := cudaoccupancy.LookupDevice(tt.input)
		if (err != nil) != tt.wantErr {
			t.Errorf("LookupDevice(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
			continue
		}
		if !tt.wantErr && dev.Architecture != tt.wantArch {
			t.Errorf("LookupDevice(%q) arch = %q, want %q", tt.input, dev.Architecture, tt.wantArch)
		}
	}
}

func TestParseCuobjdumpResUsage(t *testing.T) {
	// Sample real output from `cuobjdump -res-usage`
	cuobjdumpOutput := `
Resource usage:
 Common:
  GLOBAL:0
 Function _Z18test_kernel_nosmemPfS_i:
  REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:548 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z11test_kernelPfS_i:
  REG:12 STACK:0 SHARED:5120 LOCAL:0 CONSTANT[0]:548 TEXTURE:0 SURFACE:0 SAMPLER:0
`
	parsed, err := cudaoccupancy.ParseCuobjdumpResUsage(cuobjdumpOutput)
	if err != nil {
		t.Fatalf("ParseCuobjdumpResUsage: %v", err)
	}

	want := map[string]cudaoccupancy.KernelResource{
		"_Z18test_kernel_nosmemPfS_i": {
			Name:             "_Z18test_kernel_nosmemPfS_i",
			Registers:        10,
			StackFrameBytes:  0,
			StaticSharedMem:  0,
			ConstantBankSize: 548,
		},
		"_Z11test_kernelPfS_i": {
			Name:             "_Z11test_kernelPfS_i",
			Registers:        12,
			StackFrameBytes:  0,
			StaticSharedMem:  5120,
			ConstantBankSize: 548,
		},
	}

	if len(parsed) != len(want) {
		t.Fatalf("parsed %d kernels, want %d", len(parsed), len(want))
	}

	for name, w := range want {
		got, ok := parsed[name]
		if !ok {
			t.Errorf("kernel %q not found in parsed output", name)
			continue
		}
		if got.Registers != w.Registers {
			t.Errorf("%s: Registers = %d, want %d", name, got.Registers, w.Registers)
		}
		if got.StaticSharedMem != w.StaticSharedMem {
			t.Errorf("%s: StaticSharedMem = %d, want %d", name, got.StaticSharedMem, w.StaticSharedMem)
		}
		if got.ConstantBankSize != w.ConstantBankSize {
			t.Errorf("%s: ConstantBankSize = %d, want %d", name, got.ConstantBankSize, w.ConstantBankSize)
		}
		if got.StackFrameBytes != w.StackFrameBytes {
			t.Errorf("%s: StackFrameBytes = %d, want %d", name, got.StackFrameBytes, w.StackFrameBytes)
		}
	}
}

func BenchmarkInspectCubin(b *testing.B) {
	cubin := loadTestCubin(&testing.T{})
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := cudaoccupancy.InspectCubin(cubin)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCalculate(b *testing.B) {
	kernel := cudaoccupancy.KernelResource{
		Name:            "ds4_q8_0_matvec",
		Registers:       40,
		StaticSharedMem: 1024,
	}
	dev, _ := cudaoccupancy.LookupDevice("sm_121")
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = kernel.CalculateDevice(dev, 256, 16384)
	}
}

