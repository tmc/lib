package cudaoccupancy_test

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

func TestReadKernelInfo_Cubin(t *testing.T) {
	cubin := loadTestCubin(t)
	infos, err := cudaoccupancy.ReadKernelInfo(bytes.NewReader(cubin))
	if err != nil {
		t.Fatalf("ReadKernelInfo: %v", err)
	}

	wantKernels := map[string]struct {
		registers int
		sharedMem int
		frameSize int
	}{
		"compute":                 {registers: 8, sharedMem: 1024, frameSize: 0},
		"vector_add":              {registers: 12, sharedMem: 1024, frameSize: 0},
		"ds4_swiglu":              {registers: 16, sharedMem: 1024, frameSize: 0},
		"ds4_rmsnorm":             {registers: 30, sharedMem: 1168, frameSize: 0},
		"ds4_smem_stage_matvec":   {registers: 33, sharedMem: 1152, frameSize: 0},
		"ds4_moe_expert_dispatch": {registers: 38, sharedMem: 1024, frameSize: 0},
		"ds4_iq2_scalar":          {registers: 27, sharedMem: 1024, frameSize: 0},
		"ds4_iq2_simd":            {registers: 18, sharedMem: 1024, frameSize: 0},
		"ds4_q8_0_matvec":         {registers: 40, sharedMem: 1024, frameSize: 0},
	}

	found := make(map[string]cudaoccupancy.KernelResourceInfo)
	for _, info := range infos {
		found[info.Name] = info
	}

	for name, want := range wantKernels {
		info, ok := found[name]
		if !ok {
			t.Errorf("kernel %q not found in CUBIN", name)
			continue
		}
		if info.Registers != want.registers || info.RegCount != want.registers {
			t.Errorf("kernel %q registers = %d (RegCount=%d), want %d", name, info.Registers, info.RegCount, want.registers)
		}
		if info.SharedMemBytes != want.sharedMem {
			t.Errorf("kernel %q sharedMem = %d, want %d", name, info.SharedMemBytes, want.sharedMem)
		}
		if info.FrameSizeBytes != want.frameSize || info.FrameSize != want.frameSize {
			t.Errorf("kernel %q frameSize = %d, want %d", name, info.FrameSizeBytes, want.frameSize)
		}
	}
}

func TestReadKernelInfo_SynthesizedELF(t *testing.T) {
	// Synthesize an ELF with .nv.info.<name> attributes:
	// EIATTR_REGCOUNT (0x2f) = 48
	// EIATTR_FRAME_SIZE (0x11) = 256
	// EIATTR_MAXREG_COUNT (0x1b) = 64
	// EIATTR_REQ_NTID (0x10) = [128, 2, 1]
	// EIATTR_SHARED_SIZE_SYSMEM (0x30) = 4096
	var nvinfo bytes.Buffer

	// RegCount: 4 bytes
	nvinfo.Write([]byte{0x04, cudaoccupancy.EIATTR_REGCOUNT, 0x04, 0x00})
	binary.Write(&nvinfo, binary.LittleEndian, uint32(48))

	// FrameSize: 4 bytes
	nvinfo.Write([]byte{0x04, cudaoccupancy.EIATTR_FRAME_SIZE, 0x04, 0x00})
	binary.Write(&nvinfo, binary.LittleEndian, uint32(256))

	// MaxRegCount: 4 bytes
	nvinfo.Write([]byte{0x04, cudaoccupancy.EIATTR_MAXREG_COUNT, 0x04, 0x00})
	binary.Write(&nvinfo, binary.LittleEndian, uint32(64))

	// ReqNTID: 12 bytes [128, 2, 1]
	nvinfo.Write([]byte{0x04, cudaoccupancy.EIATTR_REQ_NTID, 0x0c, 0x00})
	binary.Write(&nvinfo, binary.LittleEndian, uint32(128))
	binary.Write(&nvinfo, binary.LittleEndian, uint32(2))
	binary.Write(&nvinfo, binary.LittleEndian, uint32(1))

	// SharedScratch: 4 bytes
	nvinfo.Write([]byte{0x04, cudaoccupancy.EIATTR_SHARED_SIZE_SYSMEM, 0x04, 0x00})
	binary.Write(&nvinfo, binary.LittleEndian, uint32(4096))

	sections := map[string][]byte{
		".text.synth_kernel":    make([]byte, 64),
		".nv.info.synth_kernel": nvinfo.Bytes(),
	}

	elfBytes := synthesizeTestELF(sections)
	infos, err := cudaoccupancy.ReadKernelInfo(bytes.NewReader(elfBytes))
	if err != nil {
		t.Fatalf("ReadKernelInfo: %v", err)
	}

	if len(infos) != 1 {
		t.Fatalf("got %d kernels, want 1", len(infos))
	}

	k := infos[0]
	if k.Name != "synth_kernel" {
		t.Errorf("kernel Name = %q, want synth_kernel", k.Name)
	}
	if k.Registers != 48 || k.RegCount != 48 {
		t.Errorf("kernel Registers = %d, want 48", k.Registers)
	}
	if k.FrameSizeBytes != 256 || k.FrameSize != 256 {
		t.Errorf("kernel FrameSizeBytes = %d, want 256", k.FrameSizeBytes)
	}
	if k.MaxRegCount != 64 {
		t.Errorf("kernel MaxRegCount = %d, want 64", k.MaxRegCount)
	}
	if k.ReqNTID != [3]int{128, 2, 1} {
		t.Errorf("kernel ReqNTID = %v, want [128 2 1]", k.ReqNTID)
	}
	if k.SharedMemBytes != 4096 {
		t.Errorf("kernel SharedMemBytes = %d, want 4096", k.SharedMemBytes)
	}
}

func TestCalculateOccupancy_SM90(t *testing.T) {
	tests := []struct {
		name         string
		regCount     int
		sharedMem    int
		threads      int
		wantWarps    int
		wantBlocks   int
		wantOcc      float64
		wantLimiting string
	}{
		{
			name:         "hopper_1024_threads_40_regs",
			regCount:     40,
			sharedMem:    0,
			threads:      1024,
			wantWarps:    32,
			wantBlocks:   1,
			wantOcc:      0.50,
			wantLimiting: cudaoccupancy.LimitRegisters,
		},
		{
			name:         "hopper_256_threads_40_regs",
			regCount:     40,
			sharedMem:    0,
			threads:      256,
			wantWarps:    48,
			wantBlocks:   6,
			wantOcc:      48.0 / 64.0,
			wantLimiting: cudaoccupancy.LimitRegisters,
		},
		{
			name:         "hopper_256_threads_32_regs",
			regCount:     32,
			sharedMem:    0,
			threads:      256,
			wantWarps:    64,
			wantBlocks:   8,
			wantOcc:      1.0,
			wantLimiting: cudaoccupancy.LimitNone,
		},
		{
			name:         "hopper_256_threads_128_regs",
			regCount:     128,
			sharedMem:    0,
			threads:      256,
			wantWarps:    16,
			wantBlocks:   2,
			wantOcc:      16.0 / 64.0,
			wantLimiting: cudaoccupancy.LimitRegisters,
		},
		{
			name:         "hopper_256_threads_smem_limited",
			regCount:     32,
			sharedMem:    65536, // 64 KB
			threads:      256,
			wantWarps:    24,
			wantBlocks:   3,
			wantOcc:      24.0 / 64.0,
			wantLimiting: cudaoccupancy.LimitSharedMemory,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, err := cudaoccupancy.CalculateOccupancy("sm_90", tt.regCount, tt.sharedMem, tt.threads)
			if err != nil {
				t.Fatalf("CalculateOccupancy: %v", err)
			}
			if res.TheoreticalWarps != tt.wantWarps || res.ActiveWarps != tt.wantWarps {
				t.Errorf("TheoreticalWarps = %d, want %d", res.TheoreticalWarps, tt.wantWarps)
			}
			if res.ActiveBlocks != tt.wantBlocks {
				t.Errorf("ActiveBlocks = %d, want %d", res.ActiveBlocks, tt.wantBlocks)
			}
			if math.Abs(res.Occupancy-tt.wantOcc) > 1e-4 {
				t.Errorf("Occupancy = %.4f, want %.4f", res.Occupancy, tt.wantOcc)
			}
			if res.LimitingFactor != tt.wantLimiting {
				t.Errorf("LimitingFactor = %q, want %q", res.LimitingFactor, tt.wantLimiting)
			}
		})
	}
}

func TestCalculateOccupancy_SM121(t *testing.T) {
	tests := []struct {
		name         string
		regCount     int
		sharedMem    int
		threads      int
		wantWarps    int
		wantBlocks   int
		wantOcc      float64
		wantLimiting string
	}{
		{
			name:         "gb10_256_threads_40_regs_full_occupancy",
			regCount:     40,
			sharedMem:    1024,
			threads:      256,
			wantWarps:    48,
			wantBlocks:   6,
			wantOcc:      1.0,
			wantLimiting: cudaoccupancy.LimitNone,
		},
		{
			name:         "gb10_1024_threads_block_size_limit",
			regCount:     40,
			sharedMem:    1024,
			threads:      1024,
			wantWarps:    32,
			wantBlocks:   1,
			wantOcc:      32.0 / 48.0,
			wantLimiting: cudaoccupancy.LimitBlockSize,
		},
		{
			name:         "gb10_256_threads_smem_16k",
			regCount:     40,
			sharedMem:    17408, // 1024 + 16384
			threads:      256,
			wantWarps:    40,
			wantBlocks:   5,
			wantOcc:      40.0 / 48.0,
			wantLimiting: cudaoccupancy.LimitSharedMemory,
		},
		{
			name:         "gb10_256_threads_smem_32k",
			regCount:     40,
			sharedMem:    33792, // 1024 + 32768
			threads:      256,
			wantWarps:    24,
			wantBlocks:   3,
			wantOcc:      24.0 / 48.0,
			wantLimiting: cudaoccupancy.LimitSharedMemory,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, err := cudaoccupancy.CalculateOccupancy("sm_121", tt.regCount, tt.sharedMem, tt.threads)
			if err != nil {
				t.Fatalf("CalculateOccupancy: %v", err)
			}
			if res.TheoreticalWarps != tt.wantWarps || res.ActiveWarps != tt.wantWarps {
				t.Errorf("TheoreticalWarps = %d, want %d", res.TheoreticalWarps, tt.wantWarps)
			}
			if res.ActiveBlocks != tt.wantBlocks {
				t.Errorf("ActiveBlocks = %d, want %d", res.ActiveBlocks, tt.wantBlocks)
			}
			if math.Abs(res.Occupancy-tt.wantOcc) > 1e-4 {
				t.Errorf("Occupancy = %.4f, want %.4f", res.Occupancy, tt.wantOcc)
			}
			if res.LimitingFactor != tt.wantLimiting {
				t.Errorf("LimitingFactor = %q, want %q", res.LimitingFactor, tt.wantLimiting)
			}
		})
	}
}

func TestCalculateOccupancy_Validation(t *testing.T) {
	_, err := cudaoccupancy.CalculateOccupancy("unknown_arch_xyz", 32, 0, 256)
	if err == nil {
		t.Errorf("expected error for unknown arch, got nil")
	}

	for _, invalidThreads := range []int{0, -1, 1025, 2048} {
		res, err := cudaoccupancy.CalculateOccupancy("sm_90", 32, 0, invalidThreads)
		if err != nil {
			t.Fatalf("unexpected error for thread count %d: %v", invalidThreads, err)
		}
		if res.TheoreticalWarps != 0 || res.ActiveWarps != 0 {
			t.Errorf("threads %d: TheoreticalWarps = %d, want 0", invalidThreads, res.TheoreticalWarps)
		}
		if res.LimitingFactor != cudaoccupancy.LimitBlockSize {
			t.Errorf("threads %d: LimitingFactor = %q, want %q", invalidThreads, res.LimitingFactor, cudaoccupancy.LimitBlockSize)
		}
	}
}

func ExampleCalculateOccupancy() {
	res, err := cudaoccupancy.CalculateOccupancy("sm_90", 40, 0, 256)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s: %d/%d warps (%.0f%%) limited by %s\n",
		res.Architecture, res.TheoreticalWarps, res.MaxWarps, res.Occupancy*100, res.LimitingFactor)
	// Output:
	// sm_90: 48/64 warps (75%) limited by registers
}

func synthesizeTestELF(sections map[string][]byte) []byte {
	var buf bytes.Buffer

	secNames := []string{""}
	secData := [][]byte{nil}
	secTypes := []uint32{0}
	secFlags := []uint64{0}

	shstrtabIdx := len(secNames)
	secNames = append(secNames, ".shstrtab")
	secTypes = append(secTypes, uint32(elf.SHT_STRTAB))
	secFlags = append(secFlags, 0)
	secData = append(secData, nil)

	for name, data := range sections {
		secNames = append(secNames, name)
		secTypes = append(secTypes, uint32(elf.SHT_PROGBITS))
		secFlags = append(secFlags, 0)
		secData = append(secData, data)
	}

	var shstrtab bytes.Buffer
	shstrtab.WriteByte(0)
	nameOffsets := make([]uint32, len(secNames))
	for i, name := range secNames {
		if i == 0 {
			nameOffsets[i] = 0
			continue
		}
		nameOffsets[i] = uint32(shstrtab.Len())
		shstrtab.WriteString(name)
		shstrtab.WriteByte(0)
	}
	secData[shstrtabIdx] = shstrtab.Bytes()

	ehdr := make([]byte, 64)
	copy(ehdr[0:4], []byte("\x7fELF"))
	ehdr[4] = 2                                     // 64-bit
	ehdr[5] = 1                                     // Little-endian
	ehdr[6] = 1                                     // Version 1
	binary.LittleEndian.PutUint16(ehdr[16:18], 1)   // ET_REL
	binary.LittleEndian.PutUint16(ehdr[18:20], 190) // EM_CUDA
	binary.LittleEndian.PutUint32(ehdr[20:24], 1)   // EV_CURRENT
	binary.LittleEndian.PutUint16(ehdr[52:54], 64)  // ehsize
	binary.LittleEndian.PutUint16(ehdr[58:60], 64)  // shentsize
	binary.LittleEndian.PutUint16(ehdr[60:62], uint16(len(secNames)))
	binary.LittleEndian.PutUint16(ehdr[62:64], uint16(shstrtabIdx))

	buf.Write(ehdr)

	secOffsets := make([]uint64, len(secNames))
	for i, data := range secData {
		if len(data) == 0 {
			secOffsets[i] = 0
			continue
		}
		secOffsets[i] = uint64(buf.Len())
		buf.Write(data)
	}

	shoff := uint64(buf.Len())
	binary.LittleEndian.PutUint64(ehdr[40:48], shoff)
	copy(buf.Bytes()[40:48], ehdr[40:48])

	for i := range secNames {
		shdr := make([]byte, 64)
		binary.LittleEndian.PutUint32(shdr[0:4], nameOffsets[i])
		binary.LittleEndian.PutUint32(shdr[4:8], secTypes[i])
		binary.LittleEndian.PutUint64(shdr[8:16], secFlags[i])
		binary.LittleEndian.PutUint64(shdr[24:32], secOffsets[i])
		binary.LittleEndian.PutUint64(shdr[32:40], uint64(len(secData[i])))
		binary.LittleEndian.PutUint64(shdr[48:56], 1)
		buf.Write(shdr)
	}

	return buf.Bytes()
}
