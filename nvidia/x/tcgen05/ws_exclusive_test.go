package tcgen05

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func TestZeroColumnMaskDesc(t *testing.T) {
	desc := ZeroColumnMaskDesc{
		StartCount:  [8]uint8{2, 0, 4, 1, 5, 3, 6, 7},
		SkipSpan:    [8]uint8{2, 3, 1, 4, 0, 2, 1, 0}, // spans: 3, 4, 2, 5, 1, 3, 2, 1
		NonZeroMask: 0b01010101,                       // groups 0, 2, 4, 6 active
	}
	encoded := desc.Encode()
	got := DecodeZeroColumnMaskDesc(encoded)
	if got != desc {
		t.Fatalf("DecodeZeroColumnMaskDesc(0x%016x) = %+v, want %+v", encoded, got, desc)
	}

	// Group 0 (cols 0..7): active, StartCount=2, SkipSpan=2 (span=3) -> cols 2, 3, 4 zeroed.
	for c := 0; c < 8; c++ {
		wantZero := c >= 2 && c < 5
		if desc.IsColumnZeroed(c) != wantZero {
			t.Fatalf("IsColumnZeroed(%d) = %v, want %v", c, desc.IsColumnZeroed(c), wantZero)
		}
	}

	// Group 1 (cols 8..15): inactive (bit 1 of NonZeroMask == 0) -> no columns zeroed.
	for c := 8; c < 16; c++ {
		if desc.IsColumnZeroed(c) {
			t.Fatalf("IsColumnZeroed(%d) = true in inactive group 1", c)
		}
	}

	// Group 2 (cols 16..23): active, StartCount=4, SkipSpan=1 (span=2) -> local cols 4, 5 (cols 20, 21) zeroed.
	for c := 16; c < 24; c++ {
		local := c - 16
		wantZero := local >= 4 && local < 6
		if desc.IsColumnZeroed(c) != wantZero {
			t.Fatalf("IsColumnZeroed(%d) = %v, want %v", c, desc.IsColumnZeroed(c), wantZero)
		}
	}

	if desc.IsColumnZeroed(-1) {
		t.Fatalf("IsColumnZeroed(-1) = true, want false")
	}
}

func TestTCGen05WeightsStationaryGEMMInstDescriptor(t *testing.T) {
	// M>>4 = 8 <<24 = 0x08000000 and N>>3 = 16 <<17 = 0x200000 in every word.
	tests := []struct {
		kind string
		want uint32
	}{
		{"tf32", 0x08200910},   // dtype F32 1<<4, atype/btype TF32 2<<7 | 2<<10
		{"f16", 0x08200490},    // dtype F32 1<<4, atype/btype BF16 1<<7 | 1<<10
		{"f8f6f4", 0x08200010}, // dtype F32 1<<4, atype/btype E4M3 0
		{"i8", 0x082004a0},     // dtype S32 2<<4, atype/btype S8 1<<7 | 1<<10
		{"ti16", 0x08200da0},   // dtype S32 2<<4, atype/btype S1Z4M11 3<<7 | 3<<10
	}
	for _, tt := range tests {
		k := TCGen05WeightsStationaryGEMM{Kind: tt.kind, M: 128, N: 128, K: 64, Target: "sm_107a"}
		got, err := k.InstDescriptor()
		if err != nil {
			t.Fatalf("%s: InstDescriptor() error: %v", tt.kind, err)
		}
		if got != tt.want {
			t.Fatalf("%s: InstDescriptor() = 0x%08x, want 0x%08x", tt.kind, got, tt.want)
		}
	}
}

func TestTCGen05WeightsStationaryGEMM(t *testing.T) {
	// 1. Target gating: .kind::ti16 requires sm_107a or sm_107f, rejects sm_100a and sm_103a.
	badTI16 := TCGen05WeightsStationaryGEMM{
		Kind:   "ti16",
		Target: "sm_100a",
	}
	if err := badTI16.Validate(); err == nil {
		t.Fatalf("expected error for .kind::ti16 on sm_100a")
	}

	// 2. PTX emission checks for .exclusive 576-col TMEM, .kind::ti16, .collector::b2::use, and .sp.
	zmask := ZeroColumnMaskDesc{
		StartCount:  [8]uint8{1, 2, 0, 3, 2, 1, 4, 0},
		SkipSpan:    [8]uint8{2, 1, 3, 2, 1, 2, 1, 3},
		NonZeroMask: 0b10110101,
	}
	k := TCGen05WeightsStationaryGEMM{
		Name:                "eg_tcgen05_ws_gemm",
		M:                   128,
		N:                   128,
		K:                   64,
		CollectorBuffer:     WSCollectorB2,
		CollectorAction:     WSActionUse,
		Kind:                "ti16",
		UseExclusive576Cols: true,
		ZeroColumnMask:      zmask,
		Target:              "sm_107a",
	}
	if err := k.Validate(); err != nil {
		t.Fatalf("Validate() failed: %v", err)
	}
	src, err := k.CUDA()
	if err != nil {
		t.Fatalf("CUDA() failed: %v", err)
	}
	for _, wantSub := range []string{
		"tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32",
		"tcgen05.mma.ws.cta_group::1.kind::ti16.collector::b2::use",
		"__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070",
	} {
		if !strings.Contains(src, wantSub) {
			t.Fatalf("CUDA() missing %q", wantSub)
		}
	}
	if fullTU, err := k.CUDAFile(); err != nil || !strings.Contains(fullTU, "EG_PRELUDE") {
		t.Fatalf("CUDAFile() failed or missing Prelude: %v", err)
	}

	// .kind::ti16: dtype S32 2<<4, atype/btype S1Z4M11 3<<7 | 3<<10,
	// N>>3 = 16 <<17, M>>4 = 8 <<24.
	if !strings.Contains(src, "const uint32_t idesc = 0x08200da0u;") {
		t.Fatalf("CUDA() missing .kind::ti16 idesc 0x08200da0")
	}

	// The sparse form needs 2:4 metadata in TMEM, which the kernel does not stage.
	kSparse := k
	kSparse.Sparse = true
	if _, err := kSparse.CUDA(); err == nil || !strings.Contains(err.Error(), "sparse") {
		t.Fatalf("sparse CUDA() error = %v, want sparse form unsupported", err)
	}

	// 3. Bit-exact cudahostemu.Run vs Reference with non-trivial ZeroColumnMaskDesc.
	rng := rand.New(rand.NewPCG(1301, 1302))
	a := make([]float32, k.M*k.K)
	b := make([]float32, k.K*k.N)
	for i := range a {
		a[i] = float32(rng.IntN(9)-4)*0.5 + 0.25
	}
	for i := range b {
		b[i] = float32(rng.IntN(9)-4)*0.5 - 0.25
	}

	want, err := k.Reference(a, b)
	if err != nil {
		t.Fatalf("Reference() failed: %v", err)
	}

	grid, block := k.Grid()
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), 0, %s, a, b, out)",
		grid[0], grid[1], grid[2], block[0], block[1], block[2], k.Name)
	res := runEmu(t, src, launch,
		emuBuf{"a", "float", f32Bytes(a)},
		emuBuf{"b", "float", f32Bytes(b)},
		emuBuf{"out", "float", poison(4 * k.M * k.N)},
	)
	got := bytesF32(res["out"])

	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))
		if math.IsNaN(float64(got[i])) || diff != 0 {
			t.Fatalf("out[%d] = %g, want bit-exact %g (diff %g)", i, got[i], want[i], diff)
		}
	}

	// Negative control: unmasked reference must differ from masked output.
	kUnmasked := k
	kUnmasked.ZeroColumnMask = ZeroColumnMaskDesc{}
	unmaskedRef, err := kUnmasked.Reference(a, b)
	if err != nil {
		t.Fatalf("unmasked Reference() failed: %v", err)
	}
	var diffs int
	for i := range want {
		if want[i] != unmaskedRef[i] {
			diffs++
		}
	}
	if diffs == 0 {
		t.Fatalf("ZeroColumnMaskDesc did not alter GEMM output relative to unmasked reference")
	}
}

func ExampleTCGen05WeightsStationaryGEMM() {
	zmask := ZeroColumnMaskDesc{
		StartCount:  [8]uint8{2, 0, 0, 0, 0, 0, 0, 0},
		SkipSpan:    [8]uint8{1, 0, 0, 0, 0, 0, 0, 0},
		NonZeroMask: 0x01,
	}
	k := TCGen05WeightsStationaryGEMM{
		M:                   128,
		N:                   128,
		K:                   64,
		CollectorBuffer:     WSCollectorB2,
		CollectorAction:     WSActionUse,
		Kind:                "ti16",
		UseExclusive576Cols: true,
		ZeroColumnMask:      zmask,
		Target:              "sm_107a",
	}
	src, err := k.CUDA()
	fmt.Println(
		err == nil,
		zmask.IsColumnZeroed(1),
		zmask.IsColumnZeroed(2),
		strings.Contains(src, "tcgen05.alloc.exclusive.cta_group::1"),
		strings.Contains(src, "tcgen05.mma.ws.cta_group::1.kind::ti16.collector::b2::use"),
	)
	// Output:
	// true false true true true
}
