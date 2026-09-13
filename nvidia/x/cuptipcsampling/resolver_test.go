package cuptipcsampling_test

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

func loadTestCode(t *testing.T) (*cudacapture.Code, *cudacapture.Capture) {
	t.Helper()
	dir := filepath.Join("..", "cudacapture", "testdata")
	cubinHex, err := os.ReadFile(filepath.Join(dir, "kernels.cubin.hex"))
	if err != nil {
		t.Fatalf("read kernels.cubin.hex: %v", err)
	}
	cubinBytes, err := hex.DecodeString(strings.TrimSpace(string(cubinHex)))
	if err != nil {
		t.Fatalf("decode cubin hex: %v", err)
	}
	cuBytes, err := os.ReadFile(filepath.Join(dir, "kernels.cu"))
	if err != nil {
		t.Fatalf("read kernels.cu: %v", err)
	}
	ptxBytes, err := os.ReadFile(filepath.Join(dir, "kernels.ptx"))
	if err != nil {
		t.Fatalf("read kernels.ptx: %v", err)
	}
	sassBytes, err := os.ReadFile(filepath.Join(dir, "kernels.sass"))
	if err != nil {
		t.Fatalf("read kernels.sass: %v", err)
	}

	cap := &cudacapture.Capture{
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.cu":    cuBytes,
			"kernels.ptx":   ptxBytes,
			"kernels.cubin": cubinBytes,
			"kernels.sass":  sassBytes,
		},
	}

	code, err := cudacapture.LoadCode(cap)
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	return code, cap
}

func TestCaptureResolverResolvePC(t *testing.T) {
	code, _ := loadTestCode(t)
	const expectedCRC = 0x1122334455667788
	const expectedGen = 42

	res := cuptipcsampling.NewCaptureResolver(code, expectedCRC)
	res.Generation = expectedGen

	mod := cuptipcsampling.ModuleID{CRC: expectedCRC, Generation: expectedGen}

	// Offset 0 (entry instruction)
	loc0 := res.ResolvePC(mod, "compute", 0)
	if !loc0.Resolved {
		t.Fatal("expected loc0 to be resolved")
	}
	if loc0.KernelName != "compute" {
		t.Errorf("KernelName = %q, want compute", loc0.KernelName)
	}
	if loc0.SourceFile != "kernels.cu" {
		t.Errorf("SourceFile = %q, want kernels.cu", loc0.SourceFile)
	}
	if loc0.SourceLine != 2 {
		t.Errorf("SourceLine = %d, want 2", loc0.SourceLine)
	}
	if loc0.PTXFile != "kernels.ptx" {
		t.Errorf("PTXFile = %q, want kernels.ptx", loc0.PTXFile)
	}
	if loc0.PTXLine != 12 {
		t.Errorf("PTXLine = %d, want 12", loc0.PTXLine)
	}
	if loc0.SASSFile != "kernels.sass" {
		t.Errorf("SASSFile = %q, want kernels.sass", loc0.SASSFile)
	}
	if loc0.SASSLine != 3 {
		t.Errorf("SASSLine = %d, want 3", loc0.SASSLine)
	}
	if loc0.CUBINOffset != 96 {
		t.Errorf("CUBINOffset = %d, want 96", loc0.CUBINOffset)
	}

	// Offset 0x10 (second instruction)
	loc1 := res.ResolvePC(mod, "compute", 0x10)
	if !loc1.Resolved {
		t.Fatal("expected loc1 to be resolved")
	}
	if loc1.SourceLine != 3 {
		t.Errorf("SourceLine = %d, want 3", loc1.SourceLine)
	}
	if loc1.PTXLine != 14 {
		t.Errorf("PTXLine = %d, want 14", loc1.PTXLine)
	}
	if loc1.SASSLine != 5 {
		t.Errorf("SASSLine = %d, want 5", loc1.SASSLine)
	}
	if loc1.CUBINOffset != 96+0x10 {
		t.Errorf("CUBINOffset = %d, want %d", loc1.CUBINOffset, 96+0x10)
	}
}

func TestCaptureResolverMismatches(t *testing.T) {
	code, _ := loadTestCode(t)
	const expectedCRC = 0x1122334455667788
	const expectedGen = 42

	res := cuptipcsampling.NewCaptureResolver(code, expectedCRC)
	res.Generation = expectedGen

	tests := []struct {
		name       string
		mod        cuptipcsampling.ModuleID
		kernelName string
		pcOffset   uint64
		wantName   string
	}{
		{
			name:       "mismatched CRC",
			mod:        cuptipcsampling.ModuleID{CRC: 0xdeadbeef, Generation: expectedGen},
			kernelName: "compute",
			pcOffset:   0,
			wantName:   "compute",
		},
		{
			name:       "mismatched generation",
			mod:        cuptipcsampling.ModuleID{CRC: expectedCRC, Generation: 999},
			kernelName: "compute",
			pcOffset:   0,
			wantName:   "compute",
		},
		{
			name:       "unknown kernel",
			mod:        cuptipcsampling.ModuleID{CRC: expectedCRC, Generation: expectedGen},
			kernelName: "unknown_kernel",
			pcOffset:   0,
			wantName:   "unknown_kernel",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loc := res.ResolvePC(tt.mod, tt.kernelName, tt.pcOffset)
			if loc.Resolved {
				t.Errorf("expected Resolved=false, got true for %s", tt.name)
			}
			if loc.KernelName != tt.wantName {
				t.Errorf("KernelName = %q, want %q", loc.KernelName, tt.wantName)
			}
		})
	}

	// Nil receiver and nil code
	var nilRes *cuptipcsampling.CaptureResolver
	mod := cuptipcsampling.ModuleID{CRC: expectedCRC, Generation: expectedGen}
	if loc := nilRes.ResolvePC(mod, "compute", 0); loc.Resolved {
		t.Error("nil resolver: expected Resolved=false")
	}

	emptyRes := &cuptipcsampling.CaptureResolver{}
	if loc := emptyRes.ResolvePC(mod, "compute", 0); loc.Resolved {
		t.Error("empty resolver: expected Resolved=false")
	}
}

func TestCaptureResolverFromCapture(t *testing.T) {
	code, cap := loadTestCode(t)
	cap.Modules = []cudacapture.Module{
		{ID: "mod1", Generation: 88},
	}

	res := cuptipcsampling.NewCaptureResolverFromCapture(code, cap)
	if res.Generation != 88 {
		t.Errorf("res.Generation = %d, want 88", res.Generation)
	}

	// Nil capture safe check
	resNil := cuptipcsampling.NewCaptureResolverFromCapture(code, nil)
	if resNil.Generation != 0 {
		t.Errorf("resNil.Generation = %d, want 0", resNil.Generation)
	}
}
