package bitopt

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestEpiloguePTXNative assembles the synthesized NVFP4 epilogue with
// ptxas for sm_100a. It runs only with CUDA_BITOPT_NATIVE=1 and ptxas on
// PATH; it launches nothing, so it needs no GPU.
func TestEpiloguePTXNative(t *testing.T) {
	if os.Getenv("CUDA_BITOPT_NATIVE") != "1" {
		t.Skip("CUDA_BITOPT_NATIVE=1 not set")
	}
	ptxas, err := exec.LookPath("ptxas")
	if err != nil {
		t.Fatal("ptxas not on PATH")
	}
	res, err := SynthesizeEpilogueDAG(nvfp4Epilogue(4), 2)
	if err != nil {
		t.Fatal(err)
	}
	coalesced, _, err := CoalesceWaits(res.PTX)
	if err != nil {
		t.Fatal(err)
	}
	for name, text := range map[string]string{"emitted": res.PTX, "coalesced": coalesced} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			src := filepath.Join(dir, "epilogue.ptx")
			if err := os.WriteFile(src, []byte(text), 0o644); err != nil {
				t.Fatal(err)
			}
			out, err := exec.Command(ptxas, "-arch=sm_100a", "-v", "-o", filepath.Join(dir, "epilogue.cubin"), src).CombinedOutput()
			t.Logf("%s", out)
			if err != nil {
				t.Fatalf("ptxas: %v\n%s", err, text)
			}
		})
	}
}
