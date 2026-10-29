package warpsynth

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

// TestNativePTXAssembles assembles the emitted skeleton with ptxas for
// sm_100a (with TMEM and tcgen05.commit) and sm_121a (no TMEM, so plain
// mbarrier arrivals). It assembles only; nothing is launched. Run it on a
// host with the CUDA toolkit, under gpu-lock:
//
//	CUDA_WARPSYNTH_NATIVE=1 go test -run NativePTX ./x/egraph/warpsynth
func TestNativePTXAssembles(t *testing.T) {
	if os.Getenv("CUDA_WARPSYNTH_NATIVE") != "1" {
		t.Skip("CUDA_WARPSYNTH_NATIVE=1 unset; skeleton PTX not assembled")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate open without inherited gpu lock: %v", err)
	}
	ptxas, err := exec.LookPath("ptxas")
	if err != nil {
		t.Fatal(err)
	}
	for _, arch := range []string{"sm_100a", "sm_121a"} {
		t.Run(arch, func(t *testing.T) {
			p := gemmProblem()
			p.Arch = arch
			if TMEMColumns(arch) == 0 {
				p.TMEM = nil
			}
			s, err := Synthesize(p)
			if err != nil {
				t.Fatal(err)
			}
			src, err := s.EmitWarpSpecializedSkeletonPTX("ws_skeleton")
			if err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			in := filepath.Join(dir, "ws.ptx")
			if err := os.WriteFile(in, []byte(src), 0o644); err != nil {
				t.Fatal(err)
			}
			out, err := exec.Command(ptxas, "-arch="+arch, "-v", "-o", filepath.Join(dir, "ws.cubin"), in).CombinedOutput()
			if err != nil {
				t.Fatalf("ptxas: %v\n%s\n%s", err, out, src)
			}
			t.Logf("%s", out)
		})
	}
}
