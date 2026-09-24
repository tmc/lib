package cudaagreement_test

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// nvccCUBIN compiles CUDA C++ source to a CUBIN for target. The mega worker
// and the decoder kernels include the CUDA runtime headers, so they go
// through nvcc rather than NVRTC. No fast-math flag is passed: the decoder
// oracle's bounds assume the accurate math functions.
func nvccCUBIN(t *testing.T, name, src, target string) ([]byte, error) {
	t.Helper()
	nvcc := nvccPath(t)
	dir := t.TempDir()
	cu := filepath.Join(dir, name+".cu")
	out := filepath.Join(dir, name+".cubin")
	if err := os.WriteFile(cu, []byte(src), 0o644); err != nil {
		return nil, err
	}
	cmd := exec.Command(nvcc, "-cubin", "-arch="+target, "-o", out, cu)
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s\n%s", cmd, b)
		return nil, err
	}
	t.Logf("compiled %s: %s", name, cmd)
	return os.ReadFile(out)
}

var nvccOnce struct {
	sync.Mutex
	path string
}

func nvccPath(t *testing.T) string {
	t.Helper()
	nvccOnce.Lock()
	defer nvccOnce.Unlock()
	if nvccOnce.path == "" {
		nvccOnce.path, _ = cudatest.NVCC(t)
	}
	return nvccOnce.path
}

// megaCompile is a MegaExecutor.Compile for t.
func megaCompile(t *testing.T) func(string, cudamega.Target) (cudakernel.Kind, []byte, error) {
	return func(src string, target cudamega.Target) (cudakernel.Kind, []byte, error) {
		image, err := nvccCUBIN(t, "dag_worker", src, string(target))
		return cudakernel.KindCUBIN, image, err
	}
}

// loadModule loads a CUBIN image on dev, closed at cleanup.
func loadModule(t *testing.T, dev *cuda.Device, image []byte) *cuda.Module {
	t.Helper()
	var raw driver.Module
	err := dev.WithContext(func() error {
		var err error
		raw, err = driver.ModuleLoadData(image)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	mod, err := cuda.OwnModule(dev, raw)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := mod.Close(); err != nil {
			t.Error(err)
		}
	})
	return mod
}

func synchronize(t *testing.T, s *cuda.Stream) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	if err := s.Synchronize(ctx); err != nil {
		t.Fatal(err)
	}
}
