package metalmlx

import (
	"os"
	"path/filepath"
	"slices"
	"sync/atomic"
	"testing"
)

// metallibPath returns MLX's built metallib, from MLX_METALLIB or from an MLX
// checkout under MLX_ROOT.
func metallibPath(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("MLX_METALLIB"); p != "" {
		return p
	}
	root := os.Getenv("MLX_ROOT")
	if root == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			t.Skip("no home directory")
		}
		root = filepath.Join(home, "ml-explore", "mlx")
	}
	p := filepath.Join(root, "python", "mlx", "lib", "mlx.metallib")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no mlx.metallib at %s; set MLX_METALLIB", p)
	}
	return p
}

// TestMLXMetallibFenceUpdate runs a kernel out of MLX's own metallib, which is
// what the library cache is for.
func TestMLXMetallibFenceUpdate(t *testing.T) {
	d := testDevice(t)
	lib, err := d.LibraryFromPath("mlx", metallibPath(t))
	if err != nil {
		t.Fatal(err)
	}
	if names := lib.FunctionNames(); !slices.Contains(names, "fence_update") {
		t.Fatalf("mlx.metallib has %d functions, none named fence_update", len(names))
	}
	kernel, err := d.Kernel(lib, "fence_update")
	if err != nil {
		t.Fatal(err)
	}

	e, err := NewCommandEncoder(d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	a := d.Allocator()
	buf, err := a.Malloc(4)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free(buf)
	timestamp := (*uint32)(buf.Contents())
	atomic.StoreUint32(timestamp, 0)

	value := uint32(42)
	e.SetComputePipelineState(kernel)
	e.SetBuffer(buf, 0, 0)
	e.SetBytes(mustValueBytes(&value), 1)
	e.DispatchThreads(Size{1, 1, 1}, Size{1, 1, 1})
	if err := e.Synchronize(); err != nil {
		t.Fatal(err)
	}
	if got := atomic.LoadUint32(timestamp); got != value {
		t.Errorf("timestamp = %d, want %d", got, value)
	}
}
