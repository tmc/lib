package nvfatbin_test

import (
	"bytes"
	"crypto/sha256"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/nvfatbin"
)

func TestTileIRPackaging(t *testing.T) {
	path := os.Getenv("NVFATBIN_TILEIR_INPUT")
	if path == "" {
		t.Skip("NVFATBIN_TILEIR_INPUT requires a real TileIR artifact and compatible nvFatbin")
	}
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	ir, err := os.ReadFile(path)
	check(err)
	if len(ir) == 0 {
		t.Fatal("empty TileIR input")
	}
	for _, mode := range []string{"true", "false"} {
		t.Run("compress="+mode, func(t *testing.T) {
			h, err := nvfatbin.Create([]string{"-64", "-cuda", "-compress=" + mode})
			check(err)
			defer func() { check(nvfatbin.Destroy(&h)) }()
			check(h.AddTileIR(unsafe.Pointer(&ir[0]), uint(len(ir)), "tile_fill", ""))
			runtime.KeepAlive(ir)
			size, err := h.Size()
			check(err)
			data, err := h.Get()
			check(err)
			if len(data) == 0 || uint(len(data)) != size {
				t.Fatal("invalid packaged size")
			}
			if mode == "false" && !bytes.Contains(data, ir) {
				t.Fatal("uncompressed package lost exact TileIR bytes")
			}
			t.Logf("input bytes=%d sha256=%x; fatbinary bytes=%d sha256=%x", len(ir), sha256.Sum256(ir), len(data), sha256.Sum256(data))
			if out := os.Getenv("NVFATBIN_TILEIR_OUTPUT"); out != "" {
				if mode == "false" {
					out += ".uncompressed"
				}
				check(os.WriteFile(out, data, 0600))
			}
			t.Log("packaging only; independent extraction and runtime loading are separate gates")
		})
	}
}
