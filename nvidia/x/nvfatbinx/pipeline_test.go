package nvfatbinx_test

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/nvfatbin"
	"github.com/tmc/lib/nvidia/nvjitlink"
	"github.com/tmc/lib/nvidia/x/nvfatbinx"
	"github.com/tmc/lib/nvidia/x/nvjitlinkx"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// TestNativePackaging needs a compatible nvFatbin library and cuobjdump. It
// packages and inventories two PTX targets without loading a module or a GPU.
func TestNativePackaging(t *testing.T) {
	if os.Getenv("NVFATBIN_TEST") != "1" {
		t.Skip("NVFATBIN_TEST=1 required")
	}
	tool := os.Getenv("CUOBJDUMP")
	if tool == "" {
		t.Fatal("CUOBJDUMP must identify the qualified utility")
	}
	major, minor, err := nvfatbin.Version()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("nvFatbin %d.%d path=%s", major, minor, nvfatbin.Path())
	var inputs []nvfatbinx.Input
	for _, arch := range []string{"80", "90"} {
		ptx := fmt.Sprintf(".version 8.0\n.target sm_%s\n.address_size 64\n.visible .entry empty() { ret; }\n", arch)
		inputs = append(inputs, nvfatbinx.Input{Kind: nvfatbinx.PTX, Name: "target" + arch, Arch: arch, Data: []byte(ptx)})
	}
	r, err := nvfatbinx.Build(inputs, "-64", "-cuda")
	if r != nil {
		t.Cleanup(func() {
			if err := r.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "targets.fatbin")
	if err := os.WriteFile(path, r.Data, 0600); err != nil {
		t.Fatal(err)
	}
	t.Logf("fatbinary SHA256=%x bytes=%d", sha256.Sum256(r.Data), len(r.Data))
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	version, err := exec.CommandContext(ctx, tool, "--version").CombinedOutput()
	if err != nil {
		t.Fatalf("cuobjdump version: %v\n%s", err, version)
	}
	t.Logf("cuobjdump: %s", version)
	listing, err := exec.CommandContext(ctx, tool, "--list-ptx", path).CombinedOutput()
	if err != nil {
		t.Fatalf("cuobjdump inventory: %v\n%s", err, listing)
	}
	t.Logf("inventory: %s", listing)
	for _, arch := range []string{"sm_80", "sm_90"} {
		if !strings.Contains(string(listing), arch) {
			t.Errorf("inventory lacks %s", arch)
		}
	}
	bad, err := nvfatbinx.Build([]nvfatbinx.Input{{Kind: nvfatbinx.PTX, Name: "invalid", Arch: "80", Data: []byte("not PTX")}})
	if bad != nil {
		t.Cleanup(func() {
			if err := bad.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err == nil {
		t.Fatal("invalid PTX accepted")
	}
	t.Logf("expected rejection: %v", err)
}

// TestCompiledPackaging checks that the linker can consume packaged compiled
// inputs. It does not execute the resulting kernel.
func TestCompiledPackaging(t *testing.T) {
	if os.Getenv("NVFATBIN_TEST") != "1" || os.Getenv("CUDA_COMPILER_TEST") != "1" {
		t.Skip("NVFATBIN_TEST=1 and CUDA_COMPILER_TEST=1 required")
	}
	for _, tt := range []struct {
		name   string
		output nvrtcx.OutputKind
		kind   nvfatbinx.Kind
	}{
		{"cubin", nvrtcx.OutputCUBIN, nvfatbinx.CUBIN},
		{"ltoir", nvrtcx.OutputLTOIR, nvfatbinx.LTOIR},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cm, err := nvrtcx.Compile(`extern "C" __global__ void packaged(float* out) { out[0] = 7; }`,
				nvrtcx.WithFlag("--gpu-architecture=sm_80"), nvrtcx.WithFlag("--device-c"), nvrtcx.WithOutput(tt.output), nvrtcx.WithNoCache())
			if err != nil {
				t.Fatal(err)
			}
			data := cm.CUBIN()
			if tt.output == nvrtcx.OutputLTOIR {
				data = cm.LTOIR()
			}
			if len(data) == 0 {
				t.Fatal("empty compiler artifact")
			}
			packed, err := nvfatbinx.Build([]nvfatbinx.Input{{Kind: tt.kind, Name: "packaged", Arch: "80", Data: data}}, "-64", "-cuda")
			if packed != nil {
				defer func() {
					if err := packed.Close(); err != nil {
						t.Error(err)
					}
				}()
			}
			if err != nil {
				t.Fatal(err)
			}
			options := []string{"-arch=sm_80"}
			if tt.kind == nvfatbinx.LTOIR {
				options = append(options, "-lto")
			}
			linked, err := nvjitlinkx.Link([]nvjitlinkx.Input{{Name: "packaged.fatbin", Kind: nvjitlink.InputFatbin, Data: packed.Data}}, options...)
			if linked != nil {
				defer func() {
					if err := linked.Close(); err != nil {
						t.Error(err)
					}
				}()
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(linked.CUBIN) < 4 || string(linked.CUBIN[:4]) != "\x7fELF" {
				t.Fatal("packaged input did not link to ELF")
			}
			t.Logf("input=%x fatbin=%x linked=%x", sha256.Sum256(data), sha256.Sum256(packed.Data), sha256.Sum256(linked.CUBIN))
		})
	}
}
