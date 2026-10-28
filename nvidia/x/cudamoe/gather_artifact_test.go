package cudamoe

import (
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

type gatherArtifactFunc func(string, cudakernel.Schema, string) (*cudakernel.Artifact, error)

func (f gatherArtifactFunc) ToArtifact(entry string, schema cudakernel.Schema, target string) (*cudakernel.Artifact, error) {
	return f(entry, schema, target)
}

func TestGatherArtifact(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "include"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "include", "cuda.h"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CUDA_HOME", dir)
	for _, tile := range []int{0, 16, 32} {
		source, _ := GatherSource(tile)
		entry := "moe_gather"
		if tile == 0 {
			source, entry = GatherScalarSource(), "moe_gather_scalar"
		}
		a, err := compileGather(source, entry, "sm_121a", func(src string, opts ...nvrtcx.Option) (gatherModule, error) {
			if src != source {
				t.Fatal("compiled wrong tile")
			}
			flags, err := nvrtcx.CompileFlags(opts...)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Contains(flags, "--gpu-architecture=sm_121a") {
				t.Fatal("missing physical target")
			}
			return gatherArtifactFunc(func(gotEntry string, schema cudakernel.Schema, target string) (*cudakernel.Artifact, error) {
				if gotEntry != entry || target != "sm_121a" || schema.TotalParamBytes != 40 || len(schema.Params) != 6 {
					t.Fatalf("entry=%s target=%s schema=%+v", gotEntry, target, schema)
				}
				for i, p := range schema.Params {
					if i < 3 && !p.IsPointer || i >= 3 && p.Type != "s32" {
						t.Fatalf("parameter %d: %+v", i, p)
					}
				}
				if _, err := schema.Pack(driver.DevicePtr(16), driver.DevicePtr(32), driver.DevicePtr(48), int32(5), int32(33), int32(3)); err != nil {
					t.Fatal(err)
				}
				return cudakernel.NewArtifact(cudakernel.KindCUBIN, artifactELFFixture(), entry, schema, target, cudakernel.WithResources(cudakernel.ResourceRequirements{AllocatedRegisters: 20}))
			}), nil
		})
		if err != nil || a.Resources().AllocatedRegisters != 20 || len(a.CompilerOptions()) == 0 {
			t.Fatalf("artifact=%+v err=%v", a, err)
		}
	}
	failure := errors.New("compile failure")
	if _, err := gatherArtifact(16, "sm_121a", func(string, ...nvrtcx.Option) (gatherModule, error) { return nil, failure }); !errors.Is(err, failure) {
		t.Fatalf("lost compiler error: %v", err)
	}
	for _, target := range []string{"", "compute_121a", "sm_bad"} {
		if _, err := gatherArtifact(16, target, func(string, ...nvrtcx.Option) (gatherModule, error) {
			t.Fatal("compiled invalid target")
			return nil, nil
		}); err == nil {
			t.Fatalf("accepted %q", target)
		}
	}
}

func TestGatherCompile(t *testing.T) {
	if os.Getenv("CUDAMOE_GATHER_COMPILE") != "1" {
		t.Skip("set CUDAMOE_GATHER_COMPILE=1 and CUDAMOE_TARGET for authorized NVRTC compilation")
	}
	for _, tile := range []int{0, 16, 32} {
		var a *cudakernel.Artifact
		var err error
		if tile == 0 {
			a, err = GatherScalarArtifact(os.Getenv("CUDAMOE_TARGET"))
		} else {
			a, err = GatherArtifact(tile, os.Getenv("CUDAMOE_TARGET"))
		}
		if err != nil {
			t.Fatal(err)
		}
		info, err := cudakernel.ParseCUBIN(a.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Contains(info.Symbols, a.EntryPoint()) {
			t.Fatalf("missing entry %s", a.EntryPoint())
		}
		t.Logf("tile=%d target=%s digest=%s resources=%+v flags=%v", tile, a.Target(), a.Digest(), a.Resources(), a.CompilerOptions())
	}
}

// artifactELFFixture is a parser-only CUDA ELF header, not executable device code.
func artifactELFFixture() []byte {
	image := make([]byte, 64)
	copy(image, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1, 0x33, 7})
	binary.LittleEndian.PutUint16(image[16:], 2)
	binary.LittleEndian.PutUint16(image[18:], 190)
	binary.LittleEndian.PutUint32(image[20:], 1)
	binary.LittleEndian.PutUint16(image[52:], 64)
	return image
}
