package cudaquant

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

type normArtifactFunc func(string, cudakernel.Schema, string) (*cudakernel.Artifact, error)

func (f normArtifactFunc) ToArtifact(entry string, schema cudakernel.Schema, target string) (*cudakernel.Artifact, error) {
	return f(entry, schema, target)
}

func TestNormQuantArtifactCompilation(t *testing.T) {
	normQuantHeaderFixture(t)
	for _, wantTarget := range []string{"sm_90", "sm_100a", "sm_121a"} {
		t.Run(wantTarget, func(t *testing.T) {
			var compiledFlags []string
			var result *cudakernel.Artifact
			a, err := normQuantArtifact(wantTarget, func(src string, opts ...nvrtcx.Option) (normQuantModule, error) {
				if src != RMSNormQ8_1Source() {
					t.Fatal("compiler received different source")
				}
				var err error
				compiledFlags, err = nvrtcx.CompileFlags(opts...)
				if err != nil {
					t.Fatal(err)
				}
				for _, flag := range []string{"--gpu-architecture=" + wantTarget, "-std=c++17"} {
					if !slices.Contains(compiledFlags, flag) {
						t.Fatalf("missing %s in %v", flag, compiledFlags)
					}
				}
				if slices.Contains(compiledFlags, "--use_fast_math") {
					t.Fatal("unqualified numerical mode")
				}
				return normArtifactFunc(func(entry string, schema cudakernel.Schema, target string) (*cudakernel.Artifact, error) {
					if entry != "rms_quant_q8_1" || target != wantTarget || schema.TotalParamBytes != 40 {
						t.Fatalf("wrong kernel contract: %s %s %+v", entry, target, schema)
					}
					wantNames := []string{"in", "weight", "out", "rows", "cols", "eps", "weighted"}
					wantTypes := []string{"ptr", "ptr", "ptr", "u32", "u32", "f32", "u32"}
					if len(schema.Params) != len(wantNames) {
						t.Fatal("wrong parameter count")
					}
					for i, p := range schema.Params {
						if p.Name != wantNames[i] || p.Type != wantTypes[i] || p.IsPointer != (i < 3) {
							t.Fatalf("parameter %d: %+v", i, p)
						}
					}
					// Synthetic compiler output tests transport only, not valid device code.
					result = &cudakernel.Artifact{
						Kind: cudakernel.KindCUBIN, Raw: []byte("compiled output fixture"),
						EntryPoint: entry, Schema: schema, Target: target,
						Resources: cudakernel.ResourceRequirements{AllocatedRegisters: 24},
						Toolchain: "compiler fixture",
					}
					return result, nil
				}), nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if a != result || a.Kind != cudakernel.KindCUBIN || a.Resources.AllocatedRegisters != 24 ||
				a.Toolchain != "compiler fixture" || !reflect.DeepEqual(a.CompilerOpts, compiledFlags) || bytes.Equal(a.Raw, []byte(RMSNormQ8_1Source())) {
				t.Fatalf("lost compiled artifact metadata: %+v", a)
			}
		})
	}
}

func TestNormQuantArtifactErrors(t *testing.T) {
	normQuantHeaderFixture(t)
	for _, target := range []string{"", "compute_121a", "sm_bad", "sm_121a\x00"} {
		if _, err := normQuantArtifact(target, func(string, ...nvrtcx.Option) (normQuantModule, error) {
			t.Fatal("compiled invalid target")
			return nil, nil
		}); err == nil {
			t.Fatalf("accepted %q", target)
		}
	}
	failure := errors.New("compiler failed")
	for _, stage := range []string{"compile", "adapter", "empty"} {
		t.Run(stage, func(t *testing.T) {
			a, err := normQuantArtifact("sm_100a", func(string, ...nvrtcx.Option) (normQuantModule, error) {
				if stage == "compile" {
					return nil, failure
				}
				if stage == "empty" {
					return nil, nil
				}
				return normArtifactFunc(func(string, cudakernel.Schema, string) (*cudakernel.Artifact, error) {
					return nil, failure
				}), nil
			})
			if a != nil || err == nil || (stage != "empty" && !errors.Is(err, failure)) {
				t.Fatalf("artifact=%v error=%v", a, err)
			}
		})
	}
}

func TestUnfusedNormArtifacts(t *testing.T) {
	normQuantHeaderFixture(t)
	entries := []string{"rms_f32", "quant_q8_1"}
	var calls []string
	a, err := normQuantArtifacts("sm_121a", RMSNormQ8_1UnfusedSource(), entries,
		func(source string, opts ...nvrtcx.Option) (normQuantModule, error) {
			if source != RMSNormQ8_1UnfusedSource() {
				t.Fatal("wrong unfused source")
			}
			return normArtifactFunc(func(entry string, schema cudakernel.Schema, target string) (*cudakernel.Artifact, error) {
				calls = append(calls, entry)
				if schema.TotalParamBytes != 40 || len(schema.Params) != 7 {
					t.Fatalf("incorrect ABI: %+v", schema)
				}
				return &cudakernel.Artifact{EntryPoint: entry, Target: target, Schema: schema,
					Resources: cudakernel.ResourceRequirements{AllocatedRegisters: uint32(len(calls) * 8)}}, nil
			}), nil
		})
	if err != nil || !slices.Equal(calls, entries) || len(a) != 2 {
		t.Fatalf("artifacts=%v calls=%v err=%v", a, calls, err)
	}
	for i := range a {
		if a[i].EntryPoint != entries[i] || a[i].Resources.AllocatedRegisters != uint32((i+1)*8) || len(a[i].CompilerOpts) == 0 {
			t.Fatalf("lost entry metadata: %+v", a[i])
		}
	}
}

// The mocked compiler never reads headers. This fixture makes option resolution
// independent of the host's toolkit installation; it is not a CUDA header set.
func normQuantHeaderFixture(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "include"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "include", "cuda.h"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CUDA_HOME", dir)
}

func TestNormQuantCompile(t *testing.T) {
	if os.Getenv("CUDAQUANT_COMPILE") != "1" {
		t.Skip("set CUDAQUANT_COMPILE=1 and CUDAQUANT_TARGET for NVRTC compilation")
	}
	a, err := RMSNormQ8_1Artifact(os.Getenv("CUDAQUANT_TARGET"))
	if err != nil {
		t.Fatal(err)
	}
	unfused, err := RMSNormQ8_1UnfusedArtifacts(os.Getenv("CUDAQUANT_TARGET"))
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range []*cudakernel.Artifact{a, unfused[0], unfused[1]} {
		if a.Kind != cudakernel.KindCUBIN {
			t.Fatalf("compiled kind=%v", a.Kind)
		}
		info, err := cudakernel.ParseCUBIN(a.Raw)
		if err != nil {
			t.Fatalf("invalid compiled CUBIN: %v", err)
		}
		if !slices.Contains(info.Symbols, a.EntryPoint) {
			t.Fatalf("compiled entry %q missing from %v", a.EntryPoint, info.Symbols)
		}
		t.Logf("target=%s entry=%s digest=%s resources=%+v flags=%v", a.Target, a.EntryPoint, a.Digest, a.Resources, a.CompilerOpts)
	}
}
