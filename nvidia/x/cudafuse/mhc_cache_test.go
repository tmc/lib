package cudafuse

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func mhcCacheTestConfig() MHCConfig { return MHCConfig{Batch: 2, Width: 33, Repeat: 10, Epsilon: 1e-6} }

func TestMHCCacheRequests(t *testing.T) {
	c := mhcCacheTestConfig()
	sources, err := MHCSource(c)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range [][2]int{{12, 8}, {12, 9}, {13, 0}} {
		requests, err := mhcCacheRequests(c, 12, 1, v[0], v[1])
		if err != nil {
			t.Fatal(err)
		}
		for i, r := range requests {
			if !reflect.DeepEqual(r.Source, sources[i]) || r.InputKey != mhcCacheInputKey(c, r) || r.Target != "compute_121" {
				t.Fatalf("request differs from real generated source: %+v", r)
			}
			want := []string{"--gpu-architecture=compute_121", "-std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"}
			if v[0] > 12 || v[1] >= 9 {
				want = append(want, "--no-cache")
			}
			if !slices.Equal(r.Flags, want) {
				t.Fatalf("effective flags for %v: got %q want %q", v, r.Flags, want)
			}
			resolved, err := nvrtcx.CompileFlags(r.Options...)
			if err != nil || !slices.Equal(resolved, append([]string{"--gpu-architecture=compute_121", "-std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"}, "--no-cache")) {
				t.Fatalf("compiler option recipe: %q %v", resolved, err)
			}
			for j, p := range r.Schema.Params {
				if j == len(r.Schema.Params)-1 {
					if p.Name != "count" || p.Type != "s32" || p.ByteWidth != 4 || p.Alignment != 4 {
						t.Fatal(p)
					}
				} else if p.Type != "ptr" || !p.IsPointer || p.ByteWidth != 8 || p.Alignment != 8 {
					t.Fatal(p)
				}
			}
		}
		if requests[0].InputKey == requests[1].InputKey {
			t.Fatal("distinct entries share an input key")
		}
	}
}

func TestMHCCacheRequestMutations(t *testing.T) {
	c := mhcCacheTestConfig()
	requests, err := mhcCacheRequests(c, 12, 1, 13, 0)
	if err != nil {
		t.Fatal(err)
	}
	base := requests[1]
	for _, tt := range []struct {
		name   string
		mutate func(*MHCConfig, *mhcCacheRequest)
	}{
		{"batch", func(c *MHCConfig, _ *mhcCacheRequest) { c.Batch++ }},
		{"width", func(c *MHCConfig, _ *mhcCacheRequest) { c.Width++ }},
		{"repeat", func(c *MHCConfig, _ *mhcCacheRequest) { c.Repeat++ }},
		{"epsilon bits", func(c *MHCConfig, _ *mhcCacheRequest) {
			c.Epsilon = math.Float32frombits(math.Float32bits(c.Epsilon) + 1)
		}},
		{"source", func(_ *MHCConfig, r *mhcCacheRequest) { r.Source.Source += "\n// changed\n" }},
		{"entry", func(_ *MHCConfig, r *mhcCacheRequest) { r.Source.Name += "_other" }},
		{"input order", func(_ *MHCConfig, r *mhcCacheRequest) {
			r.Source.Inputs[0], r.Source.Inputs[1] = r.Source.Inputs[1], r.Source.Inputs[0]
		}},
		{"output", func(_ *MHCConfig, r *mhcCacheRequest) { r.Source.Outputs[0] = "other" }},
		{"abi", func(_ *MHCConfig, r *mhcCacheRequest) { r.Schema.Params[0].Alignment = 16 }},
		{"layout", func(_ *MHCConfig, r *mhcCacheRequest) { r.Layout += "/column-major" }},
		{"target", func(_ *MHCConfig, r *mhcCacheRequest) { r.Target = "compute_100" }},
		{"ordered flags", func(_ *MHCConfig, r *mhcCacheRequest) { r.Flags[1], r.Flags[2] = r.Flags[2], r.Flags[1] }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := base
			r.Source.Inputs = slices.Clone(r.Source.Inputs)
			r.Source.Outputs = slices.Clone(r.Source.Outputs)
			r.Schema.Params = slices.Clone(r.Schema.Params)
			r.Flags = slices.Clone(r.Flags)
			c := c
			tt.mutate(&c, &r)
			if mhcCacheInputKey(c, r) == base.InputKey {
				t.Fatal("changed request retained its identity")
			}
		})
	}
}

// These files stand only for host metadata. They are never loaded as tools.
func mhcCacheMetadataTools(t *testing.T) []cudacache.ToolDigest {
	t.Helper()
	var tools []cudacache.ToolDigest
	for _, name := range []string{"nvrtc", "nvrtc-builtins"} {
		path := filepath.Join(t.TempDir(), name)
		if err := os.WriteFile(path, []byte("host metadata fixture for "+name), 0o600); err != nil {
			t.Fatal(err)
		}
		d, err := cudacache.DigestTool(name, path)
		if err != nil {
			t.Fatal(err)
		}
		tools = append(tools, d)
	}
	return tools
}

func TestMHCCacheToolAdmission(t *testing.T) {
	tools := mhcCacheMetadataTools(t)
	if err := checkMHCCacheTools(tools); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name  string
		tools []cudacache.ToolDigest
	}{
		{"none", nil},
		{"missing builtins", tools[:1]},
		{"duplicate", []cudacache.ToolDigest{tools[0], tools[0]}},
		{"extra", append(slices.Clone(tools), tools[0])},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := checkMHCCacheTools(tt.tools); !errors.Is(err, ErrMHCCompilerCacheUnavailable) {
				t.Fatalf("incomplete tool closure: %v", err)
			}
		})
	}
	if err := os.WriteFile(tools[1].File.Path, []byte("changed actual fixture contents"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := checkMHCCacheTools(tools); !errors.Is(err, ErrMHCCompilerCacheUnavailable) || !errors.Is(err, cudacache.ErrToolIdentityChanged) {
		t.Fatalf("changed tool identity: %v", err)
	}
}

func TestMHCCachePublicRefusalDoesNotLoad(t *testing.T) {
	before := nvrtc.Loaded()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, tt := range []struct {
		name string
		ctx  context.Context
		want error
	}{
		{"nil adapter", context.Background(), ErrMHCCompilerCacheUnavailable},
		{"cancellation", ctx, context.Canceled},
	} {
		t.Run(tt.name, func(t *testing.T) {
			out, err := CompileMHCCached(tt.ctx, mhcCacheTestConfig(), 12, 1, nil, cudacache.ToolProvenance{})
			if !errors.Is(err, tt.want) || out[0] != nil || out[1] != nil {
				t.Fatalf("out=%v error=%v", out, err)
			}
		})
	}
	if nvrtc.Loaded() != before {
		t.Fatal("refusal loaded the compiler")
	}
}

func mhcCacheMetadataArtifact(t *testing.T, r mhcCacheRequest, id cudacache.CompilerIdentity) *cudakernel.Artifact {
	t.Helper()
	// A host-only metadata fixture, not an NVRTC compilation result or
	// an executable claim. The real compile fixture is separately gated.
	schema := r.Schema
	schema.Params = slices.Clone(schema.Params)
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_80\n.address_size 64\n"), r.Source.Name, schema, r.Target,
		cudakernel.WithCompilerOptions(r.Flags), cudakernel.WithDependencies(mhcCacheDependencies(r, id)))
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(mhcCacheBuild{mhcCacheRecipe, mhcCacheDigest("host metadata compiler-key fixture"), id.Provenance.Fingerprint()})
	if err != nil {
		t.Fatal(err)
	}
	a.Toolchain = string(b)
	return a
}

func TestMHCCacheAdapterMetadataRefusal(t *testing.T) {
	requests, err := mhcCacheRequests(mhcCacheTestConfig(), 12, 1, 13, 0)
	if err != nil {
		t.Fatal(err)
	}
	r := requests[1]
	prov := cudacache.ToolProvenance{Recipe: mhcCacheRecipe, Tools: mhcCacheMetadataTools(t), HeadersSHA256: mhcCacheDigest("no-external-headers")}
	id := r.identity(13, 0, "host-metadata-no-driver", prov)
	for _, tt := range []struct {
		name   string
		mutate func(*cudakernel.Artifact)
	}{
		{"entry", func(a *cudakernel.Artifact) { a.EntryPoint = "other_entry" }},
		{"abi", func(a *cudakernel.Artifact) { a.Schema.Params[0].Name = "other_input" }},
		{"flags", func(a *cudakernel.Artifact) {
			a.CompilerOpts[1], a.CompilerOpts[2] = a.CompilerOpts[2], a.CompilerOpts[1]
		}},
		{"source", func(a *cudakernel.Artifact) { a.Dependencies[0].Digest = mhcCacheDigest("different source") }},
		{"layout", func(a *cudakernel.Artifact) { a.Dependencies[3].Digest = mhcCacheDigest("different layout") }},
		{"tool receipt", func(a *cudakernel.Artifact) { a.Toolchain = "nvrtc-default" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := cudacache.Config{RootDir: t.TempDir()}
			c, err := cudacache.New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			a := mhcCacheMetadataArtifact(t, r, id)
			if err := checkMHCCacheArtifact(r, id, a); err != nil {
				t.Fatal(err)
			}
			tt.mutate(a)
			adapter := cudacache.NewNVRTCAdapter(c)
			if err := adapter.Store(context.Background(), r.InputKey, id, a); err != nil {
				t.Fatalf("adapter itself must accept this metadata control: %v", err)
			}
			fresh, err := cudacache.New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			adapter = cudacache.NewNVRTCAdapter(fresh)
			if _, hit, err := adapter.Lookup(context.Background(), r.InputKey, id); err != nil || !hit {
				t.Fatalf("metadata must reach consumer on an actual adapter hit: hit=%v error=%v", hit, err)
			}
			if a, err := compileMHCCachedEntry(context.Background(), adapter, r, id); a != nil || !errors.Is(err, cudacache.ErrCorruptArtifact) {
				t.Fatalf("consumer accepted incompatible cached metadata: artifact=%v error=%v", a, err)
			}
		})
	}
	changed := id
	changed.Flags = slices.Clone(id.Flags)
	changed.Flags[0], changed.Flags[1] = changed.Flags[1], changed.Flags[0]
	if cudacache.CompilerCacheKey(r.InputKey, id) == cudacache.CompilerCacheKey(r.InputKey, changed) {
		t.Fatal("ordered flags did not alter adapter identity")
	}
}
