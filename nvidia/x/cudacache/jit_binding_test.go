package cudacache_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestJITPublicProducerManifestRoundTrip(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	c, err := cudacache.New(cudacache.Config{RootDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	adapter := cudacache.NewNVRTCAdapter(c)
	_, zero := nvrtcx.StripOfflineFlags([]string{"-O0"})
	if !zero.OptLevelSet {
		t.Fatal("public zero producer lost presence")
	}
	levels := []cudakernel.JITOptions{{}, zero, {OptLevel: 4}, {OptLevel: 4, OptLevelSet: true}}
	keys := make([]string, len(levels))
	for i, jit := range levels {
		id := cudacache.CompilerIdentity{TargetArch: "compute_80", OutputKind: nvrtcx.OutputPTX, JITOptions: jit}
		keys[i] = cudacache.CompilerCacheKey("same source", id)
		// A descriptor storage fixture, with no asserted compiler provenance.
		art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("fixture ptx"), "k", cudakernel.Schema{}, "compute_80", cudakernel.WithJIT(jit))
		if err != nil {
			t.Fatal(err)
		}
		if err := adapter.Store(ctx, "same source", id, art); err != nil {
			t.Fatal(err)
		}
		fresh, err := cudacache.New(cudacache.Config{RootDir: dir})
		if err != nil {
			t.Fatal(err)
		}
		got, hit, err := cudacache.NewNVRTCAdapter(fresh).Lookup(ctx, "same source", id)
		if err != nil || !hit || got.JITOptions != jit {
			t.Fatal(got, hit, err)
		}
		opts, vals := got.JITOptions.DriverOptions()
		if i == 0 {
			if len(opts) != 0 {
				t.Fatal(opts)
			}
		} else if len(opts) != 1 || opts[0] != driver.JITOptimizationLevel || vals[0] != uintptr(jit.OptLevel) {
			t.Fatal(opts, vals)
		}
	}
	if keys[0] == keys[1] || keys[1] == keys[2] || keys[2] != keys[3] {
		t.Fatal(keys)
	}
}

func TestJITAdapterRejectsWrongIdentityAndTampering(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	c, err := cudacache.New(cudacache.Config{RootDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	id := cudacache.CompilerIdentity{TargetArch: "compute_80", OutputKind: nvrtcx.OutputPTX, JITOptions: cudakernel.JITOptions{OptLevelSet: true}}
	art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("fixture ptx"), "k", cudakernel.Schema{}, "compute_80")
	if err != nil {
		t.Fatal(err)
	}
	adapter := cudacache.NewNVRTCAdapter(c)
	if err := adapter.Store(ctx, "source", id, art); err == nil {
		t.Fatal("unset descriptor stored for explicit zero")
	}
	entries, _ := filepath.Glob(filepath.Join(dir, "manifests", "*.manifest"))
	if len(entries) != 0 {
		t.Fatal("failed Store published manifest", entries)
	}
	art.JITOptions = id.JITOptions
	if err := adapter.Store(ctx, "source", id, art); err != nil {
		t.Fatal(err)
	}
	key := cudacache.CompilerCacheKey("source", id)
	path := filepath.Join(dir, "artifacts", key+"-"+art.Digest+".artifact")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var stored cudakernel.Artifact
	if err := json.Unmarshal(data, &stored); err != nil {
		t.Fatal(err)
	}
	stored.JITOptions.OptLevelSet = false // Raw payload and its digest stay unchanged.
	data, err = json.Marshal(stored)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	fresh, err := cudacache.New(cudacache.Config{RootDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	got, hit, err := cudacache.NewNVRTCAdapter(fresh).Lookup(ctx, "source", id)
	if got != nil || hit || !errors.Is(err, cudacache.ErrCorruptArtifact) {
		t.Fatal(got, hit, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "manifests", key+".manifest")); !os.IsNotExist(err) {
		t.Fatal("wrong identity was not quarantined", err)
	}
	id.JITOptions.OptLevel = 5
	if _, _, err := adapter.Lookup(ctx, "source", id); err == nil {
		t.Fatal("invalid identity admitted")
	}
}
