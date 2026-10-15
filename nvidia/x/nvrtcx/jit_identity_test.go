package nvrtcx_test

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestJITPublicZeroProducer(t *testing.T) {
	flags, jit := nvrtcx.StripOfflineFlags([]string{"-O0"})
	if len(flags) != 0 || jit.OptLevel != 0 || !jit.OptLevelSet {
		t.Fatal(flags, jit)
	}
	// Explicit descriptor bytes are a storage fixture, not compiler output.
	art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("fixture ptx"), "k", cudakernel.Schema{}, "compute_80", cudakernel.WithJIT(jit))
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(art)
	if err != nil {
		t.Fatal(err)
	}
	var got cudakernel.Artifact
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	opts, vals := got.JITOptions.DriverOptions()
	if !reflect.DeepEqual(opts, []driver.JITOption{driver.JITOptimizationLevel}) || !reflect.DeepEqual(vals, []uintptr{0}) {
		t.Fatal(opts, vals)
	}
	_, unset := nvrtcx.StripOfflineFlags(nil)
	if opts, _ := unset.DriverOptions(); len(opts) != 0 {
		t.Fatal("unset emits an optimization option")
	}
}

func TestJITActualResolutionAndKey(t *testing.T) {
	tests := []struct {
		name    string
		opts    []nvrtcx.Option
		level   uint32
		present bool
	}{
		{"unset", nil, 0, false},
		{"zero", []nvrtcx.Option{nvrtcx.WithJITOptions(cudakernel.JITOptions{OptLevelSet: true})}, 0, true},
		{"legacy4", []nvrtcx.Option{nvrtcx.WithJITOptions(cudakernel.JITOptions{OptLevel: 4})}, 4, true},
		{"marked4", []nvrtcx.Option{nvrtcx.WithJITOptions(cudakernel.JITOptions{OptLevel: 4, OptLevelSet: true})}, 4, true},
		{"offline0", []nvrtcx.Option{nvrtcx.WithFlag("-O0")}, 0, true},
		{"explicit0 overrides3", []nvrtcx.Option{nvrtcx.WithFlag("-O3"), nvrtcx.WithJITOptions(cudakernel.JITOptions{OptLevelSet: true})}, 0, true},
		{"last offline wins", []nvrtcx.Option{nvrtcx.WithFlag("-O3"), nvrtcx.WithFlag("-O0")}, 0, true},
	}
	keys := map[string]string{}
	for _, tc := range tests {
		jit, key, err := nvrtcx.ExportJITRequest(tc.opts...)
		if err != nil || jit.OptLevel != tc.level || (jit.OptLevelSet || jit.OptLevel > 0) != tc.present {
			t.Fatal(tc.name, jit, err)
		}
		keys[tc.name] = key
	}
	if keys["unset"] == keys["zero"] || keys["zero"] == keys["legacy4"] || keys["legacy4"] != keys["marked4"] || keys["zero"] != keys["offline0"] || keys["zero"] != keys["explicit0 overrides3"] {
		t.Fatal(keys)
	}
	_, _, err := nvrtcx.ExportJITRequest(nvrtcx.WithJITOptions(cudakernel.JITOptions{OptLevel: 5}))
	if err == nil {
		t.Fatal("invalid JIT admitted")
	}
	// This public Compile call must refuse before compiler probing or caching.
	if _, err := nvrtcx.Compile("fixture", nvrtcx.WithNoToolkitIncludes(), nvrtcx.WithJITOptions(cudakernel.JITOptions{OptLevel: 5})); err == nil {
		t.Fatal("invalid Compile request admitted")
	}
}

func TestJITCacheDoesNotShareMutableDescriptor(t *testing.T) {
	cm := nvrtcx.ExportNewCompiledModule(nvrtcx.OutputPTX, "fixture ptx", nil)
	zero := cudakernel.JITOptions{OptLevelSet: true}
	cm.SetJITOptions(zero)
	// The key is opaque here; this test exercises actual cache ownership.
	key := filepath.Base(t.TempDir())
	if err := nvrtcx.ExportStoreCache(key, cm); err != nil {
		t.Fatal(err)
	}
	cm.SetJITOptions(cudakernel.JITOptions{OptLevel: 4})
	got, ok := nvrtcx.ExportLookupCache(key)
	if !ok || got.JITOptions() != zero {
		t.Fatal("producer changed retained entry", got, ok)
	}
	got.SetJITOptions(cudakernel.JITOptions{})
	again, ok := nvrtcx.ExportLookupCache(key)
	if !ok || again.JITOptions() != zero || again == got {
		t.Fatal("reader changed retained entry", again, ok)
	}
}

func TestJITActualLoadAdmission(t *testing.T) {
	cm := nvrtcx.ExportNewCompiledModule(nvrtcx.OutputPTX, "fixture ptx", nil)
	cm.SetJITOptions(cudakernel.JITOptions{OptLevel: 5})
	// These public entrypoints must refuse invalid JIT metadata before the
	// nil-device gate. No driver call can be reached by these fixtures.
	_, load := cm.Load(nil)
	_, selected := cm.LoadOutput(nil, nvrtcx.OutputPTX)
	_, _, logged := cm.LoadOutputWithLog(nil, nvrtcx.OutputPTX)
	for _, err := range []error{load, selected, logged} {
		if err == nil || !strings.Contains(err.Error(), "optimization level") {
			t.Fatal("load dropped JIT options or validation", err)
		}
	}
}
