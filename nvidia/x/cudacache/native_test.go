package cudacache_test

import (
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudacache"
)

// TestNativeProbeToolchain probes the NVRTC and driver libraries this
// process loads. It runs only with CUDACACHE_NATIVE_TOOLCHAIN=1, and checks
// the NVRTC digest against CUDACACHE_NATIVE_NVRTC_SHA256, taken outside the
// process (sha256sum of the file the soname resolves to).
func TestNativeProbeToolchain(t *testing.T) {
	if os.Getenv("CUDACACHE_NATIVE_TOOLCHAIN") != "1" {
		t.Skip("CUDACACHE_NATIVE_TOOLCHAIN=1 not set; needs libnvrtc and libcuda")
	}
	want := os.Getenv("CUDACACHE_NATIVE_NVRTC_SHA256")
	if want == "" {
		t.Fatal("CUDACACHE_NATIVE_NVRTC_SHA256 not set; the NVRTC digest needs an oracle from outside the process")
	}
	if err := nvrtc.Load(""); err != nil {
		t.Fatalf("CUDACACHE_NATIVE_TOOLCHAIN=1 but libnvrtc does not load: %v", err)
	}
	if err := driver.Load(""); err != nil {
		t.Fatalf("CUDACACHE_NATIVE_TOOLCHAIN=1 but libcuda does not load: %v", err)
	}
	p, err := cudacache.ProbeToolchain()
	if err != nil {
		t.Fatalf("ProbeToolchain: %v", err)
	}
	tools := map[string]cudacache.ToolDigest{}
	for _, d := range p.Tools {
		tools[d.Name] = d
	}
	for _, name := range []string{"nvrtc", "driver"} {
		d, ok := tools[name]
		if !ok {
			t.Fatalf("ProbeToolchain = %+v: no %s tool", p, name)
		}
		if err := cudacache.VerifyTool(d); err != nil {
			t.Errorf("VerifyTool(%s): %v", name, err)
		}
		t.Logf("%s: %s sha256 %s", name, d.File.Path, d.SHA256)
	}
	if got := tools["nvrtc"].SHA256; got != want {
		t.Errorf("nvrtc sha256 = %s, want %s (CUDACACHE_NATIVE_NVRTC_SHA256)", got, want)
	}
	if p.DriverVersion <= 0 {
		t.Errorf("DriverVersion = %d, want > 0", p.DriverVersion)
	}
	id, err := cudacache.CurrentEnvironmentIdentity("sm_121a", nil, 0)
	if err != nil {
		t.Fatalf("CurrentEnvironmentIdentity: %v", err)
	}
	if id.Provenance.Fingerprint() != p.Fingerprint() {
		t.Errorf("CurrentEnvironmentIdentity fingerprint %s, ProbeToolchain %s", id.Provenance.Fingerprint(), p.Fingerprint())
	}
}
