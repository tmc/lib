//go:build darwin || linux || freebsd || netbsd

package nvtelemetry

import (
	"bytes"
	"context"
	"encoding/json"
	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/dcgm"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestPolicyNativePublicPipeline(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix shared library fixture")
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("C compiler unavailable")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "policy.so")
	source := filepath.Join(dir, "policy.c")
	const code = `
#include <stdint.h>
typedef struct {uint32_t version;int32_t condition;uint64_t val[2];uint32_t gpu;} response;
typedef int (*callback)(response*,uint64_t);
static callback cb;static uint64_t token;
int dcgmPolicyRegister_v2(uintptr_t h,uintptr_t g,int32_t mask,callback f,uint64_t u){cb=f;token=u;return 0;}
int dcgmPolicyUnregister(uintptr_t h,uintptr_t g,int32_t mask){return 0;}
void fixture_emit(void){response r={0x2000020,64,{9007199254740993ULL,23},7};cb(&r,token);}
`
	if err := os.WriteFile(source, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	args := []string{"-shared", "-fPIC", "-o", path, source}
	if runtime.GOOS == "darwin" {
		args[0] = "-dynamiclib"
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, cc, args...).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v %s", err, out)
	}
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestPolicyNativePipelineChild$", "-test.count=1")
	cmd.Env = append(os.Environ(), "NVTELEMETRY_POLICY_FIXTURE="+path)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("pipeline: %v %s", err, out)
	}
}

func TestPolicyNativePipelineChild(t *testing.T) {
	path := os.Getenv("NVTELEMETRY_POLICY_FIXTURE")
	if path == "" {
		t.Skip("isolated fixture")
	}
	if err := dcgm.Load(path); err != nil {
		t.Fatal(err)
	}
	h, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		t.Fatal(err)
	}
	defer purego.Dlclose(h)
	var emit func()
	purego.RegisterLibFunc(&emit, h, "fixture_emit")
	n, err := dcgm.NewPolicyNotifications(1, 1, dcgm.PolicyCondXid, 1)
	if err != nil {
		if n != nil {
			n.Close()
		}
		t.Fatal(err)
	}
	s, err := NewDCGMPolicySource(n, policyClock(), policyConfig())
	if err != nil {
		n.Close()
		t.Fatal(err)
	}
	defer s.Close()
	emit()
	event, err := s.Next(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	if err := WriteDCGMPolicyEvent(&b, event); err != nil {
		t.Fatal(err)
	}
	var result struct {
		Raw    [2]uint64 `json:"raw_payload"`
		Device string    `json:"device_uuid"`
	}
	if err := json.Unmarshal(b.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Raw[0] != 9007199254740993 || result.Device != "GPU-a" {
		t.Fatal(result)
	}
}
