package cudapipeline_test

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

type pipelineAdmission struct {
	UUID  string
	Owner cudalock.Owner
}

func pipelineNativeAdmission(env func(string) string) (pipelineAdmission, error) {
	owner, err := cudalock.Inherited(env)
	if err != nil {
		return pipelineAdmission{}, err
	}
	uuid := env("CUDA_PIPELINE_COPY_UUID")
	raw := strings.ReplaceAll(strings.TrimPrefix(uuid, "GPU-"), "-", "")
	bytes, err := hex.DecodeString(raw)
	if err != nil || len(bytes) != 16 || uuid != fmt.Sprintf("GPU-%x-%x-%x-%x-%x", bytes[:4], bytes[4:6], bytes[6:8], bytes[8:10], bytes[10:]) {
		return pipelineAdmission{}, errors.New("pipeline native test requires a canonical CUDA_PIPELINE_COPY_UUID")
	}
	return pipelineAdmission{raw, owner}, nil
}

// Driver enumeration does not retain a primary context. The native fixture
// resolves the admitted UUID again in its own process before cudaSetDevice.
const pipelineDeviceAdmissionSource = `
#include <cuda.h>
#include <cstring>
static int admittedDevice(const char *expected, int *ordinal) {
 if(strlen(expected)!=32 || cuInit(0)!=CUDA_SUCCESS) return 1;
 int count=0, found=-1;
 if(cuDeviceGetCount(&count)!=CUDA_SUCCESS) return 2;
 for(int i=0;i<count;i++) {
  CUdevice dev; CUuuid uuid;
  if(cuDeviceGet(&dev,i)!=CUDA_SUCCESS || cuDeviceGetUuid(&uuid,dev)!=CUDA_SUCCESS) return 3;
  char raw[33]; const char *digits="0123456789abcdef";
  for(int j=0;j<16;j++){unsigned char b=(unsigned char)uuid.bytes[j];raw[2*j]=digits[b>>4];raw[2*j+1]=digits[b&15];}raw[32]=0;
  if(strcmp(raw,expected)==0){if(found>=0)return 4;found=i;}
 }
 if(found<0)return 5;
 *ordinal=found; return 0;
}
`

func pipelineTestLease(t *testing.T) (*cudalock.Handle, map[string]string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	h, err := cudalock.Acquire(ctx, cudalock.WithDir(filepath.Join(t.TempDir(), "lock")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := h.Release(); err != nil {
			t.Errorf("release test lease: %v", err)
		}
	})
	return h, map[string]string{"GPU_LOCK_HELD": strconv.Itoa(h.Owner().PID), "GPU_LOCKDIR": h.Dir(), "CUDA_PIPELINE_COPY_UUID": "GPU-00112233-4455-6677-8899-aabbccddeeff"}
}

func TestPipelineNativeAdmission(t *testing.T) {
	h, values := pipelineTestLease(t)
	env := func(k string) string { return values[k] }
	if got, err := pipelineNativeAdmission(env); err != nil || got.UUID != "00112233445566778899aabbccddeeff" {
		t.Fatalf("admission: %+v %v", got, err)
	}
	for _, value := range []string{"", "1", "999999999"} {
		values["GPU_LOCK_HELD"] = value
		if _, err := pipelineNativeAdmission(env); !errors.Is(err, cudalock.ErrNotInherited) {
			t.Fatalf("holder %q admitted: %v", value, err)
		}
	}
	values["GPU_LOCK_HELD"] = strconv.Itoa(h.Owner().PID)
	for _, uuid := range []string{"", "00112233445566778899aabbccddeeff", "GPU-00112233-4455-6677-8899-aabbccddeefg"} {
		values["CUDA_PIPELINE_COPY_UUID"] = uuid
		if _, err := pipelineNativeAdmission(env); err == nil {
			t.Fatalf("UUID %q admitted", uuid)
		}
	}
	values["CUDA_PIPELINE_COPY_UUID"] = "GPU-00112233-4455-6677-8899-aabbccddeeff"
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
	if _, err := pipelineNativeAdmission(env); !errors.Is(err, cudalock.ErrNotInherited) {
		t.Fatalf("released lease admitted: %v", err)
	}
}

// Compile and execute the exact device-selection helper with synthetic driver
// enumeration. No CUDA context or device is used by this host control.
func TestPipelineNativeDeviceSelection(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("C++ compiler unavailable")
	}
	dir := t.TempDir()
	header := `typedef int CUdevice;struct CUuuid { char bytes[16]; };enum{CUDA_SUCCESS=0};
static int cuInit(int){return 0;}static int cuDeviceGetCount(int *n){*n=2;return 0;}static int cuDeviceGet(CUdevice*d,int i){*d=i;return 0;}
static int cuDeviceGetUuid(CUuuid*u,CUdevice d){for(int i=0;i<16;i++)u->bytes[i]=(char)(d+i);return 0;}`
	source := pipelineDeviceAdmissionSource + `
int main(){int ordinal=-1;if(admittedDevice("0102030405060708090a0b0c0d0e0f10",&ordinal)!=0||ordinal!=1)return 1;
if(admittedDevice("ffffffffffffffffffffffffffffffff",&ordinal)==0)return 2;
if(admittedDevice("short",&ordinal)==0)return 3;return 0;}`
	if err := os.WriteFile(filepath.Join(dir, "cuda.h"), []byte(header), 0600); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "admission.cc")
	bin := filepath.Join(dir, "admission")
	if err := os.WriteFile(path, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-I", dir, path, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile selection control: %v %s", err, out)
	}
	if out, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
		t.Fatalf("device selection control: %v %s", err, out)
	}
}
