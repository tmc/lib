//go:build linux && native

package cudahotswap

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// This test is a finite producer, not a qualification receipt. Root dispatch
// must bind its source, compiler libraries, actual test binary and GPU lease.
func TestDS4IncludePrecedenceNative(t *testing.T) {
	if os.Getenv("DS4_INCLUDE_PRECEDENCE") != "1" {
		t.Skip("DS4_INCLUDE_PRECEDENCE=1 requires reviewed native dispatch")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native header control requires inherited gpu-lock owner: %v", err)
	}
	artifacts := os.Getenv("DS4_INCLUDE_ARTIFACTS")
	if artifacts == "" {
		t.Fatal("DS4_INCLUDE_ARTIFACTS must retain source and compiler artifacts")
	}
	if err := os.MkdirAll(artifacts, 0700); err != nil {
		t.Fatal(err)
	}
	dev, err := cuda.NewDevice(0)
	var modules []*cuda.Module
	var out cudamem.Slice[int32]
	var retryOwners ds4RetryOwners
	// Adopt every partial owner before checking its error. Stop cleanup on an
	// unresolved child so the device cannot close beneath retained resources.
	closeOwners := func() error {
		for len(modules) > 0 {
			if err := modules[len(modules)-1].Close(); err != nil {
				return err
			}
			modules = modules[:len(modules)-1]
		}
		if out.Ptr() != 0 {
			if err := out.Free(); err != nil {
				return err
			}
			out = cudamem.Slice[int32]{}
		}
		if dev != nil {
			if err := dev.Close(); err != nil {
				return err
			}
			dev = nil
		}
		return nil
	}
	t.Cleanup(func() {
		if err := retryOwners.closeBefore(closeOwners); err != nil {
			t.Errorf("cleanup retained owners: %v", err)
			if err := retryOwners.closeBefore(closeOwners); err != nil {
				t.Errorf("cleanup retry retained owners: %v", err)
			}
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	nvMajor, nvMinor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	out, err = cudamem.Alloc[int32](dev, 3)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	header := filepath.Join(dir, "choice.h")
	for _, absolute := range []bool{false, true} {
		name := "choice.h"
		if absolute {
			name = header
		}
		for _, tt := range []struct {
			label  string
			disk   int
			memory bool
			want   int32
		}{
			{"competing", 11, true, 7},
			{"disk_only", 11, false, 11},
			{"edited_competing", 13, true, 7},
			{"edited_disk_only", 13, false, 13},
		} {
			if !t.Run(fmt.Sprintf("absolute_%t_%s", absolute, tt.label), func(t *testing.T) {
				disk := fmt.Sprintf("#define HEADER_VALUE %d\n", tt.disk)
				memory := "#define HEADER_VALUE 7\n"
				if err := os.WriteFile(header, []byte(disk), 0600); err != nil {
					t.Fatal(err)
				}
				src := fmt.Sprintf("#include %q\nextern \"C\" __global__ void choice(int *out) { out[1] = HEADER_VALUE; }\n", name)
				opts := []nvrtcx.Option{nvrtcx.WithName("ds4_header_control.cu"), nvrtcx.WithArch(major, minor), nvrtcx.WithOutput(nvrtcx.OutputCUBIN), nvrtcx.WithIncludePaths(dir), nvrtcx.WithFlag("--std=c++17"), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithNoCache()}
				if tt.memory {
					opts = append(opts, nvrtcx.WithHeader(name, memory))
				}
				flags, err := nvrtcx.CompileFlags(opts...)
				if err != nil {
					t.Fatal(err)
				}
				cm, err := nvrtcx.Compile(src, opts...)
				if err := retryOwners.adoptError(err); err != nil {
					t.Fatal(err)
				}
				caseDir := filepath.Join(artifacts, fmt.Sprintf("absolute_%t_%s", absolute, tt.label))
				if err := os.Mkdir(caseDir, 0700); err != nil {
					t.Fatal(err)
				}
				for file, data := range map[string][]byte{"source.cu": []byte(src), "disk.h": []byte(disk), "module.cubin": cm.CUBIN(), "module.ptx": []byte(cm.PTX()), "compile.log": []byte(cm.Log())} {
					if err := os.WriteFile(filepath.Join(caseDir, file), data, 0600); err != nil {
						t.Fatal(err)
					}
				}
				if tt.memory {
					if err := os.WriteFile(filepath.Join(caseDir, "memory.h"), []byte(memory), 0600); err != nil {
						t.Fatal(err)
					}
				}
				mod, _, err := cm.LoadOwnedOutput(dev, nvrtcx.OutputCUBIN)
				if mod != nil {
					modules = append(modules, mod)
				}
				if err := retryOwners.adoptError(err); err != nil {
					t.Fatal(err)
				}
				fn, err := mod.Function("choice")
				if err != nil {
					t.Fatal(err)
				}
				if err := out.CopyFromHost([]int32{991, -123, 992}); err != nil {
					t.Fatal(err)
				}
				ptr := out.Ptr()
				if err := fn.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&ptr)}); err != nil {
					t.Fatal(err)
				}
				if err := dev.WithContext(driver.CtxSynchronize); err != nil {
					t.Fatal(err)
				}
				got := make([]int32, 3)
				if err := out.CopyToHost(got); err != nil {
					t.Fatal(err)
				}
				receipt := struct {
					Flags       []string
					Header      string
					Memory      bool
					Compiler    [2]int32
					Target      [2]int
					Key         string
					CUBINSHA256 string
					Output      []int32
				}{flags, name, tt.memory, [2]int32{nvMajor, nvMinor}, [2]int{major, minor}, cm.Key(), fmt.Sprintf("%x", sha256.Sum256(cm.CUBIN())), got}
				data, err := json.MarshalIndent(receipt, "", "  ")
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(caseDir, "receipt.json"), data, 0600); err != nil {
					t.Fatal(err)
				}
				t.Log(string(data))
				if got[0] != 991 || got[1] != tt.want || got[2] != 992 {
					t.Fatalf("actual loaded CUBIN output %v; want [991 %d 992]", got, tt.want)
				}
			}) {
				return // A failed native arm must not dispatch another arm.
			}
		}
	}
}
