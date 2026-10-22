//go:build linux

package main

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestShapeGroupingPhysical(t *testing.T) {
	if os.Getenv("DS4_GROUP_PHYSICAL") != "1" {
		t.Skip("physical gate not requested")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("missing live lock receipt")
	}
	dir := os.Getenv("DS4_GROUP_RECEIPT")
	if dir == "" {
		t.Fatal("missing receipt directory")
	}
	mem, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		t.Fatal(err)
	}
	available := uint64(0)
	for _, line := range strings.Split(string(mem), "\n") {
		if strings.HasPrefix(line, "MemAvailable:") {
			fields := strings.Fields(line)
			available, err = strconv.ParseUint(fields[1], 10, 64)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	if available < 4*1024*1024 {
		t.Fatal("host free-memory guard below4GiB")
	}
	if err := os.WriteFile(filepath.Join(dir, "meminfo.before"), mem, 0600); err != nil {
		t.Fatal(err)
	}
	d, err := cuda.NewDevice(0)
	owner := &shapeStateNativeOwner{parent: func() error {
		if d == nil {
			return nil
		}
		return d.Close()
	}, drain: func() error {
		if d == nil {
			return nil // Partial constructor owners drain their own resources.
		}
		return d.WithContext(driver.CtxSynchronize)
	}}
	t.Cleanup(func() {
		if err := owner.Close(); err != nil {
			shapeStateNativeRetained = append(shapeStateNativeRetained, owner)
			t.Errorf("group owner close: %v", err)
		}
	})
	if err != nil {
		owner.retainError(err)
		t.Fatal(err)
	}
	if d == nil {
		t.Fatal("missing successful device owner")
	}
	name, err := d.Name()
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := d.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	driverVersion, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	if driver.Path() != "/usr/lib/aarch64-linux-gnu/libcuda.so.1" {
		t.Fatal("unqualified driver loader path")
	}
	var free uint
	if err := d.WithContext(func() error { var err error; free, _, err = driver.MemGetInfo(); return err }); err != nil {
		t.Fatal(err)
	}
	if free < 256<<20 {
		t.Fatal("device free-memory guard below256MiB")
	}
	t.Logf("host_available_kib=%d device_free_bytes=%d bounded_buffer_limit=65536", available, free)
	source := cudaShapeGroupSource
	if path := os.Getenv("DS4_GROUP_SOURCE"); path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		source = string(b)
	}
	const library = "/usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvrtc.so"
	const builtin = "/usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvrtc-builtins.so"
	const header = "/usr/local/cuda-13.0/include/nvrtc.h"
	for path, want := range map[string]string{library: "64db9c3fc128f4a57a22c314fda08831a8e3a725d439a70076a0fb98f7bea1b7", builtin: "84f1c5031bfe1146375b367b2c7fb8d31ea00430be7172177eeb5d04ab813b69", header: "9e4415b5ff5a2c58ec2b1a02eefcc743187c94ce48b00743c68092efe8c0f86e"} {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(b)) != want {
			t.Fatalf("compiler dependency seal differs: %s", path)
		}
	}
	if os.Getenv("NVRTC_LIB_PATH") != library || strings.Contains(source, "#include") {
		t.Fatal("unqualified compiler/header selection")
	}
	opts := []nvrtcx.Option{nvrtcx.WithName("cuda_shape_group.cu"), nvrtcx.WithDeviceArch(d), nvrtcx.WithIncludePaths("/usr/local/cuda-13.0/include")}
	flags, err := nvrtcx.CompileFlags(opts...)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("compile_flags=%q source_includes=none nvrtc_api_header_sealed_not_source_include", flags)
	program, err := nvrtcx.Compile(source, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if len(program.CUBIN()) == 0 {
		t.Fatal("physical arch compilation did not produce CUBIN")
	}
	t.Log("COMPILE_CUBIN_PASS")
	if err := os.WriteFile(filepath.Join(dir, "group.cubin"), program.CUBIN(), 0600); err != nil {
		t.Fatal(err)
	}
	vm, vn, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	if vm != 13 || vn != 0 || nvrtc.Path() != library {
		t.Fatalf("wrong loaded NVRTC: %s %d.%d", nvrtc.Path(), vm, vn)
	}
	t.Logf("device=%q cc=%d.%d driver=%d nvrtc=%d.%d source_sha256=%x cubin_sha256=%x driver_path=%q nvrtc_path=%q", name, major, minor, driverVersion, vm, vn, sha256.Sum256([]byte(source)), sha256.Sum256(program.CUBIN()), driver.Path(), nvrtc.Path())
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "maps.txt"), maps, 0600); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	qualifiedCompiler, qualifiedBuiltins := false, false
	for _, line := range strings.Split(string(maps), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		p := fields[5]
		if !strings.Contains(p, "libcuda") && !strings.Contains(p, "libnvrtc") {
			continue
		}
		if seen[p] {
			continue
		}
		seen[p] = true
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		hash := fmt.Sprintf("%x", sha256.Sum256(b))
		if strings.Contains(p, "libnvrtc-builtins") {
			if hash != "84f1c5031bfe1146375b367b2c7fb8d31ea00430be7172177eeb5d04ab813b69" {
				t.Fatal("loaded builtins seal mismatch")
			}
			qualifiedBuiltins = true
		} else if strings.Contains(p, "libnvrtc") {
			if hash != "64db9c3fc128f4a57a22c314fda08831a8e3a725d439a70076a0fb98f7bea1b7" {
				t.Fatal("loaded compiler seal mismatch")
			}
			qualifiedCompiler = true
		} else if strings.Contains(p, "libcuda") && hash != "a13f3101b105930ba0796b1ab99721cbee3e19360ced21253e2a09e1a102fb99" {
			t.Fatal("loaded driver seal mismatch")
		}
		t.Logf("loaded_library=%q sha256=%x", p, sha256.Sum256(b))
	}
	if len(seen) < 2 || !qualifiedCompiler || !qualifiedBuiltins {
		t.Fatal("loaded library provenance incomplete")
	}
	for _, variant := range []dsv4shape.Variant{dsv4shape.Flash, dsv4shape.Pro} {
		c, _ := dsv4shape.For(variant)
		for _, invalid := range []bool{false, true} {
			t.Run(fmt.Sprintf("experts%d_invalid%v", c.Experts, invalid), func(t *testing.T) {
				r := &cudaFoundation{device: d}
				plan, err := planShapeGrouping(c, 2)
				if err != nil {
					t.Fatal(err)
				}
				if uint64(plan.offsets+plan.pairs+plan.active+2+plan.pairs)*4 > 65536 {
					t.Fatal("bounded buffer guard exceeded")
				}
				ops := shapeGroupingDefaults(r)
				ops.load = func() (driver.Module, driver.Function, error) {
					m, _, err := program.LoadOutputWithLog(d, nvrtcx.OutputCUBIN)
					if err != nil {
						return m, 0, err
					}
					var fn driver.Function
					err = r.withContext(func() error { var err error; fn, err = m.Function("ds4_shape_group_csr"); return err })
					return m, fn, err
				}
				s, err := newCUDAShapeGroupingWithOperations(r, c, 2, ops)
				if err != nil {
					owner.retainError(err)
					t.Fatal(err)
				}
				owner.children = append(owner.children, s)
				t.Log("LOAD_KERNEL_PASS")
				values := []int32{int32(c.Experts - 1), 0, 1, 255, 0, 127, int32(c.Experts - 1), 1, 7, 127, 3, 255}
				if invalid {
					values[1] = -1
					values[7] = int32(c.Experts)
				}
				ids, err := cudamem.Alloc[int32](d, len(values))
				owner.retainAllocation(ids.Ptr() != 0, ids.Free, err)
				if err != nil {
					t.Fatal(err)
				}
				if err := ids.CopyFromHost(values); err != nil {
					t.Fatal(err)
				}
				err = s.EncodeAndValidate(ids)
				if invalid {
					if err == nil || !strings.Contains(err.Error(), "invalid ids") {
						t.Fatalf("CSR_INVALID_REJECTION: invalid input not refused: %v", err)
					}
				} else if err != nil {
					t.Fatal(err)
				}
				offsets := make([]int32, c.Experts+1)
				var pairs, active []int32
				invalidCount := int32(0)
				for _, id := range values {
					if id < 0 || id >= int32(c.Experts) {
						invalidCount++
					}
				}
				for e := 0; e < c.Experts; e++ {
					offsets[e] = int32(len(pairs))
					before := len(pairs)
					for p, id := range values {
						if id == int32(e) {
							pairs = append(pairs, int32(p))
						}
					}
					if len(pairs) != before {
						active = append(active, int32(e))
					}
				}
				offsets[c.Experts] = int32(len(pairs))
				for _, check := range []struct {
					name   string
					device cudamem.Slice[int32]
					want   []int32
				}{{"offsets", s.offsets, offsets}, {"pairs", s.pairs, pairs}, {"active", s.active, active}, {"count", s.count, []int32{int32(len(active))}}, {"invalid", s.invalid, []int32{invalidCount}}} {
					got := make([]int32, check.device.Len())
					if err := check.device.CopyToHost(got); err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(got[:len(check.want)], check.want) {
						t.Fatalf("CSR_ORACLE_MISMATCH %s: got %v want %v", check.name, got, check.want)
					}
					t.Logf("%s=%v", check.name, got[:len(check.want)])
				}
			})
		}
	}
}
