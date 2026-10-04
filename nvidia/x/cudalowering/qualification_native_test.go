//go:build native

package cudalowering

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvvm"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvvmx"
)

func qualificationWrite(t *testing.T, dir, name string, b []byte) {
	t.Helper()
	f, err := os.OpenFile(filepath.Join(dir, name), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.Write(b)
	closeErr := f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
}

func qualificationJSON(t *testing.T, dir, name string, v any) {
	t.Helper()
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	qualificationWrite(t, dir, name, append(b, '\n'))
}

func qualificationToolIdentity(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if !s.Mode().IsRegular() || s.Size() == 0 || s.Size() > 256<<20 {
		t.Fatal("invalid tool size or kind")
	}
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(f, (256<<20)+1))
	if err != nil || n != s.Size() {
		t.Fatalf("tool changed during identity read: %v", err)
	}
	return fmt.Sprintf("%s sha256=%x size=%d", path, h.Sum(nil), n)
}

// Preparation calls NVVM and offline tools only; it never initializes CUDA.
func TestAddrSpaceQualification_Prepare(t *testing.T) {
	if os.Getenv("ADDRSPACE_QUALIFICATION_PREPARE") != "1" {
		t.Skip("explicit offline preparation required")
	}
	dir := os.Getenv("ADDRSPACE_QUALIFICATION_ARTIFACTS")
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute artifact directory required")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	a, b, err := qualificationIR()
	if err != nil {
		t.Fatal(err)
	}
	build := qualificationBuild{Target: "sm_121", Optimization: 0, Tools: make(map[string]string)}
	tools := map[string]string{}
	for _, name := range []string{"ptxas", "nvdisasm"} {
		path, err := exec.LookPath(name)
		if err != nil {
			t.Fatal(err)
		}
		path, err = filepath.Abs(path)
		if err != nil {
			t.Fatal(err)
		}
		tools[name] = path
		build.Tools[name] = qualificationToolIdentity(t, path)
	}
	major, minor, err := nvvm.Version()
	if err != nil {
		t.Fatal(err)
	}
	build.Tools["nvvm"] = fmt.Sprintf("%s version=%d.%d", qualificationToolIdentity(t, nvvm.Path()), major, minor)
	run := func(name string, args ...string) []byte {
		build.Commands = append(build.Commands, append([]string{tools[name]}, args...))
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, tools[name], args...).CombinedOutput()
		if err != nil {
			t.Fatalf("%s %v: %v\n%s", name, args, err, out)
		}
		return out
	}
	for i, ir := range [][]byte{a, b} {
		name := []string{"before", "after"}[i]
		qualificationWrite(t, dir, name+".ll", ir)
		r, err := nvvmx.CompileWithOptions([]nvvmx.Input{{Name: name, Data: ir}}, nvvmx.WithArch("compute_121"), nvvmx.WithOpt(0))
		if err != nil {
			t.Fatal(err)
		}
		ptx := append([]byte(nil), r.PTX...)
		if err := r.Close(); err != nil {
			t.Fatal(err)
		}
		qualificationWrite(t, dir, name+".ptx", ptx)
		qualificationWrite(t, dir, name+"-nvvm.log", []byte(fmt.Sprintf("%s input=%s arch=compute_121 opt=0 ptx=%s\n", build.Tools["nvvm"], qualificationDigest(ir), qualificationDigest(ptx))))
		out := run("ptxas", "-arch=sm_121", filepath.Join(dir, name+".ptx"), "-o", filepath.Join(dir, name+".cubin"))
		qualificationWrite(t, dir, name+"-ptxas.log", append(out, '\n'))
		sass := run("nvdisasm", "-c", filepath.Join(dir, name+".cubin"))
		qualificationWrite(t, dir, name+".sass", sass)
		cubin, err := qualificationRead(filepath.Join(dir, name+".cubin"))
		if err != nil {
			t.Fatal(err)
		}
		pc, sc, observation, err := qualificationInspect(name, ptx, sass)
		if err != nil {
			t.Fatal(err)
		}
		build.Images = append(build.Images, qualificationImage{name, qualificationDigest(ir), qualificationDigest(ptx), qualificationDigest(cubin), qualificationDigest(sass), pc, sc, observation})
	}
	if err := qualificationCheckBuild(build); err != nil {
		t.Fatal(err)
	}
	qualificationJSON(t, dir, "build.json", build)
}

// The opt0 row establishes a conversion mechanism, not optimized-code benefit.
func TestAddrSpaceQualification_ABSemanticsAndCompiledConversion(t *testing.T) {
	if os.Getenv("ADDRSPACE_QUALIFICATION_NATIVE") != "1" {
		t.Skip("explicit native qualification required")
	}
	pid, err := strconv.Atoi(os.Getenv("GPU_LOCK_HELD"))
	if err != nil || pid <= 0 {
		t.Fatal("requires actual gpu-lock holder pid")
	}
	dir, outDir := os.Getenv("ADDRSPACE_QUALIFICATION_ARTIFACTS"), os.Getenv("ADDRSPACE_QUALIFICATION_OUTPUT")
	if !filepath.IsAbs(dir) || !filepath.IsAbs(outDir) {
		t.Fatal("absolute artifact/output directories required")
	}
	if err := os.MkdirAll(outDir, 0700); err != nil {
		t.Fatal(err)
	}
	raw, err := qualificationRead(filepath.Join(dir, "build.json"))
	if err != nil {
		t.Fatal(err)
	}
	var build qualificationBuild
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&build); err != nil {
		t.Fatal(err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		t.Fatal("trailing build JSON")
	}
	if err := qualificationCheckBuild(build); err != nil {
		t.Fatal(err)
	}
	a, b, err := qualificationIR()
	if err != nil {
		t.Fatal(err)
	}
	images := make([][]byte, 2)
	for i, v := range build.Images {
		for j, ext := range []string{"ll", "ptx", "cubin", "sass"} {
			data, err := qualificationRead(filepath.Join(dir, v.Name+"."+ext))
			if err != nil {
				t.Fatal(err)
			}
			want := []string{v.IRDigest, v.PTXDigest, v.CUBINDigest, v.SASSDigest}[j]
			if qualificationDigest(data) != want {
				t.Fatalf("stale %s %s binding", v.Name, ext)
			}
			if j == 0 && !bytes.Equal(data, [][]byte{a, b}[i]) {
				t.Fatal("IR not current explicit shared-cast workload")
			}
			if j == 2 {
				images[i] = data
			}
		}
		ptx, err := qualificationRead(filepath.Join(dir, v.Name+".ptx"))
		if err != nil {
			t.Fatal(err)
		}
		sass, err := qualificationRead(filepath.Join(dir, v.Name+".sass"))
		if err != nil {
			t.Fatal(err)
		}
		pc, sc, observation, err := qualificationInspect(v.Name, ptx, sass)
		if err != nil {
			t.Fatal(err)
		}
		if pc != v.PTXConversions || sc != v.SASSConversions || !reflect.DeepEqual(observation, v.Observation) {
			t.Fatal("inspection count mismatch")
		}
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := driver.Init(0); err != nil {
		t.Fatal(err)
	}
	dev, err := driver.DeviceGet(0)
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if major != 12 || minor != 1 {
		t.Fatal("packet requires sm121")
	}
	ctx, err := driver.NewContext(dev, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ctx.Close(); err != nil {
			t.Error(err)
		}
	}()
	facts := cudatest.QualificationFacts{ExplicitSharedCast: true, MatchedKernelInspection: true, ConversionCountA: uint64(build.Images[0].SASSConversions), ConversionCountB: uint64(build.Images[1].SASSConversions)}
	type result struct {
		Name                               string
		LoadedDigest                       string
		Words                              []uint32
		Corrupted                          []uint32
		Registers, SharedBytes, LocalBytes int32
	}
	var results []result
	for i, name := range []string{"before", "after"} {
		module, err := driver.ModuleLoadData(images[i])
		if err != nil {
			t.Fatal(err)
		}
		func() {
			var p driver.DevicePtr
			complete := true
			defer func() {
				if err := qualificationCleanup(complete, driver.CtxSynchronize, func() error {
					if p != 0 {
						return p.Free()
					}
					return nil
				}, module.Close); err != nil {
					t.Error(err)
				}
			}()
			fn, err := module.Function("kernel")
			if err != nil {
				t.Fatal(err)
			}
			regs, err := fn.Attribute(driver.FuncAttributeNumRegs)
			if err != nil || regs <= 0 {
				t.Fatalf("register attributes: %d %v", regs, err)
			}
			shared, err := fn.Attribute(driver.FuncAttributeSharedSizeBytes)
			if err != nil {
				t.Fatal(err)
			}
			local, err := fn.Attribute(driver.FuncAttributeLocalSizeBytes)
			if err != nil {
				t.Fatal(err)
			}
			p, err = driver.MemAllocBytes(12)
			if err != nil {
				t.Fatal(err)
			}
			words := []uint32{qualificationGuard, 0xdeadbeef, qualificationGuard}
			if err := p.CopyFromHostPtr(unsafe.Pointer(&words[0]), 12); err != nil {
				t.Fatal(err)
			}
			out := driver.DevicePtr(uint64(p) + 4)
			input := uint32(35)
			complete = false
			if err := fn.LaunchKernel(driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: 1, Y: 1, Z: 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&out), unsafe.Pointer(&input)}); err != nil {
				t.Fatal(err)
			}
			facts.Launches++
			if err := driver.CtxSynchronize(); err != nil {
				t.Fatal(err)
			}
			complete = true
			facts.CompletedLaunches++
			facts.Synchronizations++
			if err := p.CopyToHostPtr(unsafe.Pointer(&words[0]), 12); err != nil {
				t.Fatal(err)
			}
			if err := qualificationCheckOutput(words); err != nil {
				t.Fatal(err)
			}
			facts.PositiveRuns++
			facts.Expected++
			facts.Checked++
			facts.GuardsExpected += 2
			facts.GuardsChecked += 2
			// Corrupt the actual device output, then read through the same checker.
			bad := uint32(43)
			if err := out.CopyFromHostPtr(unsafe.Pointer(&bad), 4); err != nil {
				t.Fatal(err)
			}
			corrupt := make([]uint32, 3)
			if err := p.CopyToHostPtr(unsafe.Pointer(&corrupt[0]), 12); err != nil {
				t.Fatal(err)
			}
			if err := qualificationCheckOutput(corrupt); err == nil {
				t.Fatal("device corruption was accepted")
			}
			if corrupt[0] != qualificationGuard || corrupt[2] != qualificationGuard || corrupt[1] != 43 {
				t.Fatal("corruption/readback did not complete")
			}
			facts.NegativeRuns++
			if i == 0 {
				facts.RegistersA = uint64(regs)
			} else {
				facts.RegistersB = uint64(regs)
			}
			results = append(results, result{name, qualificationDigest(images[i]), words, corrupt, regs, shared, local})
		}()
	}
	facts.ABOutputsChecked = true
	qualificationJSON(t, outDir, "outputs.json", results)
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binary, err := qualificationRead(executable)
	if err != nil {
		t.Fatal(err)
	}
	qualificationWrite(t, outDir, "executed.test", binary)
	name, err := dev.Name()
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := dev.Uuid()
	if err != nil {
		t.Fatal(err)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	qualificationJSON(t, outDir, "producer.json", struct {
		Scope, GoVersion, Platform, DriverLibrary, DeviceName, DeviceUUID, ComputeCapability, ExecutableSHA256 string
		DriverVersion                                                                                          int
		Facts                                                                                                  cudatest.QualificationFacts
	}{"opt0 shared conversion mechanism; opt3 unqualified", runtime.Version(), runtime.GOOS + "/" + runtime.GOARCH, driver.Path(), name, fmt.Sprintf("%x", uuid.Bytes), "12.1", qualificationDigest(binary), version, facts})
}
