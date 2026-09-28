//go:build linux && cupti_physical

package cuptisass_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"debug/elf"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cuptisass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// TestPhysicalWorkload requires an exclusive GPU. Unlike discovery, an enabled
// qualification that cannot collect work fails instead of reporting a pass.
func TestPhysicalWorkload(t *testing.T) {
	d, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := d.Close(); err != nil {
			t.Error(err)
		}
	}()
	const source = `extern "C" __global__ void sass_work(unsigned *out) {
 unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
 out[i] = i * 3 + 7;
 }`
	cm, err := nvrtcx.Compile(source, nvrtcx.WithDeviceArch(d), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	cubin := cm.CUBIN()
	if len(cubin) == 0 {
		t.Fatal("missing native artifact")
	}
	if parent := os.Getenv("CUPTI_QUALIFICATION_DIR"); parent != "" {
		dir, err := os.MkdirTemp(parent, "sass-workload-")
		if err != nil {
			t.Fatal(err)
		}
		flags, err := nvrtcx.CompileFlags(nvrtcx.WithDeviceArch(d), nvrtcx.WithNoCache())
		if err != nil {
			t.Fatal(err)
		}
		for name, data := range map[string][]byte{"kernel.cu": []byte(source), "kernel.cubin": cubin, "kernel.ptx": []byte(cm.PTX()), "flags.txt": []byte(strings.Join(flags, "\n") + "\n")} {
			if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("retained source, PTX, loaded CUBIN and compiler flags: %s", dir)
	}

	mod, err := cm.LoadModule(d)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := mod.Close(); err != nil {
			t.Error(err)
		}
	}()
	kernel, err := mod.Function("sass_work")
	if err != nil {
		t.Fatal(err)
	}
	out, err := cudamem.Alloc[uint32](d, 256)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := out.Free(); err != nil {
			t.Error(err)
		}
	}()
	var ctx driver.Context
	if err := d.WithContext(func() error { var err error; ctx, err = driver.CtxGetCurrent(); return err }); err != nil {
		t.Fatal(err)
	}
	chip, err := cuptisass.DeviceChipName(0)
	if err != nil {
		t.Fatal(err)
	}
	metric, err := cuptisass.MetricProperties(chip, "smsp__sass_inst_executed")
	if err != nil {
		t.Fatal(err)
	}
	sess, err := cuptisass.NewSession(uintptr(ctx), cuptisass.Config{LazyPatching: true, Metrics: []cuptisass.MetricConfig{{MetricID: metric.ID, Granularity: cuptisass.GranularityGPU}}})
	if sess != nil {
		defer func() {
			if err := sess.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := sess.Enable(); err != nil {
		t.Fatal(err)
	}
	if err := cudalaunch.OwnedKernel(kernel).Block(256, 1, 1).Launch(out); err != nil {
		t.Fatal(err)
	}
	host := make([]uint32, 256)
	if err := out.CopyToHost(host); err != nil {
		t.Fatal(err)
	}
	for i, v := range host {
		if v != uint32(i*3+7) {
			t.Fatalf("output[%d]=%d", i, v)
		}
	}
	records, err := sess.Flush()
	if err != nil {
		t.Fatal(err)
	}
	if len(records) == 0 {
		t.Fatal("no records after verified launch")
	}
	dis, err := (cudasass.Disassembler{TempDir: t.TempDir()}).Disassemble(context.Background(), cubin, cudasass.Options{})
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := cudasass.ParseString(string(dis.Stdout))
	if err != nil {
		t.Fatal(err)
	}
	if err := cuptisass.VerifyCensus(records, parsed, "sass_work"); err != nil {
		t.Fatal(err)
	}
	f, err := elf.NewFile(bytes.NewReader(cubin))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	symbols, err := f.Symbols()
	if err != nil {
		t.Fatal(err)
	}
	var total uint64
	for _, r := range records {
		// ELF's Symbols omits the mandatory null symbol, hence index-1.
		if r.FunctionName != "sass_work" || r.FunctionIndex == 0 || int(r.FunctionIndex) > len(symbols) || symbols[r.FunctionIndex-1].Name != "sass_work" {
			t.Fatalf("unexpected function identity: %+v", r)
		}
		for _, v := range r.Values {
			if v.MetricID != metric.ID {
				t.Fatalf("unexpected metric %d", v.MetricID)
			}
			total += v.Value
		}
	}
	if total == 0 {
		t.Fatal("no executed instructions")
	}
	// Two more identical launches must double every recorded instruction count.
	// This catches empty, stale or unrelated data that mere discovery cannot.
	for i := 0; i < 2; i++ {
		if err := cudalaunch.OwnedKernel(kernel).Block(256, 1, 1).Launch(out); err != nil {
			t.Fatal(err)
		}
	}
	if err := out.CopyToHost(host); err != nil {
		t.Fatal(err)
	}
	twice, err := sess.Flush()
	if err != nil {
		t.Fatal(err)
	}
	if len(twice) != len(records) {
		t.Fatalf("record census changed: %d vs %d", len(twice), len(records))
	}
	first := make(map[uint32]cuptisass.Record, len(records))
	for _, r := range records {
		if _, ok := first[r.PCOffset]; ok {
			t.Fatalf("duplicate PC %x", r.PCOffset)
		}
		first[r.PCOffset] = r
	}
	for _, r := range twice {
		prev, ok := first[r.PCOffset]
		if !ok || r.CubinCRC != prev.CubinCRC || r.FunctionIndex != prev.FunctionIndex || r.FunctionName != prev.FunctionName || len(r.Values) != len(prev.Values) {
			t.Fatalf("instruction identity changed: %+v", r)
		}
		for i, v := range r.Values {
			if v.MetricID != prev.Values[i].MetricID || v.Value != 2*prev.Values[i].Value {
				t.Fatalf("PC %x: count=%d want=%d", r.PCOffset, v.Value, 2*prev.Values[i].Value)
			}
		}
	}
	dropped, err := sess.Disable()
	if err != nil {
		t.Fatal(err)
	}
	if dropped != 0 {
		t.Fatalf("dropped %d instruction records", dropped)
	}

	t.Logf("chip=%s source_sha256=%x cubin_sha256=%x native_crc=%08x function_index=%d records=%d instructions=%d doubled_on_two_launches=true drops=%d; one controlled module, counts aggregated across its launch, not general cross-process attribution", chip, sha256.Sum256([]byte(source)), sha256.Sum256(cubin), records[0].CubinCRC, records[0].FunctionIndex, len(records), total, dropped)
}
