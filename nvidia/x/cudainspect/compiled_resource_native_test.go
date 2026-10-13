//go:build native

package cudainspect

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// This project receipt describes retained offline outputs, not vendor JSON.
// Root runs exact nvcc/nvdisasm commands and retains their raw outputs beside it.
type compiledMLPReceipt struct {
	Target              string
	SourceDigest        string
	ImageDigest         string
	CompilerVersion     string
	DisassemblerVersion string
	CompilerOptions     []string
	Cap                 uint64
}

func compiledBoundedRead(t *testing.T, path string, limit int64) []byte {
	t.Helper()
	st, err := os.Lstat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !st.Mode().IsRegular() || st.Size() > limit {
		t.Fatalf("nonregular or excessive packet file %s", path)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil || int64(len(b)) > limit {
		t.Fatalf("read bounded packet: %v", err)
	}
	return b
}

func runCompiledMLP(t *testing.T, cap uint64) (*CompiledResources, LoadedResources, []float32) {
	t.Helper()
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Skip("SKIPPED, NOT VERIFIED: compiled resource native tests require gpu-lock")
	}
	dir := os.Getenv("COMPILED_RESOURCE_PACKET_DIR")
	if dir == "" {
		t.Fatal("COMPILED_RESOURCE_PACKET_DIR required for offline CUBIN qualification")
	}
	if os.Getenv("COMPILED_RESOURCE_OUTPUT_DIR") == "" {
		t.Fatal("COMPILED_RESOURCE_OUTPUT_DIR required to retain numerical/resource receipts")
	}
	source := compiledMLPSource(t)
	sourceDigest := fmt.Sprintf("%x", sha256.Sum256([]byte(source)))
	if !bytes.Equal(compiledBoundedRead(t, filepath.Join(dir, "mlp.cu"), 16<<20), []byte(source)) {
		t.Fatal("offline source differs from current emitted MLP")
	}
	variant := filepath.Join(dir, "cap-"+strconv.FormatUint(cap, 10))
	var receipt compiledMLPReceipt
	d := json.NewDecoder(bytes.NewReader(compiledBoundedRead(t, filepath.Join(variant, "receipt.json"), 1<<20)))
	d.DisallowUnknownFields()
	if err := d.Decode(&receipt); err != nil {
		t.Fatal(err)
	}
	if err := d.Decode(new(any)); err != io.EOF {
		t.Fatal("trailing receipt data")
	}
	if receipt.SourceDigest != sourceDigest || receipt.Cap != cap {
		t.Fatal("offline source or cap mismatch")
	}
	image := compiledBoundedRead(t, filepath.Join(variant, "mlp.cubin"), 64<<20)
	if fmt.Sprintf("%x", sha256.Sum256(image)) != receipt.ImageDigest {
		t.Fatal("offline image digest mismatch")
	}
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "n", Type: "s32"}, cudakernel.Param{Name: "in", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "params", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, image, compiledMLPEntry, schema, receipt.Target, cudakernel.WithToolchain(receipt.CompilerVersion), cudakernel.WithCompilerOptions(receipt.CompilerOptions))
	if err != nil {
		t.Fatal(err)
	}
	i, _ := NewInspector(t.TempDir())
	var ceiling *uint64
	if cap > 0 {
		ceiling = &cap
	}
	r, err := i.InspectCompiledResources(a, CompiledResourceInput{Source: []byte(source), CompilerVersion: receipt.CompilerVersion, CompilerOptions: receipt.CompilerOptions, Diagnostics: string(compiledBoundedRead(t, filepath.Join(variant, "compiler.stderr"), 4<<20)), DiagnosticImage: a.Digest, SASS: string(compiledBoundedRead(t, filepath.Join(variant, "sass.txt"), 12<<20)), SASSImage: a.Digest, DisassemblerVersion: receipt.DisassemblerVersion, RegisterCap: ceiling})
	if err != nil {
		t.Fatal(err)
	}
	dev := cudatest.RequireDevice(t)
	quarantined := false
	defer func() {
		if !quarantined {
			if err := dev.Close(); err != nil {
				t.Error(err)
			}
		}
	}()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Target != fmt.Sprintf("sm_%d%d", major, minor) {
		t.Fatal("packet target differs from actual device")
	}
	var module driver.Module
	var fn driver.Function
	l := LoadedResources{ImageDigest: a.Digest, Entry: a.EntryPoint, Target: a.Target}
	var ptrs []driver.DevicePtr
	inFlight := false
	defer func() {
		if inFlight {
			if err := dev.WithContext(driver.CtxSynchronize); err != nil {
				quarantined = true
				t.Errorf("compiled fixture remains quarantined, no frees/unload: %v", err)
				return
			}
		}
		if err := dev.WithContext(func() error {
			for _, p := range ptrs {
				if err := p.Free(); err != nil {
					return err
				}
			}
			if module != 0 {
				return module.Close()
			}
			return nil
		}); err != nil {
			quarantined = true
			t.Errorf("compiled fixture cleanup failed: %v", err)
		}
	}()
	module, err = a.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	if err := dev.WithContext(func() error {
		var e error
		fn, e = module.Function(a.EntryPoint)
		if e != nil {
			return e
		}
		for _, attr := range []struct {
			key driver.FunctionAttribute
			dst *ResourceFact
		}{{driver.FuncAttributeNumRegs, &l.Registers}, {driver.FuncAttributeSharedSizeBytes, &l.StaticShared}, {driver.FuncAttributeLocalSizeBytes, &l.LocalBytes}} {
			v, e := fn.Attribute(attr.key)
			if e != nil {
				*attr.dst = ResourceFact{Category: "runtime attribute", Reason: e.Error()}
			} else if v < 0 {
				return fmt.Errorf("negative function attribute")
			} else {
				*attr.dst = resourceFact(uint64(v), "runtime attribute")
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if r.Allocated.Value == nil || l.Registers.Value == nil || *r.Allocated.Value != *l.Registers.Value {
		t.Fatal("decoded and loaded allocated registers unavailable or unequal")
	}
	if r.StaticShared.Value != nil && (l.StaticShared.Value == nil || *r.StaticShared.Value != *l.StaticShared.Value) {
		t.Fatal("decoded and loaded shared mismatch")
	}
	const rows = 65
	x := make([]float32, rows*64)
	w := make([]float32, 3*64*64)
	for n := range x {
		x[n] = float32((n*11)%37-18) / 9
	}
	for layer := 0; layer < 3; layer++ {
		for in := 0; in < 64; in++ {
			for out := 0; out < 64; out++ {
				w[layer*4096+in*64+out] = float32((in*7+out*3+layer*5)%13-6) / 128
			}
		}
	}
	want := append([]float32(nil), x...)
	for layer := 0; layer < 3; layer++ {
		next := make([]float32, rows*64)
		for row := 0; row < rows; row++ {
			for out := 0; out < 64; out++ {
				var sum float32
				for in := 0; in < 64; in++ {
					sum += want[row*64+in] * w[layer*4096+in*64+out]
				}
				if layer < 2 && sum < 0 {
					sum = 0
				}
				next[row*64+out] = sum
			}
		}
		want = next
	}
	got := make([]float32, len(want))
	for n := range got {
		got[n] = math.Float32frombits(0x7fc00123)
	}
	var count int32 = rows
	if err := dev.WithContext(func() error {
		for _, v := range [][]float32{x, w, got} {
			p, e := driver.MemAllocBytes(uint(len(v) * 4))
			if e != nil {
				return e
			}
			ptrs = append(ptrs, p)
			if e := p.CopyFromHost(v); e != nil {
				return e
			}
		}
		args := []unsafe.Pointer{unsafe.Pointer(&count), unsafe.Pointer(&ptrs[0]), unsafe.Pointer(&ptrs[1]), unsafe.Pointer(&ptrs[2])}
		inFlight = true
		if e := fn.LaunchKernel(driver.Dim3{X: 3, Y: 1, Z: 1}, driver.Dim3{X: 32, Y: 1, Z: 1}, 0, 0, args); e != nil {
			return e
		}
		if e := driver.CtxSynchronize(); e != nil {
			return e
		}
		inFlight = false
		return ptrs[2].CopyToHost(got)
	}); err != nil {
		t.Fatal(err)
	}
	h := sha256.New()
	for _, v := range append(append([]float32(nil), x...), w...) {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		h.Write(b[:])
	}
	p := cudatest.CUDAOutputPolicy{Target: a.Target, DType: "fp32", MathMode: "offline nvcc FP32, compiler options retained", SourceDigest: sourceDigest, InputDigest: fmt.Sprintf("%x", h.Sum(nil)), Elements: len(want), Absolute: 1e-4, Relative: 1e-4, NaN: "reject", Infinity: "reject", Zero: "equal"}
	if err := cudatest.CompareCUDAOutputFP32(got, want, p); err != nil {
		t.Fatal(err)
	}
	if out := os.Getenv("COMPILED_RESOURCE_OUTPUT_DIR"); out != "" {
		if err := os.MkdirAll(out, 0700); err != nil {
			t.Fatal(err)
		}
		b, err := json.Marshal(struct {
			Report            *CompiledResources
			Loaded            LoadedResources
			Policy            cudatest.CUDAOutputPolicy
			Actual, Reference []float32
		}{r, l, p, got, want})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(out, strings.ReplaceAll(t.Name(), "/", "-")+"-cap-"+strconv.FormatUint(cap, 10)+".json"), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	summary, _ := json.Marshal(struct {
		Allocated, PeakLive, Cap, StaticShared, LocalMemory, Stack, SpillLoad, SpillStore ResourceFact
		LocalLoads, LocalStores                                                           int
		Loaded                                                                            LoadedResources
	}{r.Allocated, r.PeakLive, r.RegisterCap, r.StaticShared, r.LocalMemory, r.StackBytes, r.SpillLoadBytes, r.SpillStoreBytes, len(r.LocalLoads), len(r.LocalStores), l})
	t.Logf("source=%s image=%s input=%s values=%d resources=%s", sourceDigest, a.Digest, p.InputDigest, len(got), summary)
	return r, l, got
}

func TestCompiledResources_MLPMatchesLoadedImage(t *testing.T) {
	r, l, _ := runCompiledMLP(t, 0)
	if err := r.CheckRegisterResidency(claimFor(r), l); err != nil {
		t.Logf("numerically correct baseline; residency claim refused: %v", err)
	}
}

func TestCompiledResources_ActualSpillControlDetected(t *testing.T) {
	baseline, _, _ := runCompiledMLP(t, 0)
	if baseline.SpillLoadBytes.Value == nil || baseline.SpillStoreBytes.Value == nil || *baseline.SpillLoadBytes.Value != 0 || *baseline.SpillStoreBytes.Value != 0 || baseline.CorroboratedSpills() {
		t.Fatal("unlimited baseline already spills: healthy/spilling control is inconclusive")
	}
	for _, cap := range compiledMLPCaps[1:] {
		r, l, _ := runCompiledMLP(t, cap)
		// Both variants use the same generated input and weights and independently
		// compare every output against the same CPU reference through lane 69.
		if r.SourceDigest != baseline.SourceDigest || r.Entry != baseline.Entry || r.Target != baseline.Target {
			t.Fatal("paired compilation identity mismatch")
		}
		if r.CorroboratedSpills() {
			if err := r.CheckRegisterResidency(claimFor(r), l); err == nil {
				t.Fatal("consumer accepted corroborated spills")
			}
			t.Logf("actual spill control corroborated at predeclared cap %d", cap)
			return
		}
	}
	t.Fatal("INCONCLUSIVE: bounded caps produced no corroborated actual spills")
}
