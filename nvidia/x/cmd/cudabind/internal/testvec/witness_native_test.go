//go:build native

package testvec

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func witnessDevice(t *testing.T) *cuda.Device {
	t.Helper()
	if os.Getenv("CUDABIND_NATIVE") != "1" {
		t.Skip("set CUDABIND_NATIVE=1 under gpu-lock for native witnesses")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	dev := cudatest.Require(t)
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	return dev
}

func deviceLayout(t *testing.T, dev *cuda.Device, defines ...string) []int64 {
	t.Helper()
	want, err := vecModuleLayoutExpected()
	if err != nil {
		t.Fatal(err)
	}
	cm, err := compileVecModule(dev, vecModuleLayoutSource(), nvrtcx.WithDefines(defines...))
	if err != nil {
		t.Fatal(err)
	}
	m, err := newVecModule(dev, cm)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := m.Close(); err != nil {
			t.Error(err)
		}
	}()
	k, err := m.mod.Function("cudabind_layout")
	if err != nil {
		t.Fatal(err)
	}
	out, err := cudamem.Alloc[int64](dev, len(want))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := out.Free(); err != nil {
			t.Error(err)
		}
	}()
	p := out.Ptr()
	stream := dev.DefaultStream()
	err = stream.WithHandle(func(h driver.Stream) error {
		return k.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, h, []unsafe.Pointer{unsafe.Pointer(&p)})
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(t.Context()); err != nil {
		t.Fatal(err)
	}
	got := make([]int64, len(want))
	if err := out.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	row := struct {
		Kind, SourceSHA256 string
		Defines            []string
		Observed           []int64
	}{"abi-witness", fmt.Sprintf("%x", sha256.Sum256([]byte(vecModuleLayoutSource()))), defines, got}
	logWitnessReadback(t, row)
	return got
}

func logWitnessReadback(t *testing.T, row any) {
	t.Helper()
	b, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("native-readback %s", b)
}

func TestWitness_StaleBindingNamesField(t *testing.T) {
	dev := witnessDevice(t)
	if err := vecModuleCheckLayout(deviceLayout(t, dev)); err != nil {
		t.Fatal(err)
	}
	err := vecModuleCheckLayout(deviceLayout(t, dev, "VEC_PACKED=1"))
	if err == nil || !strings.Contains(err.Error(), "offsetof(Packing, value)") {
		t.Fatalf("stale binding error = %v", err)
	}
}

func TestWitness_UsesKernelDefines(t *testing.T) {
	dev := witnessDevice(t)
	if err := vecModuleCheckLayout(deviceLayout(t, dev)); err != nil {
		t.Fatal(err)
	}
	err := vecModuleCheckLayout(deviceLayout(t, dev, "VEC_WIDE_A=1"))
	if err == nil || !strings.Contains(err.Error(), "sizeof(Pair.a)") {
		t.Fatalf("kernel define not seen by witness: %v", err)
	}
}

func TestExistingLayoutMutationAudit(t *testing.T) {
	dev := witnessDevice(t)
	want, err := vecModuleLayoutExpected()
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name, define string
		legacyCaught bool
	}{
		{"baseline", "", false},
		{"same_layout_wider_field", "VEC_WIDE_A=1", false},
		{"packing", "VEC_PACKED=1", true},
		{"nested_vector_alignment", "VEC_ALIGNED_VECTOR=1", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var defines []string
			if tt.define != "" {
				defines = []string{tt.define}
			}
			got := deviceLayout(t, dev, defines...)
			// These are exactly the old generator's measurement kinds: aggregate
			// sizeof, field offsetof and constant values. New field widths and
			// alignment rows are excluded, never replaced with expected values.
			legacyCaught := false
			for i, w := range want {
				if strings.Contains(w.c, " argument ") || strings.HasPrefix(w.c, "alignof(") || strings.HasPrefix(w.c, "sizeof(") && strings.Contains(w.c, ".") {
					continue
				}
				if got[i] != w.v {
					legacyCaught = true
				}
			}
			if legacyCaught != tt.legacyCaught {
				t.Fatalf("legacy observation = %v, want %v", legacyCaught, tt.legacyCaught)
			}
			fullErr := vecModuleCheckLayout(got)
			if (fullErr != nil) != (tt.define != "") {
				t.Fatalf("extended witness error = %v", fullErr)
			}
			record := struct {
				Name, Define                 string
				LegacyCaught, ExtendedCaught bool
			}{tt.name, tt.define, legacyCaught, fullErr != nil}
			b, _ := json.Marshal(record)
			t.Log(string(b))
		})
	}
}

func TestLaunch_NativeEarlyRefusals(t *testing.T) {
	dev := witnessDevice(t)
	m, err := loadVecModule(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := m.Close(); err != nil {
			t.Error(err)
		}
	}()
	out, err := cudamem.Alloc[float32](dev, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := out.Free(); err != nil {
			t.Error(err)
		}
	}()
	stream := dev.DefaultStream()
	grid, block := [3]uint32{4, 1, 1}, [3]uint32{256, 1, 1}
	before := make([]float32, 1024)
	for i := range before {
		before[i] = -42
	}
	if err := out.CopyFromHost(before); err != nil {
		t.Fatal(err)
	}
	err = m.boundedCopy(stream, grid, block, out.Slice(0, 512), 1024)
	for _, s := range []string{"out", "512", "1024"} {
		if err == nil || !strings.Contains(err.Error(), s) {
			t.Fatalf("short output error = %v", err)
		}
	}
	if err := stream.Synchronize(t.Context()); err != nil {
		t.Fatal(err)
	}
	unchanged := make([]float32, 1024)
	if err := out.CopyToHost(unchanged); err != nil {
		t.Fatal(err)
	}
	logWitnessReadback(t, struct {
		Kind   string
		Output []float32
	}{"refused-short-output", unchanged})
	for i, x := range unchanged {
		if x != -42 {
			t.Fatalf("refused launch wrote output[%d] = %g", i, x)
		}
	}
	if err := m.boundedCopy(stream, grid, block, out, 1024); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(t.Context()); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 1024)
	if err := out.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	logWitnessReadback(t, struct {
		Kind   string
		Output []float32
	}{"healthy-bounded-output", got})
	for i, x := range got {
		if x != float32(i) {
			t.Fatalf("positive launch output[%d] = %g", i, x)
		}
	}
	storage, err := cudamem.Alloc[byte](dev, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := storage.Free(); err != nil {
			t.Error(err)
		}
	}()
	misaligned, err := cudamem.Reinterpret[pair](storage.Slice(4, 36))
	if err != nil {
		t.Fatal(err)
	}
	err = m.fill(stream, [3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, pair{}, misaligned)
	if err == nil || !strings.Contains(err.Error(), "out") || !strings.Contains(err.Error(), "16") {
		t.Fatalf("C pointee alignment refusal = %v", err)
	}
	limit, err := dev.Attribute(driver.DevAttrMaxSharedMemoryPerBlockOptin)
	if err != nil {
		t.Fatal(err)
	}
	ints, err := cudamem.Reinterpret[int32](out)
	if err != nil {
		t.Fatal(err)
	}
	err = m.sum(stream, [3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, uint32(limit)+1, ints.Slice(0, 1), 1, ints.Slice(0, 1))
	if err == nil || !strings.Contains(err.Error(), "sharedBytes") {
		t.Fatalf("actual opt-in refusal = %v", err)
	}
}
