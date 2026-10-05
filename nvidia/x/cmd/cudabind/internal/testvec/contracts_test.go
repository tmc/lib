package testvec

import (
	"errors"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestLaunch_ShortOutputRejectedWithCounts(t *testing.T) {
	err := vecModuleRequired("out", 512, 1024, false)
	for _, text := range []string{"out", "512", "1024"} {
		if err == nil || !strings.Contains(err.Error(), text) {
			t.Fatalf("extent error = %v, missing %s", err, text)
		}
	}
	// The actual generated method must reject the declared extent before any
	// device, module or stream query. A skipped check yields a different error.
	var m vecModule
	err = m.boundedCopy(nil, [3]uint32{4, 1, 1}, [3]uint32{256, 1, 1}, cudamem.Slice[float32]{}, 1024)
	if err == nil || !strings.Contains(err.Error(), "out") || !strings.Contains(err.Error(), "1024") || strings.Contains(err.Error(), "closed") {
		t.Fatalf("method did not check extent first: %v", err)
	}
}

func TestLaunch_MisalignedSliceRejected(t *testing.T) {
	if err := vecModuleExtent("out", 512, driver.DevicePtr(0x1001), 4); err == nil || !strings.Contains(err.Error(), "out") || !strings.Contains(err.Error(), "4") {
		t.Fatalf("alignment error = %v", err)
	}
	if err := vecModuleExtent("out", 512, driver.DevicePtr(0x1000), 4); err != nil {
		t.Fatal(err)
	}
	if err := vecModuleExtent("out", 512, 0, 4); err == nil {
		t.Fatal("null storage accepted")
	}
}

func testDeviceLimit(a driver.DeviceAttribute) (int, error) {
	switch a {
	case driver.DevAttrMaxBlockDimX, driver.DevAttrMaxBlockDimY, driver.DevAttrMaxThreadsPerBlock:
		return 1024, nil
	case driver.DevAttrMaxBlockDimZ:
		return 64, nil
	case driver.DevAttrMaxSharedMemoryPerBlockOptin:
		return 101376, nil
	}
	return 0, errors.New("unexpected attribute")
}
func testKernelLimit(a driver.FunctionAttribute) (int, error) {
	switch a {
	case driver.FuncAttributeMaxThreadsPerBlock:
		return 1024, nil
	case driver.FuncAttributeSharedSizeBytes:
		return 0, nil
	case driver.FuncAttributeMaxDynamicSharedSizeBytes:
		return 101376, nil
	}
	return 0, errors.New("unexpected attribute")
}

func TestLaunch_SharedMemoryOverOptInRejected(t *testing.T) {
	err := vecModuleGeometry([3]uint32{1, 1, 1}, [3]uint32{32, 1, 1}, 101377, testDeviceLimit, testKernelLimit)
	if err == nil || !strings.Contains(err.Error(), "101377") || !strings.Contains(err.Error(), "101376") {
		t.Fatalf("shared error = %v", err)
	}
	if err := vecModuleGeometry([3]uint32{1, 1, 1}, [3]uint32{32, 1, 1}, 101376, testDeviceLimit, testKernelLimit); err != nil {
		t.Fatal(err)
	}
	static := func(a driver.FunctionAttribute) (int, error) {
		if a == driver.FuncAttributeSharedSizeBytes {
			return 1024, nil
		}
		return testKernelLimit(a)
	}
	if err := vecModuleGeometry([3]uint32{1, 1, 1}, [3]uint32{32, 1, 1}, 100353, testDeviceLimit, static); err == nil {
		t.Fatal("static shared allocation ignored")
	}
}

func TestLaunch_BlockAndArithmeticRefusals(t *testing.T) {
	for _, block := range [][3]uint32{{0, 1, 1}, {1025, 1, 1}, {32, 1, 65}, {64, 32, 1}} {
		if err := vecModuleGeometry([3]uint32{1, 1, 1}, block, 0, testDeviceLimit, testKernelLimit); err == nil {
			t.Fatalf("block %v accepted", block)
		}
	}
	if _, err := vecModuleArithmetic("count", math.MaxUint64, 2, true); err == nil {
		t.Fatal("product overflow accepted")
	}
	if _, err := vecModuleArithmetic("count", math.MaxUint64, 1, false); err == nil {
		t.Fatal("sum overflow accepted")
	}
	if _, err := vecModuleCount("n", -1); err == nil {
		t.Fatal("negative count accepted")
	}
	deviceLow := func(a driver.DeviceAttribute) (int, error) {
		if a == driver.DevAttrMaxThreadsPerBlock {
			return 128, nil
		}
		return testDeviceLimit(a)
	}
	kernelLow := func(a driver.FunctionAttribute) (int, error) {
		if a == driver.FuncAttributeMaxThreadsPerBlock {
			return 128, nil
		}
		return testKernelLimit(a)
	}
	for _, limits := range []struct {
		device func(driver.DeviceAttribute) (int, error)
		kernel func(driver.FunctionAttribute) (int, error)
	}{{deviceLow, testKernelLimit}, {testDeviceLimit, kernelLow}} {
		err := vecModuleGeometry([3]uint32{1, 1, 1}, [3]uint32{256, 1, 1}, 0, limits.device, limits.kernel)
		if err == nil || !strings.Contains(err.Error(), "256") || !strings.Contains(err.Error(), "128") {
			t.Fatalf("independent thread limit ignored: %v", err)
		}
	}
	dynamicLow := func(a driver.FunctionAttribute) (int, error) {
		if a == driver.FuncAttributeMaxDynamicSharedSizeBytes {
			return 49152, nil
		}
		return testKernelLimit(a)
	}
	err := vecModuleGeometry([3]uint32{1, 1, 1}, [3]uint32{32, 1, 1}, 65536, testDeviceLimit, dynamicLow)
	if err == nil || !strings.Contains(err.Error(), "65536") || !strings.Contains(err.Error(), "49152") {
		t.Fatalf("configured shared limit ignored: %v", err)
	}
	var m vecModule
	if m.unchecked.Load() {
		t.Fatal("checks disabled by default")
	}
	if err := m.SetLaunchChecks(false); err != nil {
		t.Fatal(err)
	}
	if !m.unchecked.Load() {
		t.Fatal("explicit opt-out ignored")
	}
	if err := m.SetLaunchChecks(true); err != nil {
		t.Fatal(err)
	}
	if m.unchecked.Load() {
		t.Fatal("checks not reenabled")
	}
	var missing *vecModule
	if err := missing.SetLaunchChecks(false); err == nil {
		t.Fatal("nil module accepted")
	}
}

func TestLaunch_DeclaredGeometryAndNegativeCount(t *testing.T) {
	var m vecModule
	for _, tt := range []struct {
		block uint32
		n     int32
		want  string
	}{{64, 64, "block.x == 128"}, {128, 129, "grid.x * block.x >= n"}, {128, -1, "n has count -1"}} {
		err := m.contractCounts(nil, [3]uint32{1, 1, 1}, [3]uint32{tt.block, 1, 1}, tt.n)
		if err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Fatalf("declared refusal = %v, want %s", err, tt.want)
		}
	}
}

func TestWitness_ReadbackRequired(t *testing.T) {
	want, err := vecModuleLayoutExpected()
	if err != nil {
		t.Fatal(err)
	}
	got := make([]int64, len(want))
	pairAlignment := -1
	for i, w := range want {
		got[i] = w.v
		if w.c == "alignof(Pair)" {
			pairAlignment = i
			got[i] = 16
		}
	}
	if pairAlignment < 0 {
		t.Fatal("Pair ABI alignment witness missing")
	}
	if err := vecModuleCheckLayout(got); err != nil {
		t.Fatal(err)
	}
	got[pairAlignment] = 4
	if err := vecModuleCheckLayout(got); err == nil || !strings.Contains(err.Error(), "alignof(Pair)") {
		t.Fatalf("device readback ignored: %v", err)
	}
	if err := vecModuleCheckLayout(got[:len(got)-1]); err == nil {
		t.Fatal("partial device buffer accepted")
	}
}

func TestWitness_ChangedEmbeddedSourceRefused(t *testing.T) {
	if os.Getenv("CUDABIND_EXPECT_STALE") != "1" {
		t.Skip("task-owned embed overlay control only")
	}
	_, err := compileVecModule(nil, "")
	if err == nil || !strings.Contains(err.Error(), "changed since") {
		t.Fatalf("stale source error = %v", err)
	}
}

func Example() {
	var m vecModule
	_ = m.SetLaunchChecks(false)
	_ = m.SetLaunchChecks(true)
	// Output:
}
