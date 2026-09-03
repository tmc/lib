package nvtx

import (
	"testing"

	"github.com/ebitengine/purego"
)

var (
	mockMarkCount       int
	mockRangePushCount  int
	mockRangePopCount   int
	mockDomainPushCount int
	mockDomainMarkCount int
)

func mockMarkA(msg *byte) {
	mockMarkCount++
}

func mockRangePushA(msg *byte) int32 {
	mockRangePushCount++
	return 1
}

func mockRangePop() int32 {
	mockRangePopCount++
	return 0
}

func mockDomainRangePushEx(domain uintptr, attrs *eventAttributes) int32 {
	mockDomainPushCount++
	return 1
}

func mockDomainRangePop(domain uintptr) int32 {
	return 0
}

func mockDomainMarkEx(domain uintptr, attrs *eventAttributes) {
	mockDomainMarkCount++
}

func mockDomainCreateA(name *byte) uintptr {
	return 0x12345678
}

func mockDomainDestroy(domain uintptr) {}

func TestNoOpPath(t *testing.T) {
	// With enabled == false, all calls must be safe no-ops
	enabled = false
	Mark("TestMarker")
	depth := RangePush("TestRange")
	if depth != 0 {
		t.Errorf("expected 0 depth in no-op mode, got %d", depth)
	}

	remaining := RangePop()
	if remaining != 0 {
		t.Errorf("expected 0 remaining depth in no-op mode, got %d", remaining)
	}

	rID := RangeStart("AsyncRange")
	RangeEnd(rID)

	done := ScopedRange("Scoped")
	done()

	NameOSThread(1234, "WorkerGoroutine")

	dom := CreateDomain("InferenceDomain")
	defer dom.Destroy()

	dom.Mark("Layer0Start")
	dom.RangePush("AttentionBlock")
	dom.RangePop()

	rID2 := dom.RangeStart("AsyncKernel")
	dom.RangeEnd(rID2)

	handle := dom.RegisterString("TokenGeneration")
	if handle != 0 {
		t.Errorf("expected 0 handle in no-op mode, got %v", handle)
	}

	NameCategory(dom, 1, "Compute")
}

func TestExportTableHandshake(t *testing.T) {
	initTables()

	// 1. Verify VERSIONINFO export table
	ptrVer := getExportTable(etidVersionInfo)
	if ptrVer == 0 {
		t.Fatal("expected non-null VERSIONINFO export table")
	}
	verInfo := &etblVersionInfo
	if verInfo.version != 3 {
		t.Errorf("expected NVTX version 3, got %d", verInfo.version)
	}

	// 2. Verify CALLBACKS export table
	ptrCb := getExportTable(etidCallbacks)
	if ptrCb == 0 {
		t.Fatal("expected non-null CALLBACKS export table")
	}
	cbInfo := &etblCallbacks
	if cbInfo.getModuleFunctionTable == 0 {
		t.Fatal("expected non-null GetModuleFunctionTable function pointer")
	}

	// 3. Query module CORE table
	var tableCore uintptr
	var sizeCore uint32
	getModuleFunctionTable(cbModuleCore, &tableCore, &sizeCore)
	if tableCore == 0 || sizeCore == 0 {
		t.Fatalf("expected non-null CORE function table, got ptr=%v size=%d", tableCore, sizeCore)
	}

	// 4. Query module CORE2 table
	var tableCore2 uintptr
	var sizeCore2 uint32
	getModuleFunctionTable(cbModuleCore2, &tableCore2, &sizeCore2)
	if tableCore2 == 0 || sizeCore2 == 0 {
		t.Fatalf("expected non-null CORE2 function table, got ptr=%v size=%d", tableCore2, sizeCore2)
	}

	// 5. Simulate Injection Provider Writing Slots
	fnMarkA = purego.NewCallback(mockMarkA)
	fnRangePushA = purego.NewCallback(mockRangePushA)
	fnRangePop = purego.NewCallback(mockRangePop)

	fnDomainMarkEx = purego.NewCallback(mockDomainMarkEx)
	fnDomainRangePushEx = purego.NewCallback(mockDomainRangePushEx)
	fnDomainRangePop = purego.NewCallback(mockDomainRangePop)
	fnDomainCreateA = purego.NewCallback(mockDomainCreateA)
	fnDomainDestroy = purego.NewCallback(mockDomainDestroy)

	enabled = true

	// 6. Test that dispatches reach our mock functions
	Mark("TestMarker")
	if mockMarkCount != 1 {
		t.Errorf("expected mockMarkCount 1, got %d", mockMarkCount)
	}

	depth := RangePush("Forward")
	if depth != 1 || mockRangePushCount != 1 {
		t.Errorf("expected depth 1, mockRangePushCount 1; got depth=%d count=%d", depth, mockRangePushCount)
	}

	RangePop()
	if mockRangePopCount != 1 {
		t.Errorf("expected mockRangePopCount 1, got %d", mockRangePopCount)
	}

	dom := CreateDomain("InferenceDomain")
	if dom.Raw() != 0x12345678 {
		t.Errorf("expected domain raw handle 0x12345678, got %x", dom.Raw())
	}

	dom.Mark("Checkpoint")
	if mockDomainMarkCount != 1 {
		t.Errorf("expected mockDomainMarkCount 1, got %d", mockDomainMarkCount)
	}

	dom.RangePush("Layer0")
	if mockDomainPushCount != 1 {
		t.Errorf("expected mockDomainPushCount 1, got %d", mockDomainPushCount)
	}
	dom.RangePop()
	dom.Destroy()
}
