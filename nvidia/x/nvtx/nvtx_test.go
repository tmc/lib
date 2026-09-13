package nvtx

import (
	"runtime"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

var (
	mockMarkCount       int
	mockRangePushCount  int
	mockRangePopCount   int
	mockDomainPushCount int
	mockDomainMarkCount int
	mockCategory        uint32
	mockDomainCategory  uint32
	mockCUDADevice      int32
	mockCUDAContext     uintptr
	mockCUDAStream      uintptr
	mockCUDAEvent       uintptr
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

func mockNameCategoryA(category uint32, name *byte) {
	mockCategory = category
}

func mockDomainNameCategoryA(domain uintptr, category uint32, name *byte) {
	mockDomainCategory = category
}

func mockNameCuDeviceA(device int32, name *byte) {
	mockCUDADevice = device
}

func mockNameCuContextA(context uintptr, name *byte) {
	mockCUDAContext = context
}

func mockNameCuStreamA(stream uintptr, name *byte) {
	mockCUDAStream = stream
}

func mockNameCuEventA(event uintptr, name *byte) {
	mockCUDAEvent = event
}

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
	NameCUDADevice(0, "GPU 0")
	NameCUDAContext(1, "Context")
	NameCUDAStream(2, "Stream")
	NameCUDAEvent(3, "Event")

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

	NameCategory(1, "Compute")
	dom.NameCategory(2, "Transfer")
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
	fnNameCategoryA = purego.NewCallback(mockNameCategoryA)
	fnDomainNameCategoryA = purego.NewCallback(mockDomainNameCategoryA)
	fnNameCuDeviceA = purego.NewCallback(mockNameCuDeviceA)
	fnNameCuContextA = purego.NewCallback(mockNameCuContextA)
	fnNameCuStreamA = purego.NewCallback(mockNameCuStreamA)
	fnNameCuEventA = purego.NewCallback(mockNameCuEventA)

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
	NameCategory(7, "Compute")
	dom.NameCategory(9, "Transfer")
	if mockCategory != 7 || mockDomainCategory != 9 {
		t.Fatalf("category callbacks = global %d, domain %d", mockCategory, mockDomainCategory)
	}
	NameCUDADevice(3, "GPU 3")
	NameCUDAContext(0x101, "context")
	NameCUDAStream(0x202, "stream")
	NameCUDAEvent(0x303, "event")
	if mockCUDADevice != 3 || mockCUDAContext != 0x101 || mockCUDAStream != 0x202 || mockCUDAEvent != 0x303 {
		t.Fatalf("CUDA naming callbacks = device %d context %#x stream %#x event %#x", mockCUDADevice, mockCUDAContext, mockCUDAStream, mockCUDAEvent)
	}
	dom.Destroy()
}

func TestCallbackTableBounds(t *testing.T) {
	initTables()
	tests := []struct {
		name   string
		module int32
		want   uint32
		table  []uintptr
		last   *uintptr
	}{
		// Sizes are NVTX_CBID_*_SIZE from NVIDIA NVTX release-v3.
		{"core", cbModuleCore, 16, tableCORE[:], &fnNameOsThreadW},
		{"cuda", cbModuleCUDA, 9, tableCUDA[:], &slotsCUDA[len(slotsCUDA)-1]},
		{"opencl", cbModuleOpenCL, 15, tableOpenCL[:], &slotsOpenCL[len(slotsOpenCL)-1]},
		{"cudart", cbModuleCUDART, 7, tableCUDART[:], &slotsCUDART[len(slotsCUDART)-1]},
		{"core2", cbModuleCore2, 16, tableCORE2[:], &fnInitialize},
		{"sync", cbModuleSync, 7, tableSync[:], &slotsSync[len(slotsSync)-1]},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if len(test.table) != int(test.want)+1 {
				t.Fatalf("backing table length = %d, want %d including trailing guard", len(test.table), test.want+1)
			}
			var table uintptr
			var size uint32
			if got := getModuleFunctionTable(test.module, &table, &size); got != 1 {
				t.Fatalf("result = %d, want 1", got)
			}
			if table != uintptr(unsafe.Pointer(&test.table[0])) || size != test.want {
				t.Fatalf("table = %x, size = %d, want nonzero table and size %d", table, size, test.want)
			}
			entries := test.table
			if entries[0] != 0 {
				t.Fatalf("invalid callback slot = %x, want 0", entries[0])
			}
			if entries[size] != 0 {
				t.Fatalf("trailing guard = %x, want 0", entries[size])
			}
			for i := uint32(1); i < size; i++ {
				if entries[i] == 0 {
					t.Fatalf("callback slot %d is nil", i)
				}
			}
			if entries[size-1] != uintptr(unsafe.Pointer(test.last)) {
				t.Fatalf("maximum valid slot points to %x, want %x", entries[size-1], test.last)
			}
			old := *test.last
			const marker = uintptr(0xfeedface)
			*test.last = marker
			if *test.last != marker || entries[size] != 0 {
				t.Fatal("maximum valid provider write changed the trailing guard")
			}
			*test.last = old
		})
	}
}

func TestScopedRangeClosesOnce(t *testing.T) {
	initTables()
	fnRangePushA = purego.NewCallback(mockRangePushA)
	fnRangePop = purego.NewCallback(mockRangePop)
	enabled = true
	mockRangePushCount = 0
	mockRangePopCount = 0

	done := ScopedRange("scope")
	done()
	done()
	if mockRangePushCount != 1 || mockRangePopCount != 1 {
		t.Fatalf("callbacks = push %d, pop %d", mockRangePushCount, mockRangePopCount)
	}
}

func TestDomainDestroyWaitsForScopedRange(t *testing.T) {
	initTables()
	fnDomainRangePushEx = purego.NewCallback(mockDomainRangePushEx)
	fnDomainRangePop = purego.NewCallback(mockDomainRangePop)
	fnDomainDestroy = purego.NewCallback(mockDomainDestroy)
	enabled = true
	d := &Domain{raw: 1}
	done := d.ScopedRange("scope")
	destroyed := make(chan struct{})
	started := make(chan struct{})
	go func() {
		close(started)
		d.Destroy()
		close(destroyed)
	}()
	<-started
	runtime.Gosched()
	select {
	case <-destroyed:
		t.Fatal("Destroy returned while scoped range was active")
	default:
	}
	done()
	<-destroyed
	if d.Raw() != 0 {
		t.Fatalf("Raw = %x after Destroy", d.Raw())
	}
}
