package sanitizer

import (
	"testing"
	"unsafe"
)

func TestNormalizeCallbackData(t *testing.T) {
	t.Run("event 13.0 inserted field", func(t *testing.T) {
		in := [4]uintptr{11, 22, 33, 44}
		p, kind := normalizeCallbackData(523, CbDomainEvents, uint32(CbidEventsCreated), unsafe.Pointer(&in[0]))
		got := *(*EventData)(p)
		if kind != callbackDataEvent || uintptr(got.Event) != 11 || uintptr(got.Context) != 22 || uintptr(got.Stream) != 33 || uintptr(got.HStream) != 44 || got.GreenCtx != 0 {
			t.Fatalf("normalized event data = %+v, kind %d", got, kind)
		}
		releaseCallbackData(kind, p)
	})

	t.Run("context resource 13.0 inserted field", func(t *testing.T) {
		in := [2]uintptr{11, 22}
		p, kind := normalizeCallbackData(523, CbDomainResource, uint32(CbidResourceContextCreationFinished), unsafe.Pointer(&in[0]))
		got := *(*ResourceContextData)(p)
		if kind != callbackDataResourceContext || uintptr(got.Context) != 11 || uintptr(got.Device) != 22 || got.GreenContext != 0 {
			t.Fatalf("normalized context data = %+v, kind %d", got, kind)
		}
		releaseCallbackData(kind, p)
	})

	t.Run("stream resource 13.0 inserted field", func(t *testing.T) {
		in := [3]uintptr{11, 22, 33}
		p, kind := normalizeCallbackData(523, CbDomainResource, uint32(CbidResourceStreamCreated), unsafe.Pointer(&in[0]))
		got := *(*ResourceStreamData)(p)
		if kind != callbackDataResourceStream || uintptr(got.Context) != 11 || uintptr(got.Stream) != 22 || uintptr(got.HStream) != 33 || got.GreenContext != 0 {
			t.Fatalf("normalized stream data = %+v, kind %d", got, kind)
		}
		releaseCallbackData(kind, p)
	})

	t.Run("latest layout passes through", func(t *testing.T) {
		in := EventData{Event: 11, Context: 22, GreenCtx: 33, Stream: 44, HStream: 55}
		p, kind := normalizeCallbackData(cuda134RuntimeCBIDSize, CbDomainEvents, uint32(CbidEventsCreated), unsafe.Pointer(&in))
		if p != unsafe.Pointer(&in) || kind != callbackDataNone {
			t.Fatalf("latest layout copied: pointer %p, kind %d", p, kind)
		}
	})

	t.Run("appended fields are zero before their release", func(t *testing.T) {
		var memcpy [120]byte
		p, kind := normalizeCallbackData(465, CbDomainMemcpy, uint32(CbidMemcpyStarting), unsafe.Pointer(&memcpy[0]))
		if got := *(*MemcpyData)(p); got.ApiContext != 0 || got.ApiStream != 0 || got.HApiStream != 0 {
			t.Errorf("old memcpy fields = %+v", got)
		}
		releaseCallbackData(kind, p)

		var batch [44]byte
		p, kind = normalizeCallbackData(523, CbDomainBatchMemop, uint32(CbidBatchMemopWrite), unsafe.Pointer(&batch[0]))
		if got := *(*BatchMemopData)(p); got.AtomicOperation != 0 {
			t.Errorf("old batch memop atomic operation = %v", got.AtomicOperation)
		}
		releaseCallbackData(kind, p)

		var syncData [24]byte
		p, kind = normalizeCallbackData(523, CbDomainSynchronize, 0, unsafe.Pointer(&syncData[0]))
		if got := *(*SynchronizeData)(p); got.GreenContext != 0 {
			t.Errorf("old synchronize green context = %v", got.GreenContext)
		}
		releaseCallbackData(kind, p)

		var launch [136]byte
		p, kind = normalizeCallbackData(559, CbDomainLaunch, 0, unsafe.Pointer(&launch[0]))
		if got := *(*LaunchData)(p); got.SplitLaunchPart != 0 {
			t.Errorf("old launch split part = %d", got.SplitLaunchPart)
		}
		releaseCallbackData(kind, p)
	})
}

func TestNormalizeCallbackDataAllocs(t *testing.T) {
	var oldEvent = [4]uintptr{1, 2, 3, 4}
	old := unsafe.Pointer(&oldEvent[0])
	p, kind := normalizeCallbackData(523, CbDomainEvents, uint32(CbidEventsCreated), old)
	releaseCallbackData(kind, p)
	if allocs := testing.AllocsPerRun(100, func() {
		p, kind := normalizeCallbackData(523, CbDomainEvents, uint32(CbidEventsCreated), old)
		releaseCallbackData(kind, p)
	}); allocs != 0 {
		t.Errorf("older layout allocates %v times per callback", allocs)
	}

	latest := unsafe.Pointer(&EventData{})
	if allocs := testing.AllocsPerRun(100, func() {
		p, kind := normalizeCallbackData(cuda134RuntimeCBIDSize, CbDomainEvents, uint32(CbidEventsCreated), latest)
		if p != latest || kind != callbackDataNone {
			panic("latest callback data changed")
		}
	}); allocs != 0 {
		t.Errorf("latest layout allocates %v times per callback", allocs)
	}

	var resourceContext [2]uintptr
	var resourceStream [3]uintptr
	var memcpy [120]byte
	var batch [44]byte
	var synchronize [24]byte
	var launch [136]byte
	oldPaths := []func(){
		func() {
			p, k := normalizeCallbackData(523, CbDomainResource, uint32(CbidResourceContextCreationFinished), unsafe.Pointer(&resourceContext[0]))
			releaseCallbackData(k, p)
		},
		func() {
			p, k := normalizeCallbackData(523, CbDomainResource, uint32(CbidResourceStreamCreated), unsafe.Pointer(&resourceStream[0]))
			releaseCallbackData(k, p)
		},
		func() {
			p, k := normalizeCallbackData(465, CbDomainMemcpy, uint32(CbidMemcpyStarting), unsafe.Pointer(&memcpy[0]))
			releaseCallbackData(k, p)
		},
		func() {
			p, k := normalizeCallbackData(523, CbDomainBatchMemop, uint32(CbidBatchMemopWrite), unsafe.Pointer(&batch[0]))
			releaseCallbackData(k, p)
		},
		func() {
			p, k := normalizeCallbackData(523, CbDomainSynchronize, 0, unsafe.Pointer(&synchronize[0]))
			releaseCallbackData(k, p)
		},
		func() {
			p, k := normalizeCallbackData(559, CbDomainLaunch, 0, unsafe.Pointer(&launch[0]))
			releaseCallbackData(k, p)
		},
	}
	for i, run := range oldPaths {
		run()
		if allocs := testing.AllocsPerRun(100, run); allocs != 0 {
			t.Errorf("older layout path %d allocates %v times per callback", i, allocs)
		}
	}
}
