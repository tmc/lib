package main

import (
	"errors"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/nvtx"
)

type memoryProbeOptions struct {
	Placement        bool
	Prefetch, Advice string
}
type memoryQuery struct {
	Attribute string `json:"attribute"`
	Value     *int32 `json:"value,omitempty"`
	Result    *int32 `json:"native_result"`
	Error     string `json:"error,omitempty"`
}
type memorySnapshot struct {
	Stage          string        `json:"stage"`
	Pointer        uint64        `json:"pointer"`
	Bytes          uint64        `json:"bytes"`
	Queries        []memoryQuery `json:"queries"`
	Interpretation string        `json:"interpretation"`
}
type memoryRequest struct {
	Operation string `json:"operation"`
	Location  string `json:"location"`
	Result    *int32 `json:"native_result"`
	Error     string `json:"error,omitempty"`
}

func memoryDriverResult(err error) *int32 {
	var value int32
	if err != nil {
		var code driver.Result
		if !errors.As(err, &code) {
			return nil
		}
		value = int32(code)
	}
	return &value
}
func queryMemorySnapshot(stage string, pointer driver.DevicePtr, bytes uint) memorySnapshot {
	snapshot := memorySnapshot{Stage: stage, Pointer: uint64(pointer), Bytes: uint64(bytes), Queries: []memoryQuery{}, Interpretation: "native advice/history attributes; preferred location and last-prefetch destination do not observe present physical page residency"}
	for _, attribute := range []driver.MemRangeAttribute{driver.MemRangeAttributeReadMostly, driver.MemRangeAttributePreferredLocation, driver.MemRangeAttributeLastPrefetchLocation, driver.MemRangeAttributePreferredLocationType, driver.MemRangeAttributePreferredLocationId, driver.MemRangeAttributeLastPrefetchLocationType, driver.MemRangeAttributeLastPrefetchLocationId} {
		var value int32
		var pin runtime.Pinner
		pin.Pin(&value)
		err := pointer.MemRangeGetAttribute(unsafe.Pointer(&value), 4, attribute, bytes)
		pin.Unpin()
		query := memoryQuery{Attribute: attribute.String(), Result: memoryDriverResult(err)}
		if err != nil {
			query.Error = err.Error()
		} else {
			query.Value = &value
		}
		snapshot.Queries = append(snapshot.Queries, query)
	}
	return snapshot
}
func prepareMemoryProbe(report *probeReport, dev *cuda.Device, pointer driver.DevicePtr, bytes uint, options memoryProbeOptions) error {
	if !options.Placement {
		return nil
	}
	return dev.WithContext(func() error {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		message, err := marker.Encode(marker.Metadata{Phase: "prefill", Operation: "managed_memory_setup"})
		if err != nil {
			return err
		}
		nvtx.RangePush(message)
		defer nvtx.RangePop()
		report.MemorySnapshots = append(report.MemorySnapshots, queryMemorySnapshot("allocated_before_advice_or_prefetch", pointer, bytes))
		if options.Advice != "none" {
			advice := driver.MemAdviseSetPreferredLocation
			if options.Advice == "read-mostly" {
				advice = driver.MemAdviseSetReadMostly
			}
			err := pointer.MemAdvise(bytes, advice, driver.MemLocation{Type: driver.MemLocationTypeDevice, Id: int32(dev.Ordinal())})
			request := memoryRequest{Operation: "advice_" + options.Advice, Location: "device", Result: memoryDriverResult(err)}
			if err != nil {
				request.Error = err.Error()
			}
			report.MemoryRequests = append(report.MemoryRequests, request)
		}
		if options.Prefetch != "none" {
			location := driver.MemLocation{Type: driver.MemLocationTypeDevice, Id: int32(dev.Ordinal())}
			if options.Prefetch == "host" {
				location = driver.MemLocation{Type: driver.MemLocationTypeHost}
			}
			err := pointer.MemPrefetchAsync(bytes, location, 0, 0)
			request := memoryRequest{Operation: "prefetch_request", Location: options.Prefetch, Result: memoryDriverResult(err)}
			if err != nil {
				request.Error = err.Error()
			}
			report.MemoryRequests = append(report.MemoryRequests, request)
			if err == nil {
				err = driver.CtxSynchronize()
				sync := memoryRequest{Operation: "synchronize_after_prefetch", Result: memoryDriverResult(err)}
				if err != nil {
					sync.Error = err.Error()
				}
				report.MemoryRequests = append(report.MemoryRequests, sync)
			}
		}
		report.MemorySnapshots = append(report.MemorySnapshots, queryMemorySnapshot("before_controlled_window", pointer, bytes))
		return nil
	})
}
