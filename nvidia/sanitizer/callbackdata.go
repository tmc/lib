package sanitizer

import (
	"sync"
	"unsafe"
)

const cuda134RuntimeCBIDSize = 566

type callbackDataKind uint8

const (
	callbackDataNone callbackDataKind = iota
	callbackDataEvent
	callbackDataResourceContext
	callbackDataResourceStream
	callbackDataMemcpy
	callbackDataBatchMemop
	callbackDataSynchronize
	callbackDataLaunch
)

var (
	eventDataPool           = sync.Pool{New: func() any { return new(EventData) }}
	resourceContextDataPool = sync.Pool{New: func() any { return new(ResourceContextData) }}
	resourceStreamDataPool  = sync.Pool{New: func() any { return new(ResourceStreamData) }}
	memcpyDataPool          = sync.Pool{New: func() any { return new(MemcpyData) }}
	batchMemopDataPool      = sync.Pool{New: func() any { return new(BatchMemopData) }}
	synchronizeDataPool     = sync.Pool{New: func() any { return new(SynchronizeData) }}
	launchDataPool          = sync.Pool{New: func() any { return new(LaunchData) }}
)

// normalizeCallbackData copies callback data from older CUDA layouts into the
// package's CUDA 13.4 Go layout. The returned kind identifies the pool to use
// after the user's callback returns.
func normalizeCallbackData(size int, domain CallbackDomain, cbid uint32, data unsafe.Pointer) (unsafe.Pointer, callbackDataKind) {
	if data == nil || size == cuda134RuntimeCBIDSize {
		return data, callbackDataNone
	}
	switch domain {
	case CbDomainEvents:
		out := eventDataPool.Get().(*EventData)
		*out = EventData{}
		layout := callbackDataLayouts[size]
		copyField(unsafe.Pointer(out), data, unsafe.Offsetof(EventData{}.Event), layout.EventData[0], unsafe.Sizeof(out.Event))
		copyField(unsafe.Pointer(out), data, unsafe.Offsetof(EventData{}.Context), layout.EventData[1], unsafe.Sizeof(out.Context))
		copyField(unsafe.Pointer(out), data, unsafe.Offsetof(EventData{}.GreenCtx), layout.EventData[2], unsafe.Sizeof(out.GreenCtx))
		copyField(unsafe.Pointer(out), data, unsafe.Offsetof(EventData{}.Stream), layout.EventData[3], unsafe.Sizeof(out.Stream))
		copyField(unsafe.Pointer(out), data, unsafe.Offsetof(EventData{}.HStream), layout.EventData[4], unsafe.Sizeof(out.HStream))
		return unsafe.Pointer(out), callbackDataEvent
	case CbDomainResource:
		switch cbid {
		case uint32(CbidResourceContextCreationStarting), uint32(CbidResourceContextCreationFinished), uint32(CbidResourceContextDestroyStarting), uint32(CbidResourceContextDestroyFinished):
			out := resourceContextDataPool.Get().(*ResourceContextData)
			*out = ResourceContextData{}
			layout := callbackDataLayouts[size]
			copyField(unsafe.Pointer(out), data, unsafe.Offsetof(ResourceContextData{}.Context), layout.ResourceContextData[0], unsafe.Sizeof(out.Context))
			copyField(unsafe.Pointer(out), data, unsafe.Offsetof(ResourceContextData{}.GreenContext), layout.ResourceContextData[1], unsafe.Sizeof(out.GreenContext))
			copyField(unsafe.Pointer(out), data, unsafe.Offsetof(ResourceContextData{}.Device), layout.ResourceContextData[2], unsafe.Sizeof(out.Device))
			return unsafe.Pointer(out), callbackDataResourceContext
		case uint32(CbidResourceStreamCreated), uint32(CbidResourceStreamDestroyStarting), uint32(CbidResourceStreamDestroyFinished):
			out := resourceStreamDataPool.Get().(*ResourceStreamData)
			*out = ResourceStreamData{}
			layout := callbackDataLayouts[size]
			copyField(unsafe.Pointer(out), data, unsafe.Offsetof(ResourceStreamData{}.Context), layout.ResourceStreamData[0], unsafe.Sizeof(out.Context))
			copyField(unsafe.Pointer(out), data, unsafe.Offsetof(ResourceStreamData{}.GreenContext), layout.ResourceStreamData[1], unsafe.Sizeof(out.GreenContext))
			copyField(unsafe.Pointer(out), data, unsafe.Offsetof(ResourceStreamData{}.Stream), layout.ResourceStreamData[2], unsafe.Sizeof(out.Stream))
			copyField(unsafe.Pointer(out), data, unsafe.Offsetof(ResourceStreamData{}.HStream), layout.ResourceStreamData[3], unsafe.Sizeof(out.HStream))
			return unsafe.Pointer(out), callbackDataResourceStream
		}
	case CbDomainMemcpy:
		if cbid == uint32(CbidMemcpyStarting) || cbid == uint32(CbidMemcpyBatchItemStarting) {
			out := memcpyDataPool.Get().(*MemcpyData)
			*out = MemcpyData{}
			copyMemcpyData(out, data, callbackDataLayouts[size])
			return unsafe.Pointer(out), callbackDataMemcpy
		}
	case CbDomainBatchMemop:
		out := batchMemopDataPool.Get().(*BatchMemopData)
		*out = BatchMemopData{}
		copyBatchMemopData(out, data, callbackDataLayouts[size])
		return unsafe.Pointer(out), callbackDataBatchMemop
	case CbDomainSynchronize:
		out := synchronizeDataPool.Get().(*SynchronizeData)
		*out = SynchronizeData{}
		copyField(unsafe.Pointer(out), data, unsafe.Offsetof(SynchronizeData{}.Context), callbackDataLayouts[size].SynchronizeData[0], unsafe.Sizeof(out.Context))
		copyField(unsafe.Pointer(out), data, unsafe.Offsetof(SynchronizeData{}.Stream), callbackDataLayouts[size].SynchronizeData[1], unsafe.Sizeof(out.Stream))
		copyField(unsafe.Pointer(out), data, unsafe.Offsetof(SynchronizeData{}.HStream), callbackDataLayouts[size].SynchronizeData[2], unsafe.Sizeof(out.HStream))
		copyField(unsafe.Pointer(out), data, unsafe.Offsetof(SynchronizeData{}.GreenContext), callbackDataLayouts[size].SynchronizeData[3], unsafe.Sizeof(out.GreenContext))
		return unsafe.Pointer(out), callbackDataSynchronize
	case CbDomainLaunch:
		out := launchDataPool.Get().(*LaunchData)
		*out = LaunchData{}
		copyLaunchData(out, data, callbackDataLayouts[size])
		return unsafe.Pointer(out), callbackDataLaunch
	}
	return data, callbackDataNone
}

func copyField(dst, src unsafe.Pointer, dstOffset uintptr, srcOffset int, size uintptr) {
	if srcOffset < 0 {
		return
	}
	copy(unsafe.Slice((*byte)(unsafe.Add(dst, dstOffset)), int(size)), unsafe.Slice((*byte)(unsafe.Add(src, uintptr(srcOffset))), int(size)))
}

func copyMemcpyData(out *MemcpyData, data unsafe.Pointer, layout callbackStructOffsets) {
	dst := unsafe.Pointer(out)
	o := layout.MemcpyData
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.SrcContext), o[0], unsafe.Sizeof(out.SrcContext))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.DstContext), o[1], unsafe.Sizeof(out.DstContext))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.SrcStream), o[2], unsafe.Sizeof(out.SrcStream))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.HSrcStream), o[3], unsafe.Sizeof(out.HSrcStream))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.DstStream), o[4], unsafe.Sizeof(out.DstStream))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.HDstStream), o[5], unsafe.Sizeof(out.HDstStream))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.SrcAddress), o[6], unsafe.Sizeof(out.SrcAddress))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.DstAddress), o[7], unsafe.Sizeof(out.DstAddress))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.Size), o[8], unsafe.Sizeof(out.Size))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.Width), o[9], unsafe.Sizeof(out.Width))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.Height), o[10], unsafe.Sizeof(out.Height))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.Depth), o[11], unsafe.Sizeof(out.Depth))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.SrcPitch), o[12], unsafe.Sizeof(out.SrcPitch))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.DstPitch), o[13], unsafe.Sizeof(out.DstPitch))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.IsAsync), o[14], unsafe.Sizeof(out.IsAsync))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.Direction), o[15], unsafe.Sizeof(out.Direction))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.ApiContext), o[16], unsafe.Sizeof(out.ApiContext))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.ApiStream), o[17], unsafe.Sizeof(out.ApiStream))
	copyField(dst, data, unsafe.Offsetof(MemcpyData{}.HApiStream), o[18], unsafe.Sizeof(out.HApiStream))
}

func copyBatchMemopData(out *BatchMemopData, data unsafe.Pointer, layout callbackStructOffsets) {
	dst := unsafe.Pointer(out)
	o := layout.BatchMemopData
	copyField(dst, data, unsafe.Offsetof(BatchMemopData{}.Context), o[0], unsafe.Sizeof(out.Context))
	copyField(dst, data, unsafe.Offsetof(BatchMemopData{}.Stream), o[1], unsafe.Sizeof(out.Stream))
	copyField(dst, data, unsafe.Offsetof(BatchMemopData{}.HStream), o[2], unsafe.Sizeof(out.HStream))
	copyField(dst, data, unsafe.Offsetof(BatchMemopData{}.Address), o[3], unsafe.Sizeof(out.Address))
	copyField(dst, data, unsafe.Offsetof(BatchMemopData{}.Value), o[4], unsafe.Sizeof(out.Value))
	copyField(dst, data, unsafe.Offsetof(BatchMemopData{}.Type), o[5], unsafe.Sizeof(out.Type))
	copyField(dst, data, unsafe.Offsetof(BatchMemopData{}.AtomicOperation), o[6], unsafe.Sizeof(out.AtomicOperation))
}

func copyLaunchData(out *LaunchData, data unsafe.Pointer, layout callbackStructOffsets) {
	dst := unsafe.Pointer(out)
	o := layout.LaunchData
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.Context), o[0], unsafe.Sizeof(out.Context))
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.Stream), o[1], unsafe.Sizeof(out.Stream))
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.HStream), o[2], unsafe.Sizeof(out.HStream))
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.Module), o[3], unsafe.Sizeof(out.Module))
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.Function), o[4], unsafe.Sizeof(out.Function))
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.FunctionName), o[5], unsafe.Sizeof(out.FunctionName))
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.GridDim), o[6], 4)
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.GridDim)+4, o[7], 4)
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.GridDim)+8, o[8], 4)
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.BlockDim), o[9], 4)
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.BlockDim)+4, o[10], 4)
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.BlockDim)+8, o[11], 4)
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.ClusterDim), o[12], 4)
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.ClusterDim)+4, o[13], 4)
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.ClusterDim)+8, o[14], 4)
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.GridId), o[15], unsafe.Sizeof(out.GridId))
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.HLaunch), o[16], unsafe.Sizeof(out.HLaunch))
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.Device), o[17], unsafe.Sizeof(out.Device))
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.ApiContext), o[18], unsafe.Sizeof(out.ApiContext))
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.ApiStream), o[19], unsafe.Sizeof(out.ApiStream))
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.HApiStream), o[20], unsafe.Sizeof(out.HApiStream))
	copyField(dst, data, unsafe.Offsetof(LaunchData{}.SplitLaunchPart), o[21], unsafe.Sizeof(out.SplitLaunchPart))
}

func releaseCallbackData(kind callbackDataKind, data unsafe.Pointer) {
	switch kind {
	case callbackDataEvent:
		*(*EventData)(data) = EventData{}
		eventDataPool.Put((*EventData)(data))
	case callbackDataResourceContext:
		*(*ResourceContextData)(data) = ResourceContextData{}
		resourceContextDataPool.Put((*ResourceContextData)(data))
	case callbackDataResourceStream:
		*(*ResourceStreamData)(data) = ResourceStreamData{}
		resourceStreamDataPool.Put((*ResourceStreamData)(data))
	case callbackDataMemcpy:
		*(*MemcpyData)(data) = MemcpyData{}
		memcpyDataPool.Put((*MemcpyData)(data))
	case callbackDataBatchMemop:
		*(*BatchMemopData)(data) = BatchMemopData{}
		batchMemopDataPool.Put((*BatchMemopData)(data))
	case callbackDataSynchronize:
		*(*SynchronizeData)(data) = SynchronizeData{}
		synchronizeDataPool.Put((*SynchronizeData)(data))
	case callbackDataLaunch:
		*(*LaunchData)(data) = LaunchData{}
		launchDataPool.Put((*LaunchData)(data))
	}
}
