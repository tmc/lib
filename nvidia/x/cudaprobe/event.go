package cudaprobe

import (
	"context"
	"encoding/binary"
	"fmt"
)

// Process identifies a process lifetime. Generation is a collector-supplied
// start identity, not merely a PID, so PID reuse cannot join allocations.
type Process struct {
	PID        uint32
	Generation uint64
}

// Identity records the provenance of a probe target. ABI is a capture contract,
// not an inferred property of a symbol name. Version identifies the observed
// CUDA library; BuildID identifies its exact image.
type Identity struct {
	ThreadGenerationKnown bool
	// ThreadGeneration is a verified Linux task birth identity; zero is unknown.
	ThreadID         uint32
	ThreadGeneration uint64
	Process          Process
	// Provenance identifies the capture source, for example "uprobe" or
	// "synthetic". Interposed reports that an in-process shim was also active.
	Provenance string
	Interposed bool
	// AddressSpace identifies a CUDA address-space lifetime, including context
	// reuse. Zero means unavailable; it must not join allocation observations.
	AddressSpace                           uint64
	Library, BuildID, Version, Symbol, ABI string
}

// Capture contains one completed API call, or one entry-only observation.
// Data is the fixed record accepted by Decode. Identity is resolved by the
// collector; it is retained even when decoding is unavailable.
type Capture struct {
	Identity Identity
	Data     [128]byte
}

// Source supplies observations in process sequence order. Next returns io.EOF
// after target exit and honors context cancellation. The collector owns attach,
// detach and drop accounting; sequence gaps must include dropped observations.
type Source interface {
	Next(context.Context) (Capture, error)
}

// Launch describes API launch arguments, not kernel execution or tensor shapes.
type Launch struct {
	Function            uint64
	Grid, Block         [3]uint32
	SharedBytes, Stream uint64
}

// Copy describes API copy arguments. Addresses are opaque observations, not
// dereferenceable Go pointers. Direction is explicit for Driver API calls;
// Runtime API calls retain Kind without resolving cudaMemcpyDefault addresses.
// Async identifies the API variant, not device completion or host blocking.
type Copy struct {
	Destination, Source, Bytes, Stream uint64
	Kind                               int32
	Direction                          string
	Async                              bool
}

// Sync describes the synchronization scope requested by an API call. A decoded
// request does not establish device completion; retain ReturnKnown and Status.
type Sync struct {
	Scope  string
	Stream uint64
}

// Graph describes a graph executable submission. Exec is an opaque handle;
// it does not identify individual graph nodes or their launch geometry.
type Graph struct {
	Exec, Stream uint64
}

// AsyncMemory records stream-ordered allocation/free API arguments. A returned
// address does not establish device lifetime completion or allocation ordering.
type AsyncMemory struct {
	Address, Bytes, Stream uint64
	Free                   bool
}

// Control records explicit context or stream API handles. Handles are opaque;
// these observations do not establish an allocation address-space lifetime.
type Control struct {
	Scope, Operation string
	Handle           uint64
	Flags            uint32
	Device           int32
}

// Event is a decoded observation. Reason is nonempty when Decoded is false.
// Timestamp is in the collector's host monotonic clock, not GPU time or Unix
// time. ReturnKnown distinguishes success from an unobserved return value.
type Event struct {
	Identity            Identity
	Sequence, Timestamp uint64
	Decoded             bool
	Reason              string
	ReturnKnown         bool
	Status              int32
	Launch              *Launch
	Copy                *Copy
	Sync                *Sync
	Graph               *Graph
	AsyncMemory         *AsyncMemory
	Control             *Control
	Address, Bytes      uint64
}

// Decode decodes capture format version 1, a 128-byte little-endian record:
// version u16 at 0, flags u16 at 2, reserved zero u32 at 4, process sequence
// u64 at 8, host monotonic timestamp u64 at 16, return status i32 at 24,
// reserved zero u32 at 28, eleven u64 argument slots at 32, and the API output
// pointer value u64 at 120. Flags are arguments captured (1), return
// captured (2), and output captured after a successful return (4).
//
// For ABI "linux-amd64-sysv-v1", argument slots are rdi, rsi, rdx, rcx, r8,
// r9, then five stack arguments starting at entry rsp+8. For ABI
// "linux-arm64-aapcs64-v1", slots are x0..x7 followed by three stack
// arguments starting at entry sp. Both ABIs normalize complete slots before
// encoding; decoding does not select an ABI from the replay host. The collector must
// never dereference kernelParams to infer kernel arguments. For cudaMalloc
// and cuMemAlloc_v2, stream-ordered allocation, context/stream creation and
// current-context queries, the output value is read from the saved first argument
// only on successful return. This format is a cudaprobe contract, not the
// unversioned Strobelight gpukern_sample structure.
//
// Malformed records return an error. Unknown versions, ABIs, symbols and
// incomplete captures return an undecoded Event retaining its identity.
func Decode(c Capture) (Event, error) {
	e := Event{Identity: c.Identity}
	b := c.Data[:]
	u := binary.LittleEndian
	e.Sequence, e.Timestamp = u.Uint64(b[8:]), u.Uint64(b[16:])
	unknown := func(reason string) (Event, error) { e.Reason = reason; return e, nil }
	if u.Uint16(b) != 1 {
		return unknown("unsupported capture version")
	}
	if u.Uint32(b[4:]) != 0 || u.Uint32(b[28:]) != 0 {
		return e, fmt.Errorf("cudaprobe: nonzero reserved capture fields")
	}
	flags := u.Uint16(b[2:])
	if flags & ^uint16(7) != 0 {
		return unknown("unsupported capture flags")
	}
	if e.Identity.Process.PID == 0 || e.Identity.Process.Generation == 0 || e.Sequence == 0 {
		return e, fmt.Errorf("cudaprobe: missing process identity or sequence")
	}
	if e.Identity.BuildID == "" || e.Identity.Library == "" || e.Identity.Version == "" {
		return unknown("missing library identity")
	}
	if e.Identity.ABI != "linux-amd64-sysv-v1" && e.Identity.ABI != "linux-arm64-aapcs64-v1" {
		return unknown("unsupported capture ABI")
	}
	if flags&1 == 0 {
		return unknown("arguments unavailable")
	}
	e.ReturnKnown = flags&2 != 0
	if e.ReturnKnown {
		e.Status = int32(u.Uint32(b[24:]))
	}
	var a [11]uint64
	for i := range a {
		a[i] = u.Uint64(b[32+8*i:])
	}
	switch e.Identity.Symbol {
	case "cudaLaunchKernel":
		e.Launch = &Launch{Function: a[0], Grid: [3]uint32{uint32(a[1]), uint32(a[1] >> 32), uint32(a[2])}, Block: [3]uint32{uint32(a[3]), uint32(a[3] >> 32), uint32(a[4])}, SharedBytes: a[6], Stream: a[7]}
	case "cuLaunchKernel":
		e.Launch = &Launch{Function: a[0], Grid: [3]uint32{uint32(a[1]), uint32(a[2]), uint32(a[3])}, Block: [3]uint32{uint32(a[4]), uint32(a[5]), uint32(a[6])}, SharedBytes: uint64(uint32(a[7])), Stream: a[8]}
	case "cudaMemcpy", "cudaMemcpyAsync":
		e.Copy = &Copy{Destination: a[0], Source: a[1], Bytes: a[2], Kind: int32(a[3]), Direction: "runtime_kind", Async: e.Identity.Symbol == "cudaMemcpyAsync"}
		if e.Copy.Async {
			e.Copy.Stream = a[4]
		}
	case "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2", "cuMemcpyDtoD_v2", "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoDAsync_v2":
		e.Copy = &Copy{Destination: a[0], Source: a[1], Bytes: a[2]}
		switch e.Identity.Symbol {
		case "cuMemcpyHtoD_v2", "cuMemcpyHtoDAsync_v2":
			e.Copy.Direction = "host_to_device"
		case "cuMemcpyDtoH_v2", "cuMemcpyDtoHAsync_v2":
			e.Copy.Direction = "device_to_host"
		case "cuMemcpyDtoD_v2", "cuMemcpyDtoDAsync_v2":
			e.Copy.Direction = "device_to_device"
		}
		switch e.Identity.Symbol {
		case "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoDAsync_v2":
			e.Copy.Async = true
			e.Copy.Stream = a[3]
		}
	case "cudaDeviceSynchronize":
		e.Sync = &Sync{Scope: "device"}
	case "cuCtxSynchronize":
		e.Sync = &Sync{Scope: "context"}
	case "cudaStreamSynchronize", "cuStreamSynchronize":
		e.Sync = &Sync{Scope: "stream", Stream: a[0]}
	case "cudaGraphLaunch", "cuGraphLaunch":
		e.Graph = &Graph{Exec: a[0], Stream: a[1]}
	case "cudaMallocAsync", "cuMemAllocAsync":
		e.AsyncMemory = &AsyncMemory{Bytes: a[1], Stream: a[2]}
		if !e.ReturnKnown {
			return unknown("allocation return unavailable")
		}
		if e.Status == 0 {
			if flags&4 == 0 {
				return unknown("allocation output unavailable")
			}
			e.AsyncMemory.Address = u.Uint64(b[120:])
		}
	case "cudaFreeAsync", "cuMemFreeAsync":
		e.AsyncMemory = &AsyncMemory{Address: a[0], Stream: a[1], Free: true}
	case "cuCtxCreate_v2", "cuCtxGetCurrent", "cuStreamCreate", "cudaStreamCreate":
		e.Control = &Control{Scope: "stream", Operation: "create"}
		switch e.Identity.Symbol {
		case "cuCtxCreate_v2":
			e.Control.Scope = "context"
			e.Control.Flags = uint32(a[1])
			e.Control.Device = int32(a[2])
		case "cuCtxGetCurrent":
			e.Control.Scope = "context"
			e.Control.Operation = "get_current"
		case "cuStreamCreate":
			e.Control.Flags = uint32(a[1])
		}
		if !e.ReturnKnown {
			return unknown("control return unavailable")
		}
		if e.Status == 0 {
			if flags&4 == 0 {
				return unknown("control output unavailable")
			}
			e.Control.Handle = u.Uint64(b[120:])
		}
	case "cuCtxSetCurrent", "cuCtxDestroy_v2", "cuStreamDestroy_v2", "cudaStreamDestroy":
		e.Control = &Control{Scope: "stream", Operation: "destroy", Handle: a[0]}
		switch e.Identity.Symbol {
		case "cuCtxSetCurrent":
			e.Control.Scope = "context"
			e.Control.Operation = "set_current"
		case "cuCtxDestroy_v2":
			e.Control.Scope = "context"
		}
	case "cudaMalloc", "cuMemAlloc_v2":
		if !e.ReturnKnown {
			return unknown("allocation return unavailable")
		}
		e.Bytes = a[1]
		if e.Status == 0 {
			if flags&4 == 0 {
				return unknown("allocation output unavailable")
			}
			e.Address = u.Uint64(b[120:])
		}
	case "cudaFree", "cuMemFree_v2":
		if !e.ReturnKnown {
			return unknown("free return unavailable")
		}
		e.Address = a[0]
	default:
		return unknown("unsupported CUDA symbol")
	}
	e.Decoded = true
	return e, nil
}
