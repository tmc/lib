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
	Process Process
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
	Address, Bytes      uint64
}

// Decode decodes capture format version 1, a 128-byte little-endian record:
// version u16 at 0, flags u16 at 2, reserved zero u32 at 4, process sequence
// u64 at 8, host monotonic timestamp u64 at 16, return status i32 at 24,
// reserved zero u32 at 28, eleven u64 argument slots at 32, and the allocation
// output pointer value u64 at 120. Flags are arguments captured (1), return
// captured (2), and allocation output captured after a successful return (4).
//
// For ABI "linux-amd64-sysv-v1", argument slots are rdi, rsi, rdx, rcx, r8,
// r9, then five stack arguments starting at entry rsp+8. The collector must
// never dereference kernelParams to infer kernel arguments. For cudaMalloc
// and cuMemAlloc_v2, the output value is read from the saved first argument
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
	if e.Identity.ABI != "linux-amd64-sysv-v1" {
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
