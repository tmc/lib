package cupti

import (
	"bytes"
	"errors"
	"fmt"
	"runtime"
	"sort"
	"unsafe"
)

// PCSamplingReasonLayout describes the parameter sizes used by the native
// PC-sampling reason inventory calls on a 64-bit host.
//
// The layout is unchanged in the validated CUPTI releases listed by
// [PCSamplingReasonLayoutFor]. A release not listed there has not had its
// ABI checked and must be rejected by callers that pass these parameters.
type PCSamplingReasonLayout struct {
	CountParamsSize uint
	NamesParamsSize uint
}

// PCSamplingReasonLayoutFor reports the validated parameter layout for a CUPTI
// API version. 24, 26 and 28 are the CUDA 12 encodings, reported by the 12.6,
// 12.8 and 12.9 libraries on the GB10 host; 130001, 130101, 130200,
// 130202, 130301 and 130400 are the CUDA 13 versions whose cupti_pcsampling.h
// has been measured, 130200 being [ApiVersion], the header these types were
// generated from. A version absent from that list has no layout witness and is
// refused, because these structs are filled in place by CUPTI through the
// pointer it is given: a size declared for an unmeasured layout is a native
// write at the wrong offset rather than an error. The parameter structs contain
// pointers and are admitted only on 64-bit hosts.
func PCSamplingReasonLayoutFor(version uint32) (PCSamplingReasonLayout, bool) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		return PCSamplingReasonLayout{}, false
	}
	switch {
	case version == 24, version == 26, version == 28,
		version == 130001, version == 130101, version == 130200,
		version == 130202, version == 130301, version == 130400:
		return PCSamplingReasonLayout{
			CountParamsSize: uint(unsafe.Sizeof(PCSamplingGetNumStallReasonsParams{})),
			NamesParamsSize: uint(unsafe.Sizeof(PCSamplingGetStallReasonsParams{})),
		}, true
	default:
		return PCSamplingReasonLayout{}, false
	}
}

// PCSamplingStallReasonInfo is a name and native index returned by the PC
// sampling reason inventory query.
type PCSamplingStallReasonInfo struct {
	Index uint32
	Name  string
}

// PCSamplingReasonInventory is the validated reason table for one CUPTI
// version. The names are owned Go strings sorted by native index.
type PCSamplingReasonInventory struct {
	CUPTIVersion uint32
	Reasons      []PCSamplingStallReasonInfo
}

// PCSamplingStallReasons queries the native reason inventory without
// launching work. It owns the temporary C storage and rejects malformed or
// out-of-bounds native output before returning it to a caller.
func PCSamplingStallReasons(ctx Context, maxReasons int) (PCSamplingReasonInventory, error) {
	if ctx == 0 || maxReasons < 1 || maxReasons > 65536 {
		return PCSamplingReasonInventory{}, errors.New("cupti: invalid reason inventory request")
	}
	version, err := GetVersion()
	if err != nil {
		return PCSamplingReasonInventory{}, err
	}
	return pcSamplingStallReasons(ctx, maxReasons, version, PCSamplingGetNumStallReasons, PCSamplingGetStallReasons)
}

func pcSamplingStallReasons(ctx Context, maxReasons int, version uint32, countQuery func(*PCSamplingGetNumStallReasonsParams) error, namesQuery func(*PCSamplingGetStallReasonsParams) error) (PCSamplingReasonInventory, error) {
	if ctx == 0 || maxReasons < 1 || maxReasons > 65536 {
		return PCSamplingReasonInventory{}, errors.New("cupti: invalid reason inventory request")
	}
	layout, ok := PCSamplingReasonLayoutFor(version)
	if !ok {
		return PCSamplingReasonInventory{}, fmt.Errorf("cupti: unqualified reason inventory layout for CUPTI %d", version)
	}
	var pin runtime.Pinner
	defer pin.Unpin()
	var count uint
	pin.Pin(&count)
	q := PCSamplingGetNumStallReasonsParams{Size: layout.CountParamsSize, Ctx: ctx, NumStallReasons: uintptr(unsafe.Pointer(&count))}
	if err := countQuery(&q); err != nil {
		return PCSamplingReasonInventory{}, fmt.Errorf("cupti: query reason count: %w", err)
	}
	if count > uint(maxReasons) {
		return PCSamplingReasonInventory{}, fmt.Errorf("cupti: reason count %d exceeds limit %d", count, maxReasons)
	}
	out := PCSamplingReasonInventory{CUPTIVersion: version, Reasons: []PCSamplingStallReasonInfo{}}
	if count == 0 {
		return out, nil
	}
	indices := make([]uint32, int(count))
	names := make([][StallReasonStringSize]byte, int(count))
	pointers := make([]uintptr, int(count))
	for i := range pointers {
		for j := range names[i] {
			names[i][j] = 0xff
		}
		pin.Pin(&names[i][0])
		pointers[i] = uintptr(unsafe.Pointer(&names[i][0]))
	}
	pin.Pin(&indices[0])
	pin.Pin(&pointers[0])
	n := PCSamplingGetStallReasonsParams{Size: layout.NamesParamsSize, Ctx: ctx, NumStallReasons: count, StallReasonIndex: uintptr(unsafe.Pointer(&indices[0])), StallReasons: uintptr(unsafe.Pointer(&pointers[0]))}
	if err := namesQuery(&n); err != nil {
		return PCSamplingReasonInventory{}, fmt.Errorf("cupti: query reason names: %w", err)
	}
	if n.NumStallReasons != count || n.StallReasonIndex != uintptr(unsafe.Pointer(&indices[0])) || n.StallReasons != uintptr(unsafe.Pointer(&pointers[0])) {
		return PCSamplingReasonInventory{}, errors.New("cupti: native reason output changed storage")
	}
	seen := make(map[uint32]bool)
	seenNames := make(map[string]bool)
	for i, index := range indices {
		if pointers[i] != uintptr(unsafe.Pointer(&names[i][0])) {
			return PCSamplingReasonInventory{}, errors.New("cupti: native reason name changed storage")
		}
		end := bytes.IndexByte(names[i][:], 0)
		if end <= 0 {
			return PCSamplingReasonInventory{}, errors.New("cupti: empty or unterminated native reason")
		}
		name := string(names[i][:end])
		if seen[index] || seenNames[name] {
			return PCSamplingReasonInventory{}, errors.New("cupti: duplicate native reason")
		}
		seen[index], seenNames[name] = true, true
		out.Reasons = append(out.Reasons, PCSamplingStallReasonInfo{Index: index, Name: name})
	}
	sort.Slice(out.Reasons, func(i, j int) bool { return out.Reasons[i].Index < out.Reasons[j].Index })
	return out, nil
}
