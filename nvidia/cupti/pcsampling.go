package cupti

import (
	"bytes"
	"errors"
	"fmt"
	"runtime"
	"sort"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/cuptiversion"
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
// has been measured, 130200 being the historical API header version these types were
// generated from. These structs are filled in place by CUPTI through the
// pointer it is given, so the declared size must be one that was measured.
// Admission is exact, except historical source admission of 130401 using
// layout 130400. This is not fresh physical qualification. The structs contain
// pointers and are admitted only on 64-bit hosts.
func PCSamplingReasonLayoutFor(version uint32) (PCSamplingReasonLayout, bool) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		return PCSamplingReasonLayout{}, false
	}
	if _, ok := cuptiversion.AdmittedLayout(version, 24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400); !ok {
		return PCSamplingReasonLayout{}, false
	}
	return PCSamplingReasonLayout{
		CountParamsSize: uint(unsafe.Sizeof(PCSamplingGetNumStallReasonsParams{})),
		NamesParamsSize: uint(unsafe.Sizeof(PCSamplingGetStallReasonsParams{})),
	}, true
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
func PCSamplingStallReasons(ctx driver.Context, maxReasons int) (PCSamplingReasonInventory, error) {
	if ctx == 0 || maxReasons < 1 || maxReasons > 65536 {
		return PCSamplingReasonInventory{}, errors.New("cupti: invalid reason inventory request")
	}
	version, err := GetVersion()
	if err != nil {
		return PCSamplingReasonInventory{}, err
	}
	return pcSamplingStallReasons(ctx, maxReasons, version,
		func(p *PCSamplingGetNumStallReasonsParams, _ *uint) error { return PCSamplingGetNumStallReasons(p) },
		func(p *PCSamplingGetStallReasonsParams, _ pcSamplingReasonStorage) error {
			return PCSamplingGetStallReasons(p)
		})
}

// pcSamplingReasonStorage is the typed Go storage behind the addresses in a
// [PCSamplingGetStallReasonsParams]. CUPTI ignores it; a test query uses it to
// check the wire addresses and to write results without converting integers
// back into pointers.
type pcSamplingReasonStorage struct {
	indices  []uint32
	pointers []uintptr
	names    [][StallReasonStringSize]byte
}

// pcSamplingStallReasons implements [PCSamplingStallReasons]. Each query
// receives the native parameters and the typed storage their address fields
// refer to.
func pcSamplingStallReasons(ctx driver.Context, maxReasons int, version uint32, countQuery func(*PCSamplingGetNumStallReasonsParams, *uint) error, namesQuery func(*PCSamplingGetStallReasonsParams, pcSamplingReasonStorage) error) (PCSamplingReasonInventory, error) {
	if ctx == 0 || maxReasons < 1 || maxReasons > 65536 {
		return PCSamplingReasonInventory{}, errors.New("cupti: invalid reason inventory request")
	}
	layout, ok := PCSamplingReasonLayoutFor(version)
	if !ok {
		return PCSamplingReasonInventory{}, fmt.Errorf("cupti: reason inventory layout for CUPTI %d: %w", version, errors.ErrUnsupported)
	}
	var pin runtime.Pinner
	defer pin.Unpin()
	var count uint
	pin.Pin(&count)
	q := PCSamplingGetNumStallReasonsParams{Size: layout.CountParamsSize, Ctx: ctx, NumStallReasons: uintptr(unsafe.Pointer(&count))}
	if err := countQuery(&q, &count); err != nil {
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
	if err := namesQuery(&n, pcSamplingReasonStorage{indices: indices, pointers: pointers, names: names}); err != nil {
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
