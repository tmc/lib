package cuptipcsampling

import (
	"bytes"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"sort"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// StallReason identifies a reason returned by the native PC sampling API.
// Index is not a legacy ActivityPCSamplingStallReason enum value. Names are
// preserved verbatim, including distinctions such as not-issued reasons.
type StallReason struct {
	Index uint32
	Name  string
}

// ReasonInventory retains the native version with the queried reason table.
// Discovery alone establishes neither sampling permission nor collected data.
type ReasonInventory struct {
	CUPTIVersion uint32
	Reasons      []StallReason
}

// Lookup returns the name of the stall reason for the given native index,
// or an empty string if not found.
//
// Precondition: inv.Reasons must be strictly sorted ascending by Index.
// StallReasons guarantees this ordering; callers constructing ReasonInventory
// literals manually must maintain it for binary search.
func (inv ReasonInventory) Lookup(index uint32) string {
	idx, ok := slices.BinarySearchFunc(inv.Reasons, index, func(r StallReason, target uint32) int {
		if r.Index < target {
			return -1
		}
		if r.Index > target {
			return 1
		}
		return 0
	})
	if ok {
		return inv.Reasons[idx].Name
	}
	return ""
}

// StallReasons queries the modern native reason inventory without launching
// work. PC sampling must already be enabled on ctx: CUPTI rejects the query
// with CUPTI_ERROR_INVALID_OPERATION otherwise, an error its header does not
// document for this call. Use [Enable], or query through [Start], which enables
// the context before it reads the inventory. The caller must keep ctx alive and
// coordinate its use with other collectors. maxReasons must be between 1 and
// 65536. Exceeding the limit or receiving malformed output fails without a
// partial inventory.
// Names and IDs are owned copies sorted by ID. Native errors remain inspectable.
//
// The 64-bit parameter layouts for CUPTI 27 and CUDA 13 releases (13.x) are admitted;
// other versions fail before querying the layout. Runtime/device support must
// still be established by the native result on the selected configuration.
func StallReasons(ctx cupti.Context, maxReasons int) (ReasonInventory, error) {
	if ctx == 0 || maxReasons < 1 || maxReasons > 65536 {
		return ReasonInventory{}, errors.New("cuptipcsampling: invalid reason inventory request")
	}
	inv, err := cupti.PCSamplingStallReasons(ctx, maxReasons)
	if err != nil {
		return ReasonInventory{}, err
	}
	out := ReasonInventory{CUPTIVersion: inv.CUPTIVersion, Reasons: make([]StallReason, len(inv.Reasons))}
	for i, reason := range inv.Reasons {
		out.Reasons[i] = StallReason{Index: reason.Index, Name: reason.Name}
	}
	return out, nil
}

func stallReasons(ctx cupti.Context, maxReasons int, version uint32, countQuery func(*cupti.PCSamplingGetNumStallReasonsParams) error, namesQuery func(*cupti.PCSamplingGetStallReasonsParams) error) (ReasonInventory, error) {
	if ctx == 0 || maxReasons < 1 || maxReasons > 65536 {
		return ReasonInventory{}, errors.New("cuptipcsampling: invalid reason inventory request")
	}
	layout, ok := cupti.PCSamplingReasonLayoutFor(version)
	if !ok {
		return ReasonInventory{}, fmt.Errorf("cuptipcsampling: unqualified reason inventory layout for CUPTI %d", version)
	}
	var pin runtime.Pinner
	defer pin.Unpin()
	count := ^uint(0)
	pin.Pin(&count)
	q := cupti.PCSamplingGetNumStallReasonsParams{Size: layout.CountParamsSize, Ctx: ctx, NumStallReasons: uintptr(unsafe.Pointer(&count))}
	if err := countQuery(&q); err != nil {
		return ReasonInventory{}, fmt.Errorf("cuptipcsampling: query reason count: %w", err)
	}
	if count > uint(maxReasons) {
		return ReasonInventory{}, fmt.Errorf("cuptipcsampling: reason count %d exceeds limit %d", count, maxReasons)
	}
	out := ReasonInventory{CUPTIVersion: version, Reasons: []StallReason{}}
	if count == 0 {
		return out, nil
	}
	indices := make([]uint32, int(count))
	names := make([][cupti.StallReasonStringSize]byte, int(count))
	pointers := make([]uintptr, int(count))
	for i := range pointers {
		// An unwritten entry must not be mistaken for a valid empty name.
		for j := range names[i] {
			names[i][j] = 0xff
		}
		pin.Pin(&names[i][0])
		pointers[i] = uintptr(unsafe.Pointer(&names[i][0]))
	}
	pin.Pin(&indices[0])
	pin.Pin(&pointers[0])
	n := cupti.PCSamplingGetStallReasonsParams{Size: layout.NamesParamsSize, Ctx: ctx, NumStallReasons: count, StallReasonIndex: uintptr(unsafe.Pointer(&indices[0])), StallReasons: uintptr(unsafe.Pointer(&pointers[0]))}
	if err := namesQuery(&n); err != nil {
		return ReasonInventory{}, fmt.Errorf("cuptipcsampling: query reason names: %w", err)
	}
	if n.NumStallReasons != count || n.StallReasonIndex != uintptr(unsafe.Pointer(&indices[0])) || n.StallReasons != uintptr(unsafe.Pointer(&pointers[0])) {
		return ReasonInventory{}, errors.New("cuptipcsampling: native reason output changed storage")
	}
	seen := make(map[uint32]bool)
	seenNames := make(map[string]bool)
	for i, index := range indices {
		if pointers[i] != uintptr(unsafe.Pointer(&names[i][0])) {
			return ReasonInventory{}, errors.New("cuptipcsampling: native reason name changed storage")
		}
		end := bytes.IndexByte(names[i][:], 0)
		if end <= 0 {
			return ReasonInventory{}, errors.New("cuptipcsampling: empty or unterminated native reason")
		}
		name := string(names[i][:end])
		if seen[index] || seenNames[name] {
			return ReasonInventory{}, errors.New("cuptipcsampling: duplicate native reason")
		}
		seen[index], seenNames[name] = true, true
		out.Reasons = append(out.Reasons, StallReason{Index: index, Name: name})
	}
	sort.Slice(out.Reasons, func(i, j int) bool { return out.Reasons[i].Index < out.Reasons[j].Index })
	return out, nil
}
