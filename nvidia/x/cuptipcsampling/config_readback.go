package cuptipcsampling

import (
	"errors"
	"fmt"
	"runtime"
	"slices"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// verify reads every read/write setting. Write-only reason selection and
// output buffers are checked by apply's per-attribute status, not fabricated
// as readable properties. Mismatched effective settings stop collection.
func (c *samplingConfig) verify(ctx cupti.Context, query func(*cupti.PCSamplingConfigurationInfoParams) error) error {
	if !c.applied || c.released || c.verified || ctx == 0 || ctx != c.ctx || query == nil {
		return errors.New("cuptipcsampling: invalid configuration readback")
	}
	indices := []int{0, 2, 3, 4, 5, 8}
	attrs := make([]cupti.PCSamplingConfigurationInfo, len(indices))
	for i, j := range indices {
		attrs[i] = cupti.PCSamplingConfigurationInfo{AttributeType: c.attrs[j].AttributeType, AttributeStatus: cupti.Result(-1), AttributeData: [3]uint64{^uint64(0), ^uint64(0), ^uint64(0)}}
	}
	var pin runtime.Pinner
	pin.Pin(&attrs[0])
	defer pin.Unpin()
	ptr := uintptr(unsafe.Pointer(&attrs[0]))
	p := cupti.PCSamplingConfigurationInfoParams{Size: 40, Ctx: ctx, NumAttributes: uint(len(attrs)), PPCSamplingConfigurationInfo: ptr}
	err := query(&p)
	c.effective = slices.Clone(attrs)
	if err != nil {
		return fmt.Errorf("cuptipcsampling: read sampling configuration: %w", err)
	}
	if p.Size != 40 || p.PPriv != 0 || p.Ctx != ctx || p.NumAttributes != uint(len(attrs)) || p.PPCSamplingConfigurationInfo != ptr {
		return errors.New("cuptipcsampling: native readback changed input storage")
	}
	for i, j := range indices {
		a, want := attrs[i], c.attrs[j]
		if a.AttributeType != want.AttributeType || a.AttributeStatus != 0 {
			return fmt.Errorf("cuptipcsampling: invalid readback for attribute %s", want.AttributeType)
		}
		got := a.AttributeData[0]
		if j != 2 && j != 3 {
			got = uint64(uint32(got))
		} // Only buffer sizes are size_t.
		if got != want.AttributeData[0] {
			return fmt.Errorf("cuptipcsampling: effective attribute %s is %d, requested %d", want.AttributeType, got, want.AttributeData[0])
		}
	}
	c.verified = true
	return nil
}
