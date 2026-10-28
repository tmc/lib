package cuptipcsampling

import (
	"errors"
	"fmt"
	"runtime"
	"slices"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
)

type samplingOptions struct {
	period, workerSleepMS                       uint32
	mode                                        cupti.PCSamplingCollectionMode
	allowSerialization                          bool
	scratchBytes, hardwareBytes, maxNativeBytes uint64
	reasons                                     []uint32
}

// samplingConfig and its output buffer remain pinned until native teardown,
// including after a failed apply: configuration may have been partly accepted.
type samplingConfig struct {
	ctx                                    driver.Context
	output                                 *samplingBuffer
	attrs                                  []cupti.PCSamplingConfigurationInfo
	effective                              []cupti.PCSamplingConfigurationInfo
	reasons                                []uint32
	pin                                    runtime.Pinner
	attempted, applied, verified, released bool
}

func newSamplingConfig(o samplingOptions, inventory ReasonInventory, output *samplingBuffer) (*samplingConfig, error) {
	if !qualifiedSamplingVersion(inventory.CUPTIVersion) || (runtime.GOARCH != "amd64" && runtime.GOARCH != "arm64") {
		return nil, errors.New("cuptipcsampling: unqualified configuration layout")
	}
	if output == nil || output.released || len(o.reasons) == 0 || len(o.reasons) > len(output.reasons[0]) || o.period < 5 || o.period > 31 || o.workerSleepMS == 0 {
		return nil, errors.New("cuptipcsampling: invalid sampling configuration")
	}
	if o.mode != cupti.PcSamplingCollectionModeContinuous && (o.mode != cupti.PcSamplingCollectionModeKernelSerialized || !o.allowSerialization) {
		return nil, errors.New("cuptipcsampling: invalid or unauthorized sampling mode")
	}
	if o.scratchBytes == 0 || o.hardwareBytes == 0 || o.scratchBytes > o.maxNativeBytes || o.hardwareBytes > o.maxNativeBytes-o.scratchBytes {
		return nil, errors.New("cuptipcsampling: native sampling memory exceeds budget")
	}
	available := make(map[uint32]bool, len(inventory.Reasons))
	for _, r := range inventory.Reasons {
		available[r.Index] = true
	}
	selected := make(map[uint32]bool, len(o.reasons))
	for _, id := range o.reasons {
		if !available[id] || selected[id] {
			return nil, errors.New("cuptipcsampling: unknown or duplicate selected reason")
		}
		selected[id] = true
	}
	c := &samplingConfig{output: output, reasons: slices.Clone(o.reasons)}
	c.pin.Pin(&c.reasons[0])
	attr := func(kind cupti.PCSamplingConfigurationAttributeType, value uint64) cupti.PCSamplingConfigurationInfo {
		return cupti.PCSamplingConfigurationInfo{AttributeType: kind, AttributeStatus: cupti.Result(-1), AttributeData: [3]uint64{value}}
	}
	c.attrs = []cupti.PCSamplingConfigurationInfo{
		attr(cupti.PcSamplingConfigurationAttrTypeSamplingPeriod, uint64(o.period)),
		attr(cupti.PcSamplingConfigurationAttrTypeStallReason, uint64(len(c.reasons))),
		attr(cupti.PcSamplingConfigurationAttrTypeScratchBufferSize, o.scratchBytes),
		attr(cupti.PcSamplingConfigurationAttrTypeHardwareBufferSize, o.hardwareBytes),
		attr(cupti.PcSamplingConfigurationAttrTypeCollectionMode, uint64(o.mode)),
		attr(cupti.PcSamplingConfigurationAttrTypeEnableStartStopControl, 1),
		attr(cupti.PcSamplingConfigurationAttrTypeOutputDataFormat, 1),
		attr(cupti.PcSamplingConfigurationAttrTypeSamplingDataBuffer, uint64(uintptr(unsafe.Pointer(&output.data)))),
		attr(cupti.PcSamplingConfigurationAttrTypeWorkerThreadPeriodicSleepSpan, uint64(o.workerSleepMS)),
	}
	c.attrs[1].AttributeData[1] = uint64(uintptr(unsafe.Pointer(&c.reasons[0])))
	c.pin.Pin(&c.attrs[0])
	return c, nil
}

func (c *samplingConfig) apply(ctx driver.Context, query func(*cupti.PCSamplingConfigurationInfoParams) error) error {
	if c.released || c.attempted || ctx == 0 || query == nil {
		return errors.New("cuptipcsampling: invalid configuration application")
	}
	c.attempted = true
	c.ctx = ctx
	before := slices.Clone(c.attrs)
	reasons := slices.Clone(c.reasons)
	ptr := uintptr(unsafe.Pointer(&c.attrs[0]))
	size := configurationInfoParamsSize()
	p := cupti.PCSamplingConfigurationInfoParams{Size: size, Ctx: ctx, NumAttributes: uint(len(c.attrs)), PPCSamplingConfigurationInfo: ptr}
	if err := query(&p); err != nil {
		return fmt.Errorf("cuptipcsampling: configure sampling: %w", err)
	}
	if p.Size != size || p.PPriv != 0 || p.Ctx != ctx || p.NumAttributes != uint(len(c.attrs)) || p.PPCSamplingConfigurationInfo != ptr || !slices.Equal(reasons, c.reasons) {
		return errors.New("cuptipcsampling: native configuration changed input storage")
	}
	for i, a := range c.attrs {
		if a.AttributeType != before[i].AttributeType || a.AttributeData != before[i].AttributeData {
			return errors.New("cuptipcsampling: native configuration changed requested attribute")
		}
		if a.AttributeStatus != 0 {
			return fmt.Errorf("cuptipcsampling: configure attribute %s: %v", a.AttributeType, a.AttributeStatus)
		}
	}
	c.applied = true
	return nil
}

// contextConfigs holds, for each context, the configurations CUPTI may still
// reference. CUPTI keeps a context's PC sampling configuration after disable,
// including the address of the sampling data buffer: a later enable on the
// context runs with it, and the matching disable copies records into that
// buffer. A configuration is therefore released only after a later one has
// been applied in full to the same context.
var contextConfigs struct {
	sync.Mutex
	m map[driver.Context][]*samplingConfig
}

// retire gives up c and its output buffer once sampling on its context is
// disabled. A configuration CUPTI never saw is released at once; one it saw is
// held until it is superseded. The newest configuration of each context is
// therefore retained for the life of the process.
func (c *samplingConfig) retire() {
	if !c.attempted {
		c.output.release()
		c.release()
		return
	}
	contextConfigs.Lock()
	defer contextConfigs.Unlock()
	if contextConfigs.m == nil {
		contextConfigs.m = make(map[driver.Context][]*samplingConfig)
	}
	held := contextConfigs.m[c.ctx]
	if c.applied {
		for _, old := range held {
			old.output.release()
			old.release()
		}
		held = nil
	}
	contextConfigs.m[c.ctx] = append(held, c)
}

func (c *samplingConfig) release() {
	if !c.released {
		c.pin.Unpin()
		c.released = true
	}
}
