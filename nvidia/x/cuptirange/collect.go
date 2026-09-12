package cuptirange

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

type collectionAPI struct {
	config func(*cupti.RangeProfilerSetConfigParams) error
	start  func(*cupti.RangeProfilerStartParams) error
	stop   func(*cupti.RangeProfilerStopParams) error
	decode func(*cupti.RangeProfilerDecodeDataParams) error
	info   func(*cupti.RangeProfilerGetCounterDataInfoParams) error
}

// collect runs automatic kernel ranges with caller-controlled user replay.
// The store must remain exclusively owned and its CUDA context current.
func (c *counterStore) collect(config []byte, maxRanges, maxPasses uint, run func(uint) error, api collectionAPI) (uint, error) {
	if c.host == 0 || len(c.image) == 0 || len(config) == 0 || maxRanges == 0 || maxPasses == 0 || run == nil {
		return 0, errors.New("cuptirange: invalid collection state or bounds")
	}
	if c.config != nil {
		return 0, errors.New("cuptirange: counter store already configured")
	}
	c.config = append([]byte(nil), config...)
	c.pin.Pin(&c.config[0])
	params := cupti.RangeProfilerSetConfigParams{
		PRangeProfilerObject: c.host, ConfigSize: uint(len(c.config)), PConfig: uintptr(unsafe.Pointer(&c.config[0])),
		CounterDataImageSize: uint(len(c.image)), PCounterDataImage: uintptr(unsafe.Pointer(&c.image[0])),
		Range: cupti.AutoRange, ReplayMode: cupti.UserReplay, MaxRangesPerPass: maxRanges,
		NumNestingLevels: 1, MinNestingLevel: 1, TargetNestingLevel: 1,
	}
	params.StructSize = uint(unsafe.Offsetof(params.TargetNestingLevel) + unsafe.Sizeof(params.TargetNestingLevel))
	if err := api.config(&params); err != nil {
		return 0, fmt.Errorf("cuptirange: configure collection: %w", err)
	}
	err := replayRuns(maxPasses, func() error {
		p := cupti.RangeProfilerStartParams{PRangeProfilerObject: c.host}
		p.StructSize = uint(unsafe.Offsetof(p.PRangeProfilerObject) + unsafe.Sizeof(p.PRangeProfilerObject))
		err := api.start(&p)
		if err == nil {
			c.quality.PassesStarted++
		}
		return err
	}, run, func() (cupti.RangeProfilerStopParams, error) {
		// An unwritten output is neither completion nor permission to replay.
		p := cupti.RangeProfilerStopParams{PRangeProfilerObject: c.host, IsAllPassSubmitted: 255}
		p.StructSize = uint(unsafe.Offsetof(p.IsAllPassSubmitted) + unsafe.Sizeof(p.IsAllPassSubmitted))
		err := api.stop(&p)
		if err == nil {
			c.quality.PassesStopped++
			if p.IsAllPassSubmitted == 1 {
				c.quality.AllPassesSubmitted = true
			}
		}
		return p, err
	})
	if err != nil {
		return 0, err
	}
	decode := cupti.RangeProfilerDecodeDataParams{PRangeProfilerObject: c.host}
	decode.StructSize = uint(unsafe.Offsetof(decode.NumOfRangeDropped) + unsafe.Sizeof(decode.NumOfRangeDropped))
	if err := api.decode(&decode); err != nil {
		return 0, fmt.Errorf("cuptirange: decode counters: %w", err)
	}
	c.quality.Decoded = true
	c.quality.DroppedRanges = uint64(decode.NumOfRangeDropped)
	if decode.NumOfRangeDropped != 0 {
		return 0, fmt.Errorf("cuptirange: dropped %d ranges", decode.NumOfRangeDropped)
	}
	info := cupti.RangeProfilerGetCounterDataInfoParams{PCounterDataImage: params.PCounterDataImage, CounterDataImageSize: params.CounterDataImageSize}
	info.StructSize = uint(unsafe.Offsetof(info.NumTotalRanges) + unsafe.Sizeof(info.NumTotalRanges))
	if err := api.info(&info); err != nil {
		return 0, fmt.Errorf("cuptirange: counter inventory: %w", err)
	}
	if info.NumTotalRanges == 0 || info.NumTotalRanges > maxRanges {
		return 0, fmt.Errorf("cuptirange: invalid collected range count %d", info.NumTotalRanges)
	}
	return info.NumTotalRanges, nil
}
