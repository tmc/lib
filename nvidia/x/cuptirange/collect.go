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
	return c.collectReplay(config, maxRanges, maxPasses, 1, cupti.AutoRange, nil, run, api)
}

func (c *counterStore) collectReplay(config []byte, maxRanges, maxPasses uint, depth uint16, mode cupti.ProfilerRange, prepare func(uint) error, run func(uint) error, api collectionAPI) (uint, error) {
	return c.collectReplayMode(config, maxRanges, maxPasses, depth, mode, cupti.UserReplay, prepare, run, api)
}

func (c *counterStore) collectReplayMode(config []byte, maxRanges, maxPasses uint, depth uint16, mode cupti.ProfilerRange, replayMode cupti.ProfilerReplayMode, prepare func(uint) error, run func(uint) error, api collectionAPI) (uint, error) {
	if replayMode != cupti.UserReplay && replayMode != cupti.KernelReplay {
		return 0, errors.New("cuptirange: invalid replay mode")
	}
	if replayMode == cupti.KernelReplay {
		if mode != cupti.AutoRange || depth != 1 {
			return 0, errors.New("cuptirange: kernel replay requires automatic ranges at depth one")
		}
		if maxPasses > 0 {
			maxPasses = 1
		}
	}

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
		Range: mode, ReplayMode: replayMode, MaxRangesPerPass: maxRanges,
		NumNestingLevels: depth, MinNestingLevel: 1, TargetNestingLevel: 1,
	}
	params.StructSize = uint(unsafe.Offsetof(params.TargetNestingLevel) + unsafe.Sizeof(params.TargetNestingLevel))
	if err := api.config(&params); err != nil {
		return 0, fmt.Errorf("cuptirange: configure collection: %w", err)
	}
	err := replayPrepared(maxPasses, prepare, func() error {
		p := cupti.RangeProfilerStartParams{PRangeProfilerObject: c.host}
		p.StructSize = uint(unsafe.Offsetof(p.PRangeProfilerObject) + unsafe.Sizeof(p.PRangeProfilerObject))
		if err := api.start(&p); err != nil {
			return fmt.Errorf("cuptirange: native RangeProfilerStart: %w", err)
		}
		c.quality.PassesStarted++
		return nil
	}, run, func() (cupti.RangeProfilerStopParams, error) {
		// An unwritten output is neither completion nor permission to replay.
		p := cupti.RangeProfilerStopParams{PRangeProfilerObject: c.host, IsAllPassSubmitted: 255}
		p.StructSize = uint(unsafe.Offsetof(p.IsAllPassSubmitted) + unsafe.Sizeof(p.IsAllPassSubmitted))
		if err := api.stop(&p); err != nil {
			return p, fmt.Errorf("cuptirange: native RangeProfilerStop: %w", err)
		}
		c.quality.PassesStopped++
		if p.IsAllPassSubmitted == 1 {
			c.quality.AllPassesSubmitted = true
		}
		return p, nil
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
	// MaxRangesPerPass may select only the first N automatic ranges without
	// reporting decode loss (witnessed on CUPTI 130202). A full inventory at the
	// configured capacity needs an independent launch-count witness. User
	// ranges have an independent Scope trace checked against native identities.
	if mode == cupti.AutoRange {
		if c.kernelNames != nil {
			if err := c.kernelNames.checkCount(info.NumTotalRanges); err != nil {
				return 0, err
			}
		} else if info.NumTotalRanges == maxRanges {
			return 0, fmt.Errorf("cuptirange: automatic range inventory reached capacity %d; completeness unproven", maxRanges)
		}
	}
	return info.NumTotalRanges, nil
}
