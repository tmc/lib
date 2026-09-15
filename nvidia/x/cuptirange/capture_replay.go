package cuptirange

import (
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func (c *counterStore) captureReplay(config []byte, limits Limits, options CaptureOptions, replay Replay, identities bool) (uint, []RangeInfo, error) {
	return c.captureReplayWithAPI(config, limits, options, replay, identities, collectionAPI{cupti.RangeProfilerSetConfig, cupti.RangeProfilerStart, cupti.RangeProfilerStop, cupti.RangeProfilerDecodeData, cupti.RangeProfilerGetCounterDataInfo}, c.pushRange, c.popRange, cupti.RangeProfilerCounterDataGetRangeInfo)
}

func (c *counterStore) captureReplayWithAPI(config []byte, limits Limits, options CaptureOptions, replay Replay, identities bool, api collectionAPI, push func(string) error, pop func() error, info func(*cupti.RangeProfilerCounterDataGetRangeInfoParams) error) (uint, []RangeInfo, error) {
	mode := cupti.UserRange
	if options.Mode == Kernels {
		mode = cupti.AutoRange
	}
	var expected []RangeInfo
	prepare := func(pass uint) (err error) {
		if err := replay.Synchronize(); err != nil {
			return err
		}
		defer func() { err = errors.Join(err, replay.Synchronize()) }()
		if replay.Prepare != nil {
			return replay.Prepare(pass)
		}
		return nil
	}

	run := func(pass uint) (err error) {
		s := &Scope{active: true, mode: options.Mode, depth: options.MaxDepth, limit: limits.MaxRanges, counts: make(map[string]int), push: push, pop: pop}
		defer func() { s.active = false }()
		work := func() (err error) {
			if c.kernelNames != nil {
				if err := c.kernelNames.begin(); err != nil {
					return err
				}
				defer func() { err = errors.Join(err, c.kernelNames.end()) }()
			}
			defer func() { err = errors.Join(err, replay.Synchronize()) }()
			return replay.Run(pass, s)
		}
		if options.Mode == Kernels {
			err = work()
		} else {
			err = s.region("operation", "operation", work)
		}
		err = errors.Join(err, s.failure)
		if err != nil {
			return err
		}
		if pass == 0 {
			expected = append([]RangeInfo(nil), s.ranges...)
		} else if !reflect.DeepEqual(expected, s.ranges) {
			return errors.New("cuptirange: region replay trace changed")
		}
		return nil
	}
	nativeReplay := cupti.UserReplay
	if options.ReplayMode == KernelReplay {
		nativeReplay = cupti.KernelReplay
	}
	count, err := c.collectReplayMode(config, limits.MaxRanges, limits.MaxPasses, options.MaxDepth, mode, nativeReplay, prepare, run, api)
	if err != nil || !identities {
		return count, nil, err
	}
	records, err := c.rangeInfo(count, options.Mode, expected, info)
	return count, records, err
}

func (c *counterStore) pushRange(name string) error {
	if c.rangeNames == nil {
		c.rangeNames = make(map[string][]byte)
	}
	data := c.rangeNames[name]
	if data == nil {
		data = append([]byte(name), 0)
		c.pin.Pin(&data[0])
		c.rangeNames[name] = data
	}
	p := cupti.RangeProfilerPushRangeParams{PRangeProfilerObject: c.host, PRangeName: uintptr(unsafe.Pointer(&data[0]))}
	p.StructSize = uint(unsafe.Offsetof(p.PRangeName) + unsafe.Sizeof(p.PRangeName))
	err := cupti.RangeProfilerPushRange(&p)
	runtime.KeepAlive(data)
	return err
}
func (c *counterStore) popRange() error {
	p := cupti.RangeProfilerPopRangeParams{PRangeProfilerObject: c.host}
	p.StructSize = uint(unsafe.Offsetof(p.PRangeProfilerObject) + unsafe.Sizeof(p.PRangeProfilerObject))
	return cupti.RangeProfilerPopRange(&p)
}

func (c *counterStore) rangeInfo(count uint, mode Mode, expected []RangeInfo, api func(*cupti.RangeProfilerCounterDataGetRangeInfoParams) error) ([]RangeInfo, error) {
	delimiter := []byte{'/', 0}
	var pin runtime.Pinner
	pin.Pin(&delimiter[0])
	defer pin.Unpin()
	records := make([]RangeInfo, count)
	byID := make(map[string]RangeInfo, len(expected))
	for _, r := range expected {
		byID[r.ID] = r
	}
	seen := make(map[string]bool)
	for i := range records {
		p := cupti.RangeProfilerCounterDataGetRangeInfoParams{PCounterDataImage: uintptr(unsafe.Pointer(&c.image[0])), CounterDataImageSize: uint(len(c.image)), RangeIndex: uint(i), RangeDelimiter: uintptr(unsafe.Pointer(&delimiter[0]))}
		p.StructSize = uint(unsafe.Offsetof(p.RangeName) + unsafe.Sizeof(p.RangeName))
		if err := api(&p); err != nil {
			return nil, fmt.Errorf("cuptirange: range %d identity: %w", i, err)
		}
		// The generated binding represents native char* as uintptr. Read its
		// pointer-sized ABI slot through a typed pointer view; do not reconstruct
		// a Go pointer from integer arithmetic.
		name, err := rangeName(*(**byte)(unsafe.Pointer(&p.RangeName)))
		if err != nil {
			return nil, err
		}
		if mode == Kernels {
			records[i] = RangeInfo{Index: uint(i), ID: fmt.Sprintf("kernel#%d", i+1), Name: name, Kind: "kernel"}
			continue
		}
		r, ok := byID[name]
		if !ok || seen[name] {
			return nil, fmt.Errorf("cuptirange: unexpected native range %q", name)
		}
		seen[name] = true
		r.Index = uint(i)
		records[i] = r
	}
	if mode != Kernels && len(seen) != len(expected) {
		return nil, errors.New("cuptirange: missing native user ranges")
	}
	if mode == Kernels && c.kernelNames != nil {
		if err := c.kernelNames.resolve(records); err != nil {
			return nil, err
		}
	}
	return records, nil
}

// rangeName copies a native-owned NUL-terminated name before the next API call.
// The limit bounds malformed output, not the lifetime of native storage.
func rangeName(p *byte) (string, error) {
	if p == nil {
		return "", errors.New("cuptirange: missing native range name")
	}
	b := make([]byte, 0, 128)
	for i := 0; i < 1<<20; i++ {
		v := *(*byte)(unsafe.Add(unsafe.Pointer(p), i))
		if v == 0 {
			if len(b) == 0 {
				return "", errors.New("cuptirange: empty native range name")
			}
			return string(b), nil
		}
		b = append(b, v)
	}
	return "", errors.New("cuptirange: native range name exceeds limit")
}
