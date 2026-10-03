package main

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
)

// This fixture uses the measured CUDA 13.4 host initialization layout only.
// NVIDIA's PM host API requires a PM profiler object, not a range config image.
type pmHostAPI struct {
	version    func() (uint32, error)
	initialize func(*cupti.ProfilerHostInitializeParams) error
	add        func(*cupti.ProfilerHostConfigAddMetricsParams) error
	size       func(*cupti.ProfilerHostGetConfigImageSizeParams) error
	image      func(*cupti.ProfilerHostGetConfigImageParams) error
	passes     func(*cupti.ProfilerHostGetNumOfPassesParams) error
	close      func(*cupti.ProfilerHostDeinitializeParams) error
}

type pmHost struct {
	handle   uintptr
	pin      runtime.Pinner
	storage  [][]byte
	pointers []uintptr
	api      pmHostAPI
}

// A failed native close retains pinned storage until the fixture process exits.
// It is reported as preparation failure, never a usable configuration.
var pendingPMHosts []*pmHost

func (h *pmHost) close() error {
	if h.handle != 0 {
		p := cupti.ProfilerHostDeinitializeParams{PHostObject: h.handle}
		p.StructSize = uint(unsafe.Offsetof(p.PHostObject) + unsafe.Sizeof(p.PHostObject))
		if e := h.api.close(&p); e != nil {
			return e
		}
		h.handle = 0
	}
	h.pin.Unpin()
	return nil
}

func actualPMConfig(chip, metric string) (cuptipmsampling.Config, error) {
	return buildPMConfig(chip, metric, 4<<20, pmHostAPI{cupti.GetVersion, cupti.ProfilerHostInitialize, cupti.ProfilerHostConfigAddMetrics, cupti.ProfilerHostGetConfigImageSize, cupti.ProfilerHostGetConfigImage, cupti.ProfilerHostGetNumOfPasses, cupti.ProfilerHostDeinitialize})
}

func buildPMConfig(chip, metric string, maxBytes int, api pmHostAPI) (cfg cuptipmsampling.Config, err error) {
	if chip == "" || len(chip) > 256 || strings.ContainsRune(chip, 0) || metric == "" || len(metric) > 256 || strings.ContainsRune(metric, 0) || maxBytes <= 0 || maxBytes > 4<<20 {
		return cfg, errors.New("PM host: invalid bounded configuration input")
	}
	version, err := api.version()
	if err != nil {
		return cfg, err
	}
	if version != 130400 || unsafe.Sizeof(uintptr(0)) != 8 {
		return cfg, fmt.Errorf("PM host: unqualified initialization layout for CUPTI %d", version)
	}
	h := &pmHost{api: api, storage: [][]byte{append([]byte(chip), 0), append([]byte(metric), 0)}}
	for _, b := range h.storage {
		h.pin.Pin(&b[0])
	}
	defer func() {
		if e := h.close(); e != nil {
			pendingPMHosts = append(pendingPMHosts, h)
			err = errors.Join(err, fmt.Errorf("PM host close: %w", e))
		}
		if err != nil {
			cfg = cuptipmsampling.Config{}
		}
	}()
	p := cupti.ProfilerHostInitializeParams{ProfilerType: cupti.ProfilerTypePmSampling, PChipName: uintptr(unsafe.Pointer(&h.storage[0][0]))}
	p.StructSize = uint(unsafe.Offsetof(p.PSinglePassMetricSetName) + unsafe.Sizeof(p.PSinglePassMetricSetName))
	err = api.initialize(&p)
	h.handle = p.PHostObject
	if err != nil {
		return cfg, err
	}
	if h.handle == 0 {
		return cfg, errors.New("PM host: empty native object")
	}
	h.pointers = []uintptr{uintptr(unsafe.Pointer(&h.storage[1][0]))}
	h.pin.Pin(&h.pointers[0])
	add := cupti.ProfilerHostConfigAddMetricsParams{PHostObject: h.handle, PpMetricNames: uintptr(unsafe.Pointer(&h.pointers[0])), NumMetrics: 1}
	add.StructSize = uint(unsafe.Offsetof(add.NumMetrics) + unsafe.Sizeof(add.NumMetrics))
	if err := api.add(&add); err != nil {
		return cfg, err
	}
	size := cupti.ProfilerHostGetConfigImageSizeParams{PHostObject: h.handle}
	size.StructSize = uint(unsafe.Offsetof(size.ConfigImageSize) + unsafe.Sizeof(size.ConfigImageSize))
	if err := api.size(&size); err != nil {
		return cfg, err
	}
	if size.ConfigImageSize == 0 || size.ConfigImageSize > uint(maxBytes) {
		return cfg, errors.New("PM host: native configuration exceeds byte limit")
	}
	image := make([]byte, size.ConfigImageSize)
	h.storage = append(h.storage, image)
	h.pin.Pin(&image[0])
	get := cupti.ProfilerHostGetConfigImageParams{PHostObject: h.handle, ConfigImageSize: size.ConfigImageSize, PConfigImage: uintptr(unsafe.Pointer(&image[0]))}
	get.StructSize = uint(unsafe.Offsetof(get.PConfigImage) + unsafe.Sizeof(get.PConfigImage))
	if err := api.image(&get); err != nil {
		return cfg, err
	}
	passes := cupti.ProfilerHostGetNumOfPassesParams{ConfigImageSize: size.ConfigImageSize, PConfigImage: get.PConfigImage}
	passes.StructSize = uint(unsafe.Offsetof(passes.NumOfPasses) + unsafe.Sizeof(passes.NumOfPasses))
	if err := api.passes(&passes); err != nil {
		return cfg, err
	}
	if passes.NumOfPasses != 1 {
		return cfg, fmt.Errorf("PM host: periodic sampling requires one native configuration pass, got %d", passes.NumOfPasses)
	}
	return cuptipmsampling.Config{DeviceIndex: 0, ConfigImage: append([]byte(nil), image...), HardwareBufferSize: 1 << 20, SamplingInterval: 1000, TriggerMode: cuptipmsampling.TriggerModeTimeInterval, AppendMode: cuptipmsampling.AppendModeKeepOldest, MaxSamples: 128, MetricNames: []string{metric}}, nil
}
