package main

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func pmTestAPI(t *testing.T) pmHostAPI {
	t.Helper()
	return pmHostAPI{
		version: func() (uint32, error) { return 130400, nil },
		initialize: func(p *cupti.ProfilerHostInitializeParams) error {
			if p.ProfilerType != cupti.ProfilerTypePmSampling {
				t.Fatal("constructed range host object for PM sampling")
			}
			if p.PChipName == 0 {
				t.Fatal("chip not pinned")
			}
			p.PHostObject = 1
			return nil
		},
		add: func(p *cupti.ProfilerHostConfigAddMetricsParams) error {
			if p.NumMetrics != 1 || p.PpMetricNames == 0 || p.PHostObject != 1 {
				t.Fatal("missing actual metric input")
			}
			return nil
		},
		size: func(p *cupti.ProfilerHostGetConfigImageSizeParams) error { p.ConfigImageSize = 16; return nil },
		image: func(p *cupti.ProfilerHostGetConfigImageParams) error {
			if p.ConfigImageSize != 16 || p.PConfigImage == 0 {
				t.Fatal("missing bounded native output")
			}
			return nil
		},
		passes: func(p *cupti.ProfilerHostGetNumOfPassesParams) error { p.NumOfPasses = 1; return nil },
		close: func(p *cupti.ProfilerHostDeinitializeParams) error {
			if p.PHostObject != 1 {
				t.Fatal("wrong cleanup handle")
			}
			return nil
		},
	}
}

func TestPMHostUsesDistinctProfilerTypeAndSinglePass(t *testing.T) {
	api := pmTestAPI(t)
	cfg, err := buildPMConfig("observed-chip", "requested-metric", 1024, api)
	if err != nil || len(cfg.ConfigImage) != 16 || len(cfg.MetricNames) != 1 || cfg.MetricNames[0] != "requested-metric" || cfg.SamplingInterval == 0 {
		t.Fatalf("PM config %+v %v", cfg, err)
	}
	for _, n := range []uint{0, 2} {
		api := pmTestAPI(t)
		api.passes = func(p *cupti.ProfilerHostGetNumOfPassesParams) error { p.NumOfPasses = n; return nil }
		if _, err := buildPMConfig("chip", "metric", 1024, api); err == nil {
			t.Fatalf("native passes%d admitted", n)
		}
	}
}

func TestPMHostBoundsAndUnknownABI(t *testing.T) {
	api := pmTestAPI(t)
	api.version = func() (uint32, error) { return 130401, nil }
	api.initialize = func(*cupti.ProfilerHostInitializeParams) error {
		t.Fatal("unknown ABI reached native initialization")
		return nil
	}
	if _, err := buildPMConfig("chip", "metric", 1024, api); err == nil {
		t.Fatal("unknown ABI admitted")
	}
	api = pmTestAPI(t)
	api.size = func(p *cupti.ProfilerHostGetConfigImageSizeParams) error { p.ConfigImageSize = 1025; return nil }
	api.image = func(*cupti.ProfilerHostGetConfigImageParams) error {
		t.Fatal("oversized allocation reached native fill")
		return nil
	}
	if _, err := buildPMConfig("chip", "metric", 1024, api); err == nil {
		t.Fatal("oversized config admitted")
	}
}

func TestPMHostRetainsStorageOnCleanupFailure(t *testing.T) {
	api := pmTestAPI(t)
	fail := true
	api.close = func(*cupti.ProfilerHostDeinitializeParams) error {
		if fail {
			return errors.New("pending native close")
		}
		return nil
	}
	before := len(pendingPMHosts)
	cfg, err := buildPMConfig("chip", "metric", 1024, api)
	if err == nil || len(cfg.ConfigImage) != 0 || len(pendingPMHosts) != before+1 {
		t.Fatal("failed cleanup became usable configuration")
	}
	retained := pendingPMHosts[before]
	if retained.handle != 1 || len(retained.storage) == 0 {
		t.Fatal("native cleanup ownership discarded")
	}
	fail = false
	if err := retained.close(); err != nil {
		t.Fatal(err)
	}
	pendingPMHosts = pendingPMHosts[:before]
}
