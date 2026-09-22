package cudacensus

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
)

func TestSamplerRejectsInvalidRequestsBeforeNativeCalls(t *testing.T) {
	var block nvml.GpmMetricsGetParams
	oversize := make([]Counter, len(block.Metrics)+1)
	for i := range oversize {
		oversize[i] = SMActive
	}
	for _, want := range [][]Counter{nil, {SMActive, SMActive}, {"unknown"}, oversize} {
		// A zero device must not reach any native query for an invalid list.
		if _, _, err := OpenSampler(0, want, StructVersions{1, 1}); err == nil {
			t.Fatalf("accepted %v", want)
		}
		if _, err := DiscoverSupport(0, want, StructVersions{1, 1}); err == nil {
			t.Fatalf("discovered invalid %v", want)
		}
		if _, err := ReadIdentity(0, want, StructVersions{1, 1}); err == nil {
			t.Fatalf("read identity for invalid %v", want)
		}
	}
}

func TestGPMDoesNotImplyPerMetricSupport(t *testing.T) {
	for _, gpm := range []bool{false, true} {
		s := unverifiedSupport(AllCounters(), gpm)
		if s.GPM != gpm {
			t.Fatal("lost device capability")
		}
		for _, c := range AllCounters() {
			if s.Has(c) || s.Unverified[c] == "" {
				t.Fatalf("invented support for %s: %+v", c, s)
			}
		}
	}
}

func TestSamplerUnavailableIntervalAndFreshResults(t *testing.T) {
	failure := errors.New("sample failed")
	fail := false
	write := true
	s := &gpmSampler{want: []Counter{SMActive}, prev: 1, next: 2, vers: StructVersions{1, 1},
		getSample: func(nvml.GpmSample, nvml.Device) error {
			if fail {
				return failure
			}
			return nil
		},
		getMetrics: func(p *nvml.GpmMetricsGetParams) error {
			if write {
				p.Metrics[0].NvmlReturn = nvml.Success
				p.Metrics[0].Value = 50
			}
			return nil
		},
	}
	dst := make([]Reading, 1)
	warm, err := s.Read(dst)
	if err != nil || warm.IntervalNs != 0 || dst[0].Available {
		t.Fatalf("warm=%+v err=%v", warm, err)
	}
	sample, err := s.Read(dst)
	if err != nil || sample.IntervalNs != -1 || !dst[0].Available || dst[0].Value != 0.5 {
		t.Fatalf("sample=%+v err=%v", sample, err)
	}
	write = false
	if _, err := s.Read(dst); err != nil || dst[0].Available || dst[0].Value != 0 {
		t.Fatalf("unwritten metric carried over: %v err=%v", dst, err)
	}
	write = true
	if _, err := s.Read(dst); err != nil {
		t.Fatal(err)
	}
	fail = true
	if _, err := s.Read(dst); !errors.Is(err, failure) || dst[0].Available || dst[0].Value != 0 {
		t.Fatalf("failed query retained values: %v err=%v", dst, err)
	}
	fail = false
	if allocs := testing.AllocsPerRun(100, func() {
		if _, err := s.Read(dst); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("steady read allocations=%g", allocs)
	}
	s.prev = 0
	if _, err := s.Read(dst); err == nil {
		t.Fatal("closed sampler read")
	}
}
