package cudareg_test

import (
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudareg"
)

func registerResources() cudareg.Resources {
	return cudareg.Resources{
		ImageDigest: strings.Repeat("a", 64), Entry: "kernel", Target: "sm_90a", DecodedTarget: "sm_90",
		Allocated: cudareg.Fact{Count: 64, Status: cudareg.FactAvailable},
		PeakLive:  cudareg.Fact{Status: cudareg.FactUnavailable, Reason: "no matched life ranges"},
		Cap:       cudareg.Fact{Count: 64, Status: cudareg.FactAvailable},
		Modeled:   cudareg.Fact{Count: 12, Status: cudareg.FactAvailable},
	}
}

func TestResourcesValidate(t *testing.T) {
	for _, tt := range []struct {
		name string
		edit func(*cudareg.Resources)
	}{
		{"digest", func(r *cudareg.Resources) { r.ImageDigest = "abc" }},
		{"entry", func(r *cudareg.Resources) { r.Entry = "" }},
		{"entry_space", func(r *cudareg.Resources) { r.Entry = " kernel" }},
		{"entry_nul", func(r *cudareg.Resources) { r.Entry = "ker\x00nel" }},
		{"target", func(r *cudareg.Resources) { r.Target = "hopper" }},
		{"decoded_major", func(r *cudareg.Resources) { r.DecodedTarget = "sm_100" }},
		{"decoded_minor", func(r *cudareg.Resources) { r.DecodedTarget = "sm_91" }},
		{"decoded_suffix", func(r *cudareg.Resources) { r.DecodedTarget = "sm_90a" }},
		{"decoded_small", func(r *cudareg.Resources) { r.DecodedTarget = "sm_9" }},
		{"decoded_prefix", func(r *cudareg.Resources) { r.DecodedTarget = "compute_90" }},
		{"status", func(r *cudareg.Resources) { r.Allocated.Status = "invented" }},
		{"unavailable_count", func(r *cudareg.Resources) { r.PeakLive.Count = 1 }},
		{"unavailable_reason", func(r *cudareg.Resources) { r.PeakLive.Reason = "" }},
		{"unverified_reason", func(r *cudareg.Resources) { r.Modeled.Status = cudareg.FactUnverified }},
		{"zero_allocated", func(r *cudareg.Resources) { r.Allocated.Count = 0 }},
		{"zero_cap", func(r *cudareg.Resources) { r.Cap.Count = 0 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := registerResources()
			tt.edit(&r)
			if err := r.Validate(); err == nil {
				t.Fatal("invalid resources accepted")
			}
		})
	}
	r := registerResources()
	r.PeakLive = cudareg.Fact{Status: cudareg.FactAvailable}
	if err := r.Validate(); err != nil {
		t.Fatal("known zero peak:", err)
	}
	r.DecodedTarget = ""
	if err := r.Validate(); err != nil {
		t.Fatal("unavailable decoded target:", err)
	}
}

func TestPublishedResourceAvailability(t *testing.T) {
	for _, tt := range []struct {
		name string
		edit func(*cudareg.Resources)
		want cudareg.PublicationStatus
	}{
		{"available", func(*cudareg.Resources) {}, cudareg.ProtocolFeasible},
		{"unavailable", func(r *cudareg.Resources) {
			r.Allocated = cudareg.Fact{Status: cudareg.FactUnavailable, Reason: "absent"}
		}, cudareg.PublicationUnverified},
		{"unverified", func(r *cudareg.Resources) { r.Allocated.Status = cudareg.FactUnverified; r.Allocated.Reason = "caveat" }, cudareg.PublicationUnverified},
		{"decoded_unknown", func(r *cudareg.Resources) { r.DecodedTarget = "" }, cudareg.PublicationUnverified},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := registerResources()
			tt.edit(&r)
			s, _, _ := cudareg.GenerateFixture()
			c, err := cudareg.PublishCandidate(&r, s)
			if err != nil {
				t.Fatal(err)
			}
			if c.Status != tt.want || (c.Reason == "") != (tt.want == cudareg.ProtocolFeasible) {
				t.Fatalf("status=%s reason=%q", c.Status, c.Reason)
			}
			if !strings.Contains(c.TargetReason, "does not establish") {
				t.Fatal("suffix limitation lost")
			}
			if c.Schedule.Groups["producer"].PeakLive != 32 || c.Schedule.Groups["consumer"].Cap != 96 || c.Schedule.Groups["consumer"].Dynamic != 64 || c.Resources.Cap.Count != 64 {
				t.Fatal("global facts replaced role budgets")
			}
		})
	}
	s, _, _ := cudareg.GenerateFixture()
	c, err := cudareg.PublishCandidate(nil, s)
	if err != nil || c.Status != cudareg.PublicationUnverified || c.Resources != nil {
		t.Fatalf("nil publication: %+v %v", c, err)
	}
}

func TestPublishedRefusals(t *testing.T) {
	for _, tt := range []struct {
		name string
		edit func(*cudareg.Resources)
	}{
		{"target", func(r *cudareg.Resources) { r.Target = "sm_100a"; r.DecodedTarget = "sm_100" }},
		{"allocation", func(r *cudareg.Resources) { r.Allocated.Count = 72 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := registerResources()
			tt.edit(&r)
			before := r
			s, _, _ := cudareg.GenerateFixture()
			c, err := cudareg.PublishCandidate(&r, s)
			if err == nil || !reflect.DeepEqual(c, cudareg.Candidate{}) || r != before {
				t.Fatalf("nontransactional refusal: %+v %v", c, err)
			}
		})
	}
}

func TestPublishedScheduleAdmission(t *testing.T) {
	for _, name := range []string{"pool", "participants", "direction", "progress", "initial"} {
		t.Run(name, func(t *testing.T) {
			s, _, _ := cudareg.GenerateFixture()
			switch name {
			case "pool":
				s.CTARegisterPool++
			case "participants":
				s.Steps[0].Participants = 32
			case "direction":
				s.Steps[0].Action = cudareg.Increase
			case "progress":
				s.Steps[0], s.Steps[1] = s.Steps[1], s.Steps[0]
			case "initial":
				b := s.Groups["consumer"]
				b.Dynamic = 32
				s.Groups["consumer"] = b
			}
			r := registerResources()
			c, err := cudareg.PublishCandidate(&r, s)
			if err == nil || !reflect.DeepEqual(c, cudareg.Candidate{}) {
				t.Fatalf("bad schedule accepted: %+v %v", c, err)
			}
		})
	}
}

func TestPublishedTargetProvisional(t *testing.T) {
	s, _, _ := cudareg.GenerateFixture()
	s.Target, _ = cudareg.TargetFor("sm_999a")
	r := registerResources()
	r.Target = "sm_999a"
	r.DecodedTarget = "sm_999"
	c, err := cudareg.PublishCandidate(&r, s)
	if err != nil || c.Status != cudareg.ProtocolFeasible || c.TargetEvidence.Measured {
		t.Fatalf("lost provisional target: %+v %v", c, err)
	}
	s.Target.Name = "sm_90"
	r.Target = "sm_90"
	r.DecodedTarget = "sm_90"
	if _, err := cudareg.PublishCandidate(&r, s); err == nil {
		t.Fatal("recorded target rejection accepted")
	}
}

func TestPublishedDetached(t *testing.T) {
	s, _, _ := cudareg.GenerateFixture()
	r := registerResources()
	c, err := cudareg.PublishCandidate(&r, s)
	if err != nil {
		t.Fatal(err)
	}
	c.Schedule.Target.LegalCounts[0] = 1
	delete(c.Schedule.Groups, "producer")
	c.Schedule.Steps[0].Count = 24
	c.Resources.Allocated.Count = 8
	if s.Target.LegalCounts[0] != 24 || len(s.Groups) != 2 || s.Steps[0].Count != 32 || r.Allocated.Count != 64 {
		t.Fatal("publication aliases inputs")
	}
	d, err := cudareg.PublishCandidate(&r, s)
	if err != nil || d.Resources.Allocated.Count != 64 || len(d.Schedule.Groups) != 2 {
		t.Fatal("publication aliases another candidate")
	}
}

func TestPublishedConcurrent(t *testing.T) {
	s, _, _ := cudareg.GenerateFixture()
	r := registerResources()
	var wg sync.WaitGroup
	for i := 0; i < 24; i++ {
		wg.Go(func() {
			c, err := cudareg.PublishCandidate(&r, s)
			if err != nil || c.Status != cudareg.ProtocolFeasible {
				t.Errorf("publication: %v", err)
			}
			if c.Resources != nil {
				c.Resources.Allocated.Count = 1
			}
			delete(c.Schedule.Groups, "producer")
		})
	}
	wg.Wait()
	if driver.Loaded() || nvrtc.Loaded() {
		t.Fatal("host publication loaded CUDA libraries")
	}
}
