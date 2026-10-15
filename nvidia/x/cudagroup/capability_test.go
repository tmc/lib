package cudagroup_test

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudagroup"
)

func groupCapabilityRecord(t *testing.T) cudadevicetable.Architecture {
	t.Helper()
	record, ok := cudadevicetable.Lookup("sm_121")
	if !ok {
		t.Fatal("missing recorded sm_121 device")
	}
	attrs := make(map[string]cudadevicetable.Value, len(record.Attributes))
	for name, v := range record.Attributes {
		attrs[name] = v
	}
	record.Attributes = attrs
	if record.Probe != nil {
		probe := *record.Probe
		record.Probe = &probe
	}
	return record
}

func capabilityWarp(t *testing.T) cudagroup.Group {
	t.Helper()
	warp, err := cudagroup.New(cudagroup.Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	if err != nil {
		t.Fatal(err)
	}
	return warp
}

func TestCapabilityRecordedFixture(t *testing.T) {
	warp := capabilityWarp(t)
	for _, target := range []string{"sm_121", "sm_121a"} {
		t.Run(target, func(t *testing.T) {
			record := groupCapabilityRecord(t)
			before := groupCapabilityRecord(t)
			f, err := cudagroup.GenerateReductionFixtureWithCapability(warp, target, record)
			if err != nil {
				t.Fatal(err)
			}
			original, err := cudagroup.GenerateReductionFixtureForTarget(warp, target)
			if err != nil || !reflect.DeepEqual(f, original) {
				t.Fatal("capability consumer changed the actual generated fixture", err)
			}
			if err := f.ValidateCapability(record); err != nil {
				t.Fatal(err)
			}
			if f.Target != target || !strings.Contains(f.Source, "warp_reduce_sum<float>") || !strings.Contains(f.Source, "__shfl_down_sync") || !strings.Contains(f.Source, "target="+target) {
				t.Fatal("not the target's actual warp reduction source")
			}
			if f.Launch.Block != (driver.Dim3{X: 32, Y: 1, Z: 1}) || f.Launch.Grid != (driver.Dim3{X: 1, Y: 1, Z: 1}) || f.Launch.Cluster != (driver.Dim3{}) {
				t.Fatal("changed generated launch", f.Launch)
			}
			if !reflect.DeepEqual(record, before) {
				t.Fatal("admission mutated the capability record")
			}
		})
	}
}

func TestCapabilityRecordRefusals(t *testing.T) {
	warp := capabilityWarp(t)
	for _, tt := range []struct {
		name   string
		change func(*cudadevicetable.Architecture)
		want   error
	}{
		{"absent", func(r *cudadevicetable.Architecture) { *r = cudadevicetable.Architecture{} }, cudagroup.ErrCapabilityAbsent},
		{"entire record unverified", func(r *cudadevicetable.Architecture) {
			r.Probe = nil
			r.Attributes = map[string]cudadevicetable.Value{
				"warpSize":           {Value: 32, Source: cudadevicetable.Unverified, Note: "not measured"},
				"maxThreadsPerBlock": {Value: 1024, Source: cudadevicetable.Unverified, Note: "not measured"},
			}
		}, cudagroup.ErrCapabilityUnverified},
		{"other architecture", func(r *cudadevicetable.Architecture) { r.Architecture = "sm_100" }, cudagroup.ErrCapabilityInvalid},
		{"suffix is not an architecture", func(r *cudadevicetable.Architecture) { r.Architecture = "sm_121a" }, cudagroup.ErrCapabilityInvalid},
		{"other compute capability", func(r *cudadevicetable.Architecture) { r.Probe.ComputeCapability = "9.0" }, cudagroup.ErrCapabilityInvalid},
		{"noncanonical compute capability", func(r *cudadevicetable.Architecture) { r.Probe.ComputeCapability = "12.01" }, cudagroup.ErrCapabilityInvalid},
		{"no probe", func(r *cudadevicetable.Architecture) { r.Probe = nil }, cudagroup.ErrCapabilityInvalid},
		{"no host", func(r *cudadevicetable.Architecture) { r.Probe.Host = "" }, cudagroup.ErrCapabilityInvalid},
		{"no tool", func(r *cudadevicetable.Architecture) { r.Probe.Tool = "" }, cudagroup.ErrCapabilityInvalid},
		{"no date", func(r *cudadevicetable.Architecture) { r.Probe.ProbedAt = "" }, cudagroup.ErrCapabilityInvalid},
		{"no attributes", func(r *cudadevicetable.Architecture) { r.Attributes = nil }, cudagroup.ErrCapabilityInvalid},
	} {
		t.Run(tt.name, func(t *testing.T) {
			record := groupCapabilityRecord(t)
			tt.change(&record)
			assertCapabilityRefusal(t, warp, record, tt.want)
		})
	}
	for _, name := range []string{"warpSize", "maxThreadsPerBlock"} {
		for _, tt := range []struct {
			name   string
			change func(cudadevicetable.Value) cudadevicetable.Value
			want   error
		}{
			{"unverified", func(v cudadevicetable.Value) cudadevicetable.Value {
				v.Source, v.Instrument, v.Note = cudadevicetable.Unverified, "", "not measured"
				return v
			}, cudagroup.ErrCapabilityUnverified},
			{"derived", func(v cudadevicetable.Value) cudadevicetable.Value {
				v.Source, v.Instrument, v.Note = cudadevicetable.Derived, "", "test derivation, not a direct probe"
				return v
			}, cudagroup.ErrCapabilityUnverified},
			{"unverified placeholder", func(v cudadevicetable.Value) cudadevicetable.Value {
				v.Value, v.Source, v.Instrument, v.Note = 0, cudadevicetable.Unverified, "", "not measured"
				return v
			}, cudagroup.ErrCapabilityUnverified},
			{"invalid source", func(v cudadevicetable.Value) cudadevicetable.Value { v.Source = ""; return v }, cudagroup.ErrCapabilityInvalid},
			{"no instrument", func(v cudadevicetable.Value) cudadevicetable.Value { v.Instrument = ""; return v }, cudagroup.ErrCapabilityInvalid},
			{"not a device instrument", func(v cudadevicetable.Value) cudadevicetable.Value { v.Instrument = "ptxas -arch=sm_121a"; return v }, cudagroup.ErrCapabilityInvalid},
			{"wrong device attribute", func(v cudadevicetable.Value) cudadevicetable.Value { v.Instrument = "DevAttrClockRate"; return v }, cudagroup.ErrCapabilityInvalid},
			{"unverified with instrument", func(v cudadevicetable.Value) cudadevicetable.Value {
				v.Source, v.Note = cudadevicetable.Unverified, "not measured"
				return v
			}, cudagroup.ErrCapabilityInvalid},
			{"zero", func(v cudadevicetable.Value) cudadevicetable.Value { v.Value = 0; return v }, cudagroup.ErrCapabilityInvalid},
			{"negative", func(v cudadevicetable.Value) cudadevicetable.Value { v.Value = -1; return v }, cudagroup.ErrCapabilityInvalid},
		} {
			t.Run(name+"/"+tt.name, func(t *testing.T) {
				record := groupCapabilityRecord(t)
				record.Attributes[name] = tt.change(record.Attributes[name])
				assertCapabilityRefusal(t, warp, record, tt.want)
			})
		}
		t.Run(name+"/missing", func(t *testing.T) {
			record := groupCapabilityRecord(t)
			delete(record.Attributes, name)
			assertCapabilityRefusal(t, warp, record, cudagroup.ErrCapabilityAbsent)
		})
	}
}

func assertCapabilityRefusal(t *testing.T, warp cudagroup.Group, record cudadevicetable.Architecture, want error) {
	t.Helper()
	// Generation must actually consume the record, not return unchecked source
	// while leaving validation as an unused helper.
	f, err := cudagroup.GenerateReductionFixtureWithCapability(warp, "sm_121a", record)
	if !errors.Is(err, want) || !reflect.DeepEqual(f, cudagroup.Fixture{}) {
		t.Fatalf("generation returned fixture %v, error %v; want empty fixture and %v", f, err, want)
	}
	original, err := cudagroup.GenerateReductionFixtureForTarget(warp, "sm_121a")
	if err != nil {
		t.Fatal(err)
	}
	if err := original.ValidateCapability(record); !errors.Is(err, want) {
		t.Fatalf("direct admission error %v, want %v", err, want)
	}
}

func TestCapabilityMeasuredLimits(t *testing.T) {
	// All changed values below are controlled host mutations, not device facts.
	warp := capabilityWarp(t)
	for _, tt := range []struct {
		name  string
		warp  int64
		limit int64
		want  error
	}{
		{"exact block threshold", 32, 32, nil},
		{"below block threshold", 32, 31, cudagroup.ErrCapabilityUnsupported},
		{"large positive limit", 32, math.MaxInt64, nil},
		{"smaller warp", 16, 1024, cudagroup.ErrCapabilityUnsupported},
		{"larger warp", 64, 1024, cudagroup.ErrCapabilityUnsupported},
		{"large warp", math.MaxInt64, 1024, cudagroup.ErrCapabilityUnsupported},
	} {
		t.Run(tt.name, func(t *testing.T) {
			record := groupCapabilityRecord(t)
			v := record.Attributes["warpSize"]
			v.Value = tt.warp
			record.Attributes["warpSize"] = v
			v = record.Attributes["maxThreadsPerBlock"]
			v.Value = tt.limit
			record.Attributes["maxThreadsPerBlock"] = v
			f, err := cudagroup.GenerateReductionFixtureWithCapability(warp, "sm_121a", record)
			if tt.want != nil {
				if !errors.Is(err, tt.want) || !reflect.DeepEqual(f, cudagroup.Fixture{}) {
					t.Fatal("did not refuse measured incompatible limits", err)
				}
			} else if err != nil || f.Source == "" {
				t.Fatal("rejected supported declared limits", err)
			}
		})
	}
}

func TestCapabilityAssemblerIsNotDeviceEvidence(t *testing.T) {
	warp := capabilityWarp(t)
	feature, ok := cudadevicetable.LookupTarget("sm_100a")
	if !ok {
		t.Fatal("missing assembler feature control")
	}
	if supported, err := feature.Supports("tcgen05"); err != nil || !supported {
		t.Fatal("assembler positive control changed", supported, err)
	}
	record, ok := cudadevicetable.Lookup("sm_100a")
	if ok {
		t.Fatal("source-pinned absence control changed; refresh the fixture")
	}
	f, err := cudagroup.GenerateReductionFixtureForTarget(warp, "sm_100a")
	if err != nil || f.Source == "" || f.ValidateLaunch() != nil {
		t.Fatal("legacy source-only compatibility changed", err)
	}
	if err := f.ValidateCapability(record); !errors.Is(err, cudagroup.ErrCapabilityAbsent) || errors.Is(err, cudagroup.ErrCapabilityUnsupported) {
		t.Fatal("assembler acceptance or missing device record became device support", err)
	}
	f, err = cudagroup.GenerateReductionFixtureWithCapability(warp, "sm_100a", record)
	if !errors.Is(err, cudagroup.ErrCapabilityAbsent) || f.Source != "" {
		t.Fatal("returned capability-admitted source without device data", err)
	}
	legacy, err := cudagroup.GenerateReductionFixture(warp)
	if err != nil || legacy.Target != "sm_90" || legacy.Source == "" {
		t.Fatal("legacy default source-only compatibility changed", err)
	}
	if err := legacy.ValidateCapability(groupCapabilityRecord(t)); !errors.Is(err, cudagroup.ErrCapabilityInvalid) {
		t.Fatal("default target borrowed a different architecture's limits", err)
	}
}

func TestCapabilityLaunchAdmission(t *testing.T) {
	warp := capabilityWarp(t)
	record := groupCapabilityRecord(t)
	for _, tt := range []struct {
		name   string
		change func(*cudagroup.Fixture)
	}{
		{"block shape", func(f *cudagroup.Fixture) { f.Launch.Block.X = 64 }},
		{"grid shape", func(f *cudagroup.Fixture) { f.Launch.Grid.X = 2 }},
		{"cluster", func(f *cudagroup.Fixture) { f.Launch.Cluster = driver.Dim3{X: 1, Y: 1, Z: 1} }},
		{"unknown target", func(f *cudagroup.Fixture) { f.Target = "sm_999a" }},
		{"wrong group", func(f *cudagroup.Fixture) { f.Group = cudagroup.Group{} }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			f, err := cudagroup.GenerateReductionFixtureWithCapability(warp, "sm_121a", record)
			if err != nil {
				t.Fatal(err)
			}
			tt.change(&f)
			if err := f.ValidateCapability(record); err == nil {
				t.Fatal("capability record bypassed actual fixture launch validation")
			}
		})
	}
	for _, target := range []string{"sm_999a", "SM_121a", "sm_121a\n"} {
		f, err := cudagroup.GenerateReductionFixtureWithCapability(warp, target, record)
		if err == nil || f.Source != "" {
			t.Fatal("capability admission widened the exact emitter target set", target)
		}
	}
}

func TestCapabilityConcurrentReaders(t *testing.T) {
	warp := capabilityWarp(t)
	record := groupCapabilityRecord(t)
	f, err := cudagroup.GenerateReductionFixtureWithCapability(warp, "sm_121a", record)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			for range 8 {
				if err := f.ValidateCapability(record); err != nil {
					t.Error(err)
				}
			}
		})
	}
	wg.Wait()
	if !reflect.DeepEqual(record, groupCapabilityRecord(t)) {
		t.Fatal("concurrent admission mutated record")
	}
}
