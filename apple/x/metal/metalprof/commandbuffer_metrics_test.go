package metalprof_test

import (
	"bytes"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/apple/x/metal/metalprof"
)

func TestCommandBufferMetrics(t *testing.T) {
	b := metalprof.NewBuilder()
	f := b.AddFunction("submit", "", 0)
	loc := b.AddLocation(f, 0)
	// Two dispatches inside one created and committed command buffer.
	b.AddSample([]uint64{loc}, metalprof.SampleValues{Samples: 1, GPUTimeNs: 100}, nil)
	b.AddSample([]uint64{loc}, metalprof.SampleValues{Samples: 1, GPUTimeNs: 200}, nil)
	b.AddSample([]uint64{loc}, metalprof.SampleValues{CommandBufferCount: 1, CommandBufferCreateTimeNs: 42}, nil)
	b.AddSample([]uint64{loc}, metalprof.SampleValues{CommandBufferCommitCount: 1}, nil)
	var buf bytes.Buffer
	if err := b.Encode(&buf); err != nil {
		t.Fatal(err)
	}
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if err := p.CheckValid(); err != nil {
		t.Fatal(err)
	}
	wantTypes := []profile.ValueType{
		{Type: "samples", Unit: "count"},
		{Type: "gpu_time", Unit: "nanoseconds"},
		{Type: "command_buffer_count", Unit: "count"},
		{Type: "command_buffer_commit_count", Unit: "count"},
		{Type: "command_buffer_create_time", Unit: "nanoseconds"},
	}
	if len(p.SampleType) != len(wantTypes) {
		t.Fatalf("sample types: %v", p.SampleType)
	}
	for i, want := range wantTypes {
		if *p.SampleType[i] != want {
			t.Errorf("sample type %d = %v, want %v", i, *p.SampleType[i], want)
		}
	}
	want := []int64{2, 300, 1, 1, 42}
	got := make([]int64, len(p.SampleType))
	for _, s := range p.Sample {
		for i, v := range s.Value {
			got[i] += v
		}
	}
	if len(got) != len(want) {
		t.Fatalf("sample types = %v", p.SampleType)
	}
	for i, v := range want {
		if got[i] != v {
			t.Errorf("%s = %d, want %d", p.SampleType[i].Type, got[i], v)
		}
	}
	if p.DefaultSampleType != "gpu_time" {
		t.Errorf("default = %q", p.DefaultSampleType)
	}
}

func TestCommandBufferMetricsOmitted(t *testing.T) {
	b := metalprof.NewBuilder()
	f := b.AddFunction("kernel", "", 0)
	b.AddSample([]uint64{b.AddLocation(f, 0)}, metalprof.SampleValues{Samples: 1, GPUTimeNs: 10}, nil)
	var buf bytes.Buffer
	if err := b.Encode(&buf); err != nil {
		t.Fatal(err)
	}
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if len(p.SampleType) != 2 || len(p.Sample[0].Value) != 2 {
		t.Fatal("uncollected command-buffer metrics appeared")
	}
	if p.DefaultSampleType != "gpu_time" {
		t.Fatalf("default = %q", p.DefaultSampleType)
	}
}
