package cuptipcsampling

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func testSamplingBuffer(t *testing.T) *samplingBuffer {
	t.Helper()
	b, err := newSamplingBuffer(2, 2, 240)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.release)
	b.data.TotalNumPcs = 1
	b.data.TotalSamples, b.data.DroppedSamples = 1<<63+10, 1<<63+2
	b.data.NonUsrKernelsTotalSamples = 3
	b.data.RangeId, b.data.RemainingNumPcs = 1<<63+7, 5
	b.pcs[0].CubinCrc, b.pcs[0].PcOffset = 1<<63+9, 1<<63+11
	b.pcs[0].FunctionName, b.pcs[0].StallReasonCount = 123, 2
	b.reasons[0][0] = cupti.PCSamplingStallReason{PcSamplingStallReasonIndex: 77, Samples: 1<<31 + 1}
	b.reasons[0][1] = cupti.PCSamplingStallReason{PcSamplingStallReasonIndex: 9, Samples: 1}
	return b
}

func TestSamplingBufferSnapshot(t *testing.T) {
	b := testSamplingBuffer(t)
	got, err := b.snapshot(func(p uintptr) (string, error) {
		if p != 123 {
			t.Fatal("wrong name pointer")
		}
		return "kernel", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.total != 1<<63+10 || got.dropped != 1<<63+2 || got.rangeID != 1<<63+7 || got.remaining != 5 || got.nonUser != 3 || len(got.pcs) != 1 {
		t.Fatalf("batch=%+v", got)
	}
	p := got.pcs[0]
	if p.crc != 1<<63+9 || p.offset != 1<<63+11 || p.reasons[0].Samples != 1<<31+1 || p.name != "kernel" {
		t.Fatalf("PC=%+v", p)
	}
	b.reset()
	if p.reasons[0].Samples != 1<<31+1 {
		t.Fatal("snapshot aliases native reasons")
	}
	empty, err := b.snapshot(nil)
	if err != nil || len(empty.pcs) != 0 {
		t.Fatalf("empty=%+v %v", empty, err)
	}
	b.release()
	if _, err := b.snapshot(nil); err == nil {
		t.Fatal("read after release")
	}
}

func TestSamplingBufferRejectsNativeMutation(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*samplingBuffer)
	}{
		{"data pointer", func(b *samplingBuffer) { b.data.PPcData++ }},
		{"PC count", func(b *samplingBuffer) { b.data.TotalNumPcs = 3 }},
		{"capacity", func(b *samplingBuffer) { b.data.CollectNumPcs++ }},
		{"data size", func(b *samplingBuffer) { b.data.Size-- }},
		{"hardware flag", func(b *samplingBuffer) { b.data.HardwareBufferFull = 2 }},
		{"PC size", func(b *samplingBuffer) { b.pcs[0].Size-- }},
		{"reasons pointer", func(b *samplingBuffer) { b.pcs[0].StallReason++ }},
		{"reason count", func(b *samplingBuffer) { b.pcs[0].StallReasonCount = 3 }},
		{"duplicate reason", func(b *samplingBuffer) { b.reasons[0][1].PcSamplingStallReasonIndex = 77 }},
		{"missing name", func(b *samplingBuffer) { b.pcs[0].FunctionName = 0 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := testSamplingBuffer(t)
			tt.mutate(b)
			got, err := b.snapshot(func(uintptr) (string, error) { return "kernel", nil })
			if err == nil || got.pcs != nil || got.dropped != 1<<63+2 {
				t.Fatalf("got=%+v err=%v", got, err)
			}
		})
	}
	b := testSamplingBuffer(t)
	failure := errors.New("name budget")
	got, err := b.snapshot(func(uintptr) (string, error) { return "", failure })
	if !errors.Is(err, failure) || got.pcs != nil {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}

func TestSamplingBufferBudget(t *testing.T) {
	for _, tt := range []struct {
		pcs, reasons int
		bytes        uint64
	}{
		{0, 1, 100}, {1, 0, 100}, {1, 65537, 1000000}, {2, 2, 239}, {1, 1, 0}, {int(^uint(0) >> 1), 1, ^uint64(0)},
	} {
		if b, err := newSamplingBuffer(tt.pcs, tt.reasons, tt.bytes); err == nil {
			b.release()
			t.Fatalf("accepted %+v", tt)
		}
	}
}
