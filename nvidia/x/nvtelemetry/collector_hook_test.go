package nvtelemetry

import (
	"context"
	"testing"
	"time"
)

func TestCollectorHookAfterBufferDrop(t *testing.T) {
	for _, drop := range []bool{false, true} {
		t.Run(map[bool]string{false: "stop", true: "drop"}[drop], func(t *testing.T) {
			sampler := NewFakeSampler(DeviceMetadata{UUID: "GPU-a"})
			sampler.Enqueue([]Reading{httpReading()}, []Reading{httpReading()})
			clock := NewFakeClock(1, 1000)
			collector := NewCollector("hook", sampler, clock, CollectorConfig{IntervalNanos: int64(time.Second), MaxBufferedRecords: 1, DropOnFull: drop}, nil)
			var seen []uint64
			collector.SetSampleHook(func(s Sample) { seen = append(seen, s.Seq); s.Readings[0].IntValue = 0 })
			if err := collector.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			clock.Advance(time.Second)
			err := collector.Step(context.Background())
			if drop {
				if err != nil || len(seen) != 2 {
					t.Fatalf("dropped buffer lost callback: %v %v", seen, err)
				}
			} else if err == nil || len(seen) != 1 {
				t.Fatalf("failed acquisition notified: %v %v", seen, err)
			}
			if collector.Samples()[0].Readings[0].IntValue == 0 {
				t.Fatal("hook mutated retained sample")
			}
			if len(collector.Samples()) != 1 {
				t.Fatal("buffer bound changed")
			}
		})
	}
}
