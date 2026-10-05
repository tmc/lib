package cudatrial

import (
	"encoding/json"
	"sync"
	"testing"
)

func TestTrajectoryMemoryConcurrentCopies(t *testing.T) {
	m, _ := NewTrajectoryMemory(32)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				r := trajectoryFixture("fast", 50, true)
				if err := m.ImportTrusted(r); err != nil {
					t.Error(err)
				}
				rank := m.Rank(r.Domain, r.BottleneckState, r.TargetArch)
				if len(rank) != 1 || rank[0].Observations != 1 {
					t.Errorf("duplicate replay: %+v", rank)
				}
			}
		}()
	}
	wg.Wait()
}

func TestTrajectoryLiveSealExactValue(t *testing.T) {
	// This internal synthetic seal tests value binding, not native qualification.
	r := trajectoryFixture("fast", 50, true)
	r.liveSeal = r.SHA256
	if !r.HardwareObserved() {
		t.Fatal("unchanged sealed value refused")
	}
	r.MeasuredLatencyNs = 25
	r.SpeedupRatio = 4
	r.SHA256 = r.checksum()
	if r.HardwareObserved() {
		t.Fatal("recomputed declaration inherited live admission")
	}
	r = trajectoryFixture("fast", 50, true)
	r.liveSeal = r.SHA256
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var restored TrajectoryReceipt
	if err := json.Unmarshal(b, &restored); err != nil {
		t.Fatal(err)
	}
	if restored.HardwareObserved() {
		t.Fatal("serialization created live admission")
	}
}

func FuzzTrajectoryReceiptValidation(f *testing.F) {
	r := trajectoryFixture("fast", 50, true)
	b, _ := json.Marshal(r)
	f.Add(b)
	f.Add([]byte(`{"pending_hardware":true}`))
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 64<<10 {
			return
		}
		var r TrajectoryReceipt
		if json.Unmarshal(b, &r) != nil {
			return
		}
		if r.Validate() == nil && (r.PendingHardware || r.MeasuredLatencyNs <= 0 || r.SHA256 != r.checksum()) {
			t.Fatal("invalid receipt admitted")
		}
	})
}
