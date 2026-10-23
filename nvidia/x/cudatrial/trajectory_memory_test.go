package cudatrial

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/egraphcore"
)

var errReplayMemory = errors.New("replay memory exhausted")

type replayMemoryLedger struct{ live, limit int64 }

func (m *replayMemoryLedger) Reserve(n int64) error {
	if n > m.limit-m.live {
		return errReplayMemory
	}
	m.live += n
	return nil
}

func (m *replayMemoryLedger) Release(n int64) { m.live -= n }

func TestTrajectoryReplayWithMemory(t *testing.T) {
	memory, _ := NewTrajectoryMemory(4)
	for _, r := range []TrajectoryReceipt{trajectoryFixture("fast", 50, true), trajectoryFixture("slow", 200, true)} {
		r.AppliedRules = append(r.AppliedRules, r.AppliedRules[0])
		r.SHA256 = r.checksum()
		if err := memory.ImportTrusted(r); err != nil {
			t.Fatal(err)
		}
	}
	domain := strings.Repeat("a", 64)
	want := memory.Replay(domain, MemoryBandwidthLimited, "sm_121")
	m := &replayMemoryLedger{limit: 1 << 20}
	scratch, _ := egraphcore.NewMemoryScope(m)
	result, _ := egraphcore.NewMemoryScope(m)
	got, err := memory.ReplayWithMemory(domain, MemoryBandwidthLimited, "sm_121", scratch, result)
	if err != nil || !reflect.DeepEqual(got, want) || len(got.Rewards) != 2 || got.Rewards[0].Reward != 1 || got.Rewards[0].Observations != 1 || got.Rewards[1].Reward != -1 {
		t.Fatalf("snapshot=%+v want=%+v error=%v", got, want, err)
	}
	got.Rewards[0].Rule = "changed"
	got.ReceiptIDs[0] = "changed"
	if !reflect.DeepEqual(memory.Replay(domain, MemoryBandwidthLimited, "sm_121"), want) {
		t.Fatal("replay aliases retained history")
	}
	scratch.Close()
	if m.live != result.Bytes() || m.live <= 128 {
		t.Fatal("result lost replay reservation")
	}
	result.Close()
	if m.live != 0 {
		t.Fatal("replay reservation leaked")
	}
	m = &replayMemoryLedger{limit: 1024}
	scratch, _ = egraphcore.NewMemoryScope(m)
	result, _ = egraphcore.NewMemoryScope(m)
	m.limit = m.live
	if got, err := memory.ReplayWithMemory(domain, MemoryBandwidthLimited, "sm_121", scratch, result); !errors.Is(err, errReplayMemory) || len(got.ReceiptIDs)+len(got.Rewards) != 0 {
		t.Fatalf("unadmitted replay returned: %+v, %v", got, err)
	}
	scratch.Close()
	result.Close()
	if m.live != 0 {
		t.Fatal("refused replay retained reservations")
	}
}

func ExampleTrajectoryMemory_ReplayWithMemory() {
	memory, _ := NewTrajectoryMemory(1)
	m := &replayMemoryLedger{limit: 4096}
	scratch, _ := egraphcore.NewMemoryScope(m)
	result, _ := egraphcore.NewMemoryScope(m)
	snapshot, err := memory.ReplayWithMemory(strings.Repeat("a", 64), MemoryBandwidthLimited, "sm_90", scratch, result)
	fmt.Println(len(snapshot.Rewards), err)
	scratch.Close()
	result.Close()
	fmt.Println(m.live)
	// Output:
	// 0 <nil>
	// 0
}
