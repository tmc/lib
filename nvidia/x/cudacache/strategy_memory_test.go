package cudacache

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

var errStrategyMemory = errors.New("strategy memory exhausted")

type strategyMemoryLedger struct{ live, limit int64 }

func (m *strategyMemoryLedger) Reserve(n int64) error {
	if n > m.limit-m.live {
		return errStrategyMemory
	}
	m.live += n
	return nil
}

func (m *strategyMemoryLedger) Release(n int64) { m.live -= n }

func TestStrategyQueryWithMemory(t *testing.T) {
	k := new(StrategyKnowledgeBase)
	row := Strategy{Family: cudatrial.MemoryBandwidthLimited, TargetArch: "sm_90", Technique: "tiling", Confidence: 1, PredictedSpeedup: 1,
		Observations: 1, HistoricalSpeedups: []float64{2}, TrajectoryIDs: []string{strings.Repeat("a", 64)}}
	if err := k.Update(row); err != nil {
		t.Fatal(err)
	}
	want := k.Query(row.Family, row.TargetArch)
	m := &strategyMemoryLedger{limit: 1 << 20}
	scratch, _ := egraphcore.NewMemoryScope(m)
	result, _ := egraphcore.NewMemoryScope(m)
	got, err := k.QueryWithMemory(row.Family, row.TargetArch, scratch, result)
	if err != nil || !reflect.DeepEqual(got, want) || result.Bytes() < 64+8+int64(reflect.TypeFor[Strategy]().Size()) {
		t.Fatalf("snapshot=%+v want=%+v bytes=%d error=%v", got, want, result.Bytes(), err)
	}
	got[0].HistoricalSpeedups[0] = 100
	got[0].TrajectoryIDs[0] = "changed"
	if !reflect.DeepEqual(k.Query(row.Family, row.TargetArch), want) {
		t.Fatal("snapshot aliases catalog storage")
	}
	scratch.Close()
	if m.live != result.Bytes() || m.live == 0 {
		t.Fatal("returned snapshot lost its result reservation")
	}
	result.Close()
	if m.live != 0 {
		t.Fatalf("leaked %d units", m.live)
	}
	m = &strategyMemoryLedger{limit: 1024}
	scratch, _ = egraphcore.NewMemoryScope(m)
	result, _ = egraphcore.NewMemoryScope(m)
	m.limit = m.live
	if rows, err := k.QueryWithMemory(row.Family, row.TargetArch, scratch, result); !errors.Is(err, errStrategyMemory) || rows != nil {
		t.Fatalf("unadmitted copy returned: %+v, %v", rows, err)
	}
	scratch.Close()
	result.Close()
	if m.live != 0 {
		t.Fatal("refused query retained reservations")
	}
}

func ExampleStrategyKnowledgeBase_QueryWithMemory() {
	k := new(StrategyKnowledgeBase)
	_ = k.Update(Strategy{Family: cudatrial.MemoryBandwidthLimited, TargetArch: "sm_90", Technique: "tiling", Confidence: 1, PredictedSpeedup: 1})
	m := &strategyMemoryLedger{limit: 4096}
	scratch, _ := egraphcore.NewMemoryScope(m)
	result, _ := egraphcore.NewMemoryScope(m)
	rows, err := k.QueryWithMemory(cudatrial.MemoryBandwidthLimited, "sm_90", scratch, result)
	fmt.Println(len(rows), err)
	scratch.Close()
	result.Close()
	fmt.Println(m.live)
	// Output:
	// 1 <nil>
	// 0
}
