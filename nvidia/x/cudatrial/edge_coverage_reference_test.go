package cudatrial_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// Compare the existing host models on nonnegative signed-int block IDs. Their
// counter widths, quantization and warp aggregation are different contracts.
func TestThreadEdgeMapMatchesSASSIndices(t *testing.T) {
	m, err := cudatrial.NewThreadEdgeMap(cudasass.CoverageMapSize, 2)
	if err != nil {
		t.Fatal(err)
	}
	var reference cudasass.EdgeCoverage
	for _, step := range []struct {
		thread int
		block  int32
		want   int
	}{
		{0, 5, 32773}, {1, 9, 32777},
		{0, 9, 32779}, {1, 5, 32769},
		{0, 32773, 32769}, {1, 0, 32770},
	} {
		got, err := m.Record(step.thread, uint64(step.block))
		ref := reference.Record(uint64(step.thread), step.block)
		if err != nil || got != step.want || ref != step.want {
			t.Fatalf("record(%d, %d) = %d, %v; reference %d; want %d", step.thread, step.block, got, err, ref, step.want)
		}
	}
	before := m.Snapshot().Bytes()
	if before[32779] != 1 || before[32769] != 2 || before[32780] != 0 {
		t.Fatal("opposite transitions lost direction")
	}
	m.Reset()
	got, err := m.Record(0, 9)
	if err != nil || got != 32777 || m.Snapshot().Bytes()[32779] != 0 || before[32779] != 1 {
		t.Fatal("reset retained state or changed the snapshot")
	}
}
