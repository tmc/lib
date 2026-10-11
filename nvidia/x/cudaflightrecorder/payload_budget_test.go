package cudaflightrecorder

import (
	"errors"
	"math"
	"testing"
)

func TestSnapshotPayloadExactBoundary(t *testing.T) {
	files := map[string][]byte{"fault.json": make([]byte, 40), "negative_rules.json": make([]byte, 60)}
	for _, tc := range []struct {
		name  string
		limit int64
		pass  bool
	}{
		{"exact", 100, true}, {"one-less", 99, false}, {"maximum-config", math.MaxInt64, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := checkSnapshotPayload(files, tc.limit)
			if tc.pass && err != nil || !tc.pass && !errors.Is(err, ErrSnapshotBudgetExceeded) {
				t.Fatalf("budget result: %v", err)
			}
		})
	}
	files["attachment"] = []byte{0}
	if !errors.Is(checkSnapshotPayload(files, 100), ErrSnapshotBudgetExceeded) {
		t.Fatal("attachment byte not counted")
	}
}
