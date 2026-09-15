package cuptirange

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestCachePolicyAdmission(t *testing.T) {
	for _, version := range []uint32{24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400, 0, 130500} {
		for _, policy := range []CachePolicy{CacheDefault, CachePreserve, CacheFlushL2, CacheFlushAll, CachePolicy(-1), CachePolicy(4)} {
			want := qualifiedLayout(version) && (policy == CacheDefault || policy == CachePreserve)
			if err := CachePolicySupport(version, policy); (err == nil) != want {
				t.Errorf("version=%d policy=%v error=%v want admitted=%t", version, policy, err, want)
			}
		}
	}
}

func TestCacheResetFailsBeforeWorkload(t *testing.T) {
	names, err := newMetricNames([]string{"metric"})
	if err != nil {
		t.Fatal(err)
	}
	defer names.close()
	p := &Plan{host: 1, names: names, passes: 1, version: 130400}
	for _, policy := range []CachePolicy{CacheFlushL2, CacheFlushAll} {
		result, err := p.CaptureReplay(new(cuda.Device), Limits{MaxRanges: 1, MaxRangeTreeNodes: 1, MaxCounterBytes: 1024, MaxPasses: 1}, CaptureOptions{Mode: Whole, CachePolicy: policy}, Replay{Prepare: func(uint) error { t.Fatal("unsupported cache policy executed prepare"); return nil }, Run: func(uint, *Scope) error { t.Fatal("unsupported cache policy executed run"); return nil }, Synchronize: func() error { t.Fatal("unsupported cache policy executed synchronization"); return nil }})
		if result != nil || err == nil || !strings.Contains(err.Error(), "cache reset ABI") {
			t.Fatalf("policy=%s result=%v error=%v", policy, result, err)
		}
	}
}

func ExampleCachePolicy() {
	fmt.Println(CacheDefault, CachePreserve, CacheFlushL2, CacheFlushAll)
	// Output: default preserve flush-l2 flush-all
}

func ExampleCachePolicySupport() {
	fmt.Println(CachePolicySupport(130202, CachePreserve))
	fmt.Println(CachePolicySupport(130202, CacheFlushL2) != nil)
	// Output:
	// <nil>
	// true
}
