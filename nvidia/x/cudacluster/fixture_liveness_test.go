package cudacluster

import (
	"strings"
	"testing"
)

func TestReductionFixturePreservesInitializedSkewControl(t *testing.T) {
	for _, mut := range []Mutant{MutantNone, MutantLocalWindow, MutantNoSync} {
		source, err := ReductionFixture("sm_121a", 4, mut)
		if err != nil {
			t.Fatal(err)
		}
		initial := strings.Index(source, "window = 0;")
		ready := strings.Index(source, "cluster.sync();")
		delay := strings.Index(source, "clock64() - start < 1000000ull")
		publish := strings.Index(source, "window = seed_for(rank);")
		if initial < 0 || ready <= initial || delay <= ready || publish <= delay {
			t.Fatalf("%s: defined initialization and bounded skew do not precede publication", mut)
		}
		// The mutant removes only the publication barrier. Initialization and
		// the post-read lifetime barrier must remain in all three variants.
		count := strings.Count(source, "barrier.cluster.arrive;")
		want := 2
		if mut == MutantNoSync {
			want = 1
		}
		if count != want {
			t.Fatalf("%s: emitted barriers=%d want=%d", mut, count, want)
		}
	}
}
