package cudatrial_test

import (
	"crypto/sha256"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

type simpleFilter struct {
	seen map[[32]byte]bool
}

func (f *simpleFilter) Evaluate(raw []byte) (bool, error) {
	h := sha256.Sum256(raw)
	if f.seen[h] {
		return false, nil
	}
	f.seen[h] = true
	return true, nil
}

func ExampleNoveltyScheduler() {
	filter := &simpleFilter{seen: make(map[[32]byte]bool)}
	sched, err := cudatrial.NewNoveltyScheduler(filter)
	if err != nil {
		panic(err)
	}

	b1, _ := cudatrial.NewEdgeBitmap(32)
	b1.Bytes()[16] = 1
	r1 := cudatrial.Receipt{Name: "trial1", Coverage: b1}

	b2, _ := cudatrial.NewEdgeBitmap(32)
	b2.Bytes()[16] = 1
	r2 := cudatrial.Receipt{Name: "trial2", Coverage: b2}

	admitted1, _ := sched.ScheduleTrial(r1)
	admitted2, _ := sched.ScheduleTrial(r2)

	fmt.Printf("Trial 1 novel: %v\n", admitted1)
	fmt.Printf("Trial 2 novel: %v\n", admitted2)

	stats := sched.Stats()
	fmt.Printf("Evaluated: %d, Novel: %d, Duplicate: %d\n",
		stats.EvaluatedTrials, stats.NovelTrials, stats.DuplicateTrials)

	// Output:
	// Trial 1 novel: true
	// Trial 2 novel: false
	// Evaluated: 2, Novel: 1, Duplicate: 1
}
