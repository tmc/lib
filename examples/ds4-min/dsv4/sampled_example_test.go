package dsv4

import (
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/sampler"
)

func ExampleNewSampledSession() {
	// Native backends provide these request-bound contracts; this example
	// uses a host fixture with a deterministic point-mass distribution.
	b := &sampledTestBackend{rows: [][]float32{{0, 9, -9}, {0, 9, -9}, {0, 9, -9}, {0, 9, -9}, {0, 9, -9}}}
	request, err := NewSession(b, 64, NewStopSet(2))
	if err != nil {
		fmt.Println(err)
		return
	}
	defer request.Close()
	if err = request.Sync([]int{0}); err != nil {
		fmt.Println(err)
		return
	}
	d := &sampledTestDraft{b: b, preparedDrafter: preparedDrafter{proposal: []int{1, 1}}}
	s, err := NewSampledSession(request, &sampledTestVerifier{b: b}, d, sampler.Params{}, sampler.RNG{State: 7}, ExactSampling, SampledPolicy{}, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer s.Close()
	cycle, err := s.Next(12)
	fmt.Println(cycle.Tokens, cycle.Accepted, err)
	fmt.Println(s.Random().State)
	// Output:
	// [1 1 1] 2 <nil>
	// 7
}

func ExampleSampledPolicy() {
	var policy SampledPolicy
	for i := 0; i < 4; i++ {
		policy.Allow(20)
		policy.Note(1, false, nil)
	}
	skip := policy.Skip
	fmt.Println(skip, policy.Allow(20), policy.Allow(20), policy.Allow(20))
	// Output:
	// 2 false false true
}

func ExampleSampledSession_Close() {
	b := &sampledTestBackend{rows: [][]float32{{0, 9, -9}}, factoryErr: true, ownerFail: true}
	request, _ := NewSession(b, 64, NewStopSet(2))
	request.Sync([]int{0})
	d := &sampledTestDraft{b: b}
	s, err := NewSampledSessionForBackend(request, d, sampler.Params{}, sampler.RNG{State: 7}, ExactSampling, SampledPolicy{}, 2)
	fmt.Println("cleanup retained:", s != nil && err != nil)
	// Keep request alive until cleanup succeeds; failed Close is retryable.
	fmt.Println("retry required:", s.Close() != nil)
	b.ownerFail = false
	fmt.Println(s.Close())
	fmt.Println(request.Close())
	// Output:
	// cleanup retained: true
	// retry required: true
	// <nil>
	// <nil>
}
