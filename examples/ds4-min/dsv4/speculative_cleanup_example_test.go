package dsv4

import (
	"errors"
	"fmt"
	"io"
)

func ExampleSpeculator_Close() {
	v := &preparedVerifier{speculativeVerifier: speculativeVerifier{values: []int{8}, logits: []float32{0, 1}, tops: []int{1}}, closeErr: errors.New("release failed")}
	d := &preparedDrafter{proposal: []int{1}, seed: 8, position: 1}
	s, _ := NewSpeculator(v, d)
	_, err := s.VerifyPrepared(8, 1, 1)
	var owner io.Closer
	fmt.Println("cleanup retained:", errors.As(err, &owner))
	fmt.Println("retry needed:", owner.Close() != nil)
	v.closeErr = nil
	fmt.Println(owner.Close())
	// Only now may the caller release the verifier, drafter and request.
	_, err = s.VerifyPrepared(8, 1, 1)
	fmt.Println(errors.Is(err, ErrClosed))
	// Output:
	// cleanup retained: true
	// retry needed: true
	// <nil>
	// true
}
