package cudatrial_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func ExampleRaceCorpus() {
	corpus := cudatrial.NewRaceCorpus()
	fmt.Printf("Total test cases: %d\n", len(corpus.TestCases()))

	// Evaluate false positive case
	resFP, err := corpus.EvaluateCase("false_positives/shared_bytewise_readwrite", false, "sm_90")
	if err != nil {
		panic(err)
	}
	fmt.Printf("False positive race detected: %v (passed=%v)\n", resFP.HasRace, resFP.Passed())

	// Evaluate synchronized memcpy case
	resFixed, err := corpus.EvaluateCase("memcpy/shared_readwrite_race", true, "sm_80")
	if err != nil {
		panic(err)
	}
	fmt.Printf("Fixed memcpy race detected: %v (passed=%v)\n", resFixed.HasRace, resFixed.Passed())

	// Output:
	// Total test cases: 27
	// False positive race detected: false (passed=true)
	// Fixed memcpy race detected: false (passed=true)
}
