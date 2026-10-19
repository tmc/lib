package cudasearch_test

import (
	"bytes"
	"context"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"strings"
)

func ExampleNewCandidateQuarantine() {
	store, err := cudasearch.NewCandidateQuarantine(16, 16384)
	if err != nil {
		panic(err)
	}
	entries, bytes := store.Retained()
	fmt.Println(entries, bytes > 0)
	// Output: 0 true
}

func ExampleCandidateQuarantine_Import() {
	// This example supplies local policy assertions, not a GPU-fault proof.
	association := cudaflightrecorder.CandidateAssociation{Version: 1, Kind: cudaflightrecorder.CandidateReferenceMismatch, IdentityProvenance: cudaflightrecorder.CallerAssertedCandidateIdentity, ExecutableID: "compiler artifact and launch", Generation: 1, WorkloadKey: strings.Repeat("a", 64), InputKey: strings.Repeat("b", 64), VerificationKey: strings.Repeat("c", 64)}
	r, err := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 1})
	if err != nil {
		panic(err)
	}
	defer r.Close()
	var archive bytes.Buffer
	_, err = r.FaultSnapshot(context.Background(), &archive, cudaflightrecorder.FaultInfo{ErrorName: cudaflightrecorder.CandidateReferenceMismatch, Candidate: &association})
	if err != nil {
		panic(err)
	}
	store, err := cudasearch.NewCandidateQuarantine(16, 16384)
	if err != nil {
		panic(err)
	}
	if err := store.Import(&archive); err != nil {
		panic(err)
	}
	entries, _ := store.Retained()
	fmt.Println(entries)
	// Install &cudasearch.FailureEvidence{Quarantine: store} on the concrete
	// ExecutableEvaluator. Matching compiled owners close before evaluation.
	// Output: 1
}
