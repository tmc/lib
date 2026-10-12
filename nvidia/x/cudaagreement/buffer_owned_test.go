package cudaagreement

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestOwnedBufferRefusesRawAndInvalidOwners(t *testing.T) {
	for _, tt := range []struct {
		name string
		b    Buffer
	}{
		{"raw", Buffer{Ptr: 4096, Bytes: 4}},
		{"zero owned", BufferFromSlice(cudamem.Slice[uint32]{})},
		{"altered pointer", func() Buffer { b := BufferFromSlice(cudamem.Slice[uint32]{}); b.Ptr = 4096; return b }()},
		{"altered range", func() Buffer { b := BufferFromSlice(cudamem.Slice[uint32]{}); b.Bytes = 4; return b }()},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if release, err := tt.b.borrow(&cuda.Device{}); err == nil || release != nil {
				t.Fatal("invalid binding admitted", err)
			}
		})
	}
}

func TestMegaRawOwnershipRefusedBeforeCompiler(t *testing.T) {
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "in", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("control"), "control", schema, "sm_80")
	if err != nil {
		t.Fatal(err)
	}
	plan := &Plan{Name: "control", Steps: []Step{{Name: "add", Artifact: a, Inputs: []string{"x"}, Output: "y", Grid: 1, Block: 32}}}
	calls := 0
	e := &MegaExecutor{Device: &cuda.Device{}, Modules: &Modules{}, Grid: cudamega.Grid{CTAs: 1, ThreadsPerCTA: 32}, Operations: map[string]cudamega.Operation{"add": {Kind: cudamega.Add, Scalar: 1}}, Compile: func(string, cudamega.Target) (cudakernel.Kind, []byte, error) {
		calls++
		return cudakernel.KindPTX, nil, nil
	}}
	_, err = e.Execute(&cuda.Stream{}, plan, map[string]Buffer{"x": {Ptr: 4096, Bytes: 4}, "y": {Ptr: 8192, Bytes: 4}})
	if err == nil || !strings.Contains(err.Error(), "BufferFromSlice") || calls != 0 || e.pending != nil {
		t.Fatalf("raw ownership: %v, compiler calls %d, pending %v", err, calls, e.pending)
	}
}

func ExampleBufferFromSlice() {
	// An actual allocated slice keeps its owner in the persistent binding.
	// The zero slice is useful only as an empty descriptor; Execute refuses it.
	b := BufferFromSlice(cudamem.Slice[uint32]{})
	fmt.Println(b.Bytes)
	// Output: 0
}
