package cudasearch

import (
	"encoding/binary"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func searchHandleFixture(t *testing.T, entry, paramType string) *cudakernel.Artifact {
	t.Helper()
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "count", Type: paramType})
	if err != nil {
		t.Fatal(err)
	}
	// This retained-byte fixture is never compiled, loaded or executed.
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("host-only artifact handle fixture "+entry), entry, schema, "compute_121")
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestSequenceArtifactHandleAdmission(t *testing.T) {
	a := searchHandleFixture(t, "first", "u32")
	for _, tc := range []struct {
		name      string
		artifacts []*cudakernel.Artifact
		launches  int
		want      string
	}{
		{"empty", nil, 0, "artifact/launch count mismatch"},
		{"count", []*cudakernel.Artifact{a}, 2, "artifact/launch count mismatch"},
		{"nil", []*cudakernel.Artifact{nil}, 1, "nil sequence artifact"},
		{"later-nil", []*cudakernel.Artifact{a, nil}, 2, "nil sequence artifact"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := sequenceArtifactHandles(tc.artifacts, tc.launches)
			if got != nil || err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("capture = %v, %v; want no handles and %q", got, err, tc.want)
			}
		})
	}
}

func TestSequenceArtifactHandleRetention(t *testing.T) {
	a := searchHandleFixture(t, "first", "u32")
	b := searchHandleFixture(t, "second", "f32")
	first, second := *a, *b
	input := []*cudakernel.Artifact{a, b}
	captured, err := sequenceArtifactHandles(input, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(captured) != 2 || captured[0] != first || captured[1] != second {
		t.Fatal("captured handles changed image identity or order")
	}

	// Replace both caller wrappers and the input slice after capture. The
	// loaded-stage handles must still identify the original images and ABI.
	*a, *b = second, first
	input[0], input[1] = b, a
	if a.EntryPoint() != "second" || b.EntryPoint() != "first" {
		t.Fatal("caller wrappers were not replaced")
	}
	stage := sequenceStage{artifact: captured[0]}
	if stage.artifact != first || captured[1] != second {
		t.Fatal("retained handle aliases caller wrapper or input slice")
	}
	data, err := stage.artifact.Pack(uint32(7))
	if err != nil || len(data) != 8 || binary.LittleEndian.Uint32(data) != 7 {
		t.Fatalf("retained ABI pack = %x, %v", data, err)
	}
	if _, err := stage.artifact.Pack(float32(7)); err == nil {
		t.Fatal("retained u32 ABI accepted the replacement f32 argument")
	}
	if _, err := a.Pack(float32(7)); err != nil {
		t.Fatalf("replacement ABI precondition: %v", err)
	}
	detached := stage.artifact.Schema()
	detached.Params[0].Type = "f32"
	if _, err := stage.artifact.Pack(float32(7)); err == nil {
		t.Fatal("detached schema edit changed retained ABI")
	}
}

func TestSequenceArtifactHandleRetirement(t *testing.T) {
	a := searchHandleFixture(t, "retained", "u32")
	x := &sequenceExecutable{stages: []sequenceStage{{artifact: *a}}}
	// No device/module/stream/buffers exist: this exercises actual Close without
	// permitting a native call. The descriptor itself must retire with its owner.
	if err := x.Close(); err != nil {
		t.Fatal(err)
	}
	if x.stages[0].artifact != (cudakernel.Artifact{}) {
		t.Fatal("closed stage retained artifact image")
	}
	if err := a.Validate(); err != nil {
		t.Fatalf("retiring a captured handle changed caller: %v", err)
	}
	if err := x.Close(); err != nil {
		t.Fatalf("repeat close: %v", err)
	}
}
