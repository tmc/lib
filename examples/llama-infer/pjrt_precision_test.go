package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/tmc/lib/x/xla"
)

// TestNativePJRTPrecisionDiagnostic records logits on a fixed F32 reference
// history. Passing means the measurement completed with finite outputs; it does
// not establish numerical parity or a token-match threshold.
func TestNativePJRTPrecisionDiagnostic(t *testing.T) {
	checkpoint, reference := os.Getenv("PJRT_PRETRAINED_CHECKPOINT"), os.Getenv("PJRT_PRETRAINED_REFERENCE")
	plugin, precision, output := os.Getenv("XLA_PJRT_CPU_PLUGIN"), os.Getenv("PJRT_PRECISION"), os.Getenv("PJRT_PRECISION_OUTPUT")
	if checkpoint == "" || reference == "" || plugin == "" || precision == "" || output == "" {
		t.Skip("set PJRT_PRETRAINED_CHECKPOINT, PJRT_PRETRAINED_REFERENCE, XLA_PJRT_CPU_PLUGIN, PJRT_PRECISION and PJRT_PRECISION_OUTPUT")
	}
	element, err := pjrtElementType(precision)
	if err != nil {
		t.Fatal(err)
	}
	ref := readPJRTReference(t, reference)
	if ref.TeacherForced || (ref.ComputeDType != "" && ref.ComputeDType != "float32") {
		t.Fatal("diagnostic requires a free-running F32 teacher")
	}
	for name, hash := range ref.Files {
		checkPJRTReferenceHash(t, filepath.Join(checkpoint, name), hash)
	}
	tok, err := preflightPJRTText(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	// Reserve a new directory before the expensive model setup. Partial results
	// remain inspectable on failure; a later invocation cannot overwrite them.
	if err := os.MkdirAll(filepath.Dir(output), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(output, 0755); err != nil {
		t.Fatal(err)
	}
	p, err := loadPJRTCheckpoint(checkpoint, ref.Capacity, element)
	if err != nil {
		t.Fatal(err)
	}
	if p.config.VocabSize != ref.Vocab {
		t.Fatal("reference vocabulary mismatch")
	}
	engine, err := xla.NewEngine(plugin)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := engine.Close(); err != nil {
			t.Error(err)
		}
	}()
	model, err := p.newModel(context.Background(), engine)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := model.close(); err != nil {
			t.Error(err)
		}
	}()
	recorded := *ref
	recorded.ComputeDType = map[string]string{"f32": "float32", "f16": "float16", "bf16": "bfloat16"}[precision]
	recorded.Implementation, recorded.TeacherForced = "tmc", true
	recorded.TeacherReferenceSHA256 = ref.sourceHash
	recorded.Cases = slices.Clone(ref.Cases)
	for i, c := range ref.Cases {
		ids, err := tok.Encode(c.Prompt)
		if err != nil || !slices.Equal(ids, c.Input) {
			t.Fatalf("prompt %d token IDs differ: %v", i, err)
		}
		checkPJRTReferenceHash(t, filepath.Join(reference, c.Logits), c.SHA256)
		if err := model.reset(); err != nil {
			t.Fatal(err)
		}
		raw := make([]byte, len(c.Generated)*ref.Vocab*4)
		actual := make([]int32, len(c.Generated))
		input, position, matches := ids, 0, 0
		for step, teacher := range c.Generated {
			logits, err := model.forward(context.Background(), input, position)
			if err != nil {
				t.Fatal(err)
			}
			if len(logits) != ref.Vocab {
				t.Fatal("output vocabulary mismatch")
			}
			actual[step] = sampleArgmax(logits)
			if actual[step] == teacher {
				matches++
			}
			for j, value := range logits {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("prompt %d step %d logit %d is nonfinite", i, step, j)
				}
				binary.LittleEndian.PutUint32(raw[(step*ref.Vocab+j)*4:], math.Float32bits(value))
			}
			position += len(input)
			input = []int32{teacher}
		}
		path := filepath.Join(output, c.Logits)
		if err := os.WriteFile(path, raw, 0600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(raw)
		recorded.Cases[i].SHA256 = hex.EncodeToString(sum[:])
		recorded.Cases[i].TeacherIDs = slices.Clone(c.Generated)
		recorded.Cases[i].Generated = actual
		// Per-row choices under forced histories are not a free-running text sample.
		recorded.Cases[i].Decoded = ""
		t.Logf("diagnostic prompt %d: finite logits %d, teacher choices matched %d/%d", i, len(raw)/4, matches, len(actual))
	}
	data, err := json.MarshalIndent(recorded, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(output, "reference.json"), append(data, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
}
