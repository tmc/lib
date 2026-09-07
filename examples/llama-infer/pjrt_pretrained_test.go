package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/tmc/lib/x/xla"
)

// TestNativePJRTPretrainedReference compares real checkpoint logits against the
// independent Transformers oracle in testdata/pretrained/generate.py.
func TestNativePJRTPretrainedReference(t *testing.T) {
	dir, refDir, plugin := os.Getenv("PJRT_PRETRAINED_CHECKPOINT"), os.Getenv("PJRT_PRETRAINED_REFERENCE"), os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if dir == "" || refDir == "" || plugin == "" {
		t.Skip("set PJRT_PRETRAINED_CHECKPOINT, PJRT_PRETRAINED_REFERENCE and XLA_PJRT_CPU_PLUGIN")
	}
	ref := readPJRTReference(t, refDir)
	if ref.TeacherForced || (ref.ComputeDType != "" && ref.ComputeDType != "float32") {
		t.Fatal("strict reference test requires a free-running F32 reference")
	}

	for name, hash := range ref.Files {
		checkPJRTReferenceHash(t, filepath.Join(dir, name), hash)
	}
	tok, err := preflightPJRTText(dir)
	if err != nil {
		t.Fatal(err)
	}
	p, err := loadPJRTCheckpoint(dir, ref.Capacity, xla.F32)
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
	model, err := newPJRTModel(context.Background(), engine, &p.config, p.weights, p.theta, xla.F32)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := model.close(); err != nil {
			t.Error(err)
		}
	}()
	for _, tt := range ref.Cases {
		t.Run(tt.Prompt, func(t *testing.T) {
			if len(tt.Generated) == 0 || len(tt.Input) == 0 {
				t.Fatal("empty case")
			}
			ids, err := tok.Encode(tt.Prompt)
			if err != nil || !slices.Equal(ids, tt.Input) {
				t.Fatalf("prompt IDs %v %v, want %v", ids, err, tt.Input)
			}
			path := filepath.Join(refDir, tt.Logits)
			checkPJRTReferenceHash(t, path, tt.SHA256)
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if len(raw) != len(tt.Generated)*ref.Vocab*4 {
				t.Fatal("reference logit shape")
			}
			if err := model.reset(); err != nil {
				t.Fatal(err)
			}
			input, position := ids, 0
			maxError := 0.0
			for step, wantID := range tt.Generated {
				logits, err := model.forward(context.Background(), input, position)
				if err != nil {
					t.Fatal(err)
				}
				if got := sampleArgmax(logits); got != wantID {
					t.Fatalf("step %d token %d, want %d", step, got, wantID)
				}
				for i, got := range logits {
					want := math.Float32frombits(binary.LittleEndian.Uint32(raw[(step*ref.Vocab+i)*4:]))
					delta := math.Abs(float64(got - want))
					maxError = math.Max(maxError, delta)
					// Different f32 reduction orders are allowed; token selection must match.
					if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) || delta > 1e-3+1e-4*math.Abs(float64(want)) {
						t.Fatalf("step %d logit %d: %g, want %g (error %g)", step, i, got, want, delta)
					}
				}
				position += len(input)
				input = []int32{wantID}
			}
			var generated []int32
			var text strings.Builder
			err = model.generateText(context.Background(), tok, tt.Prompt, len(tt.Generated), nil, func(id int32, fragment string) bool {
				if !utf8.ValidString(fragment) {
					t.Errorf("invalid UTF-8 fragment %x", fragment)
				}
				generated = append(generated, id)
				text.WriteString(fragment)
				return true
			})
			if err != nil || !slices.Equal(generated, tt.Generated) || text.String() != tt.Decoded {
				t.Fatalf("stream IDs %v text %q error %v; want %v %q", generated, text.String(), err, tt.Generated, tt.Decoded)
			}
			t.Logf("compared %d logits, %d greedy IDs; maximum absolute logit error %g", len(tt.Generated)*ref.Vocab, len(generated), maxError)
		})
	}
}

type pjrtReference struct {
	ComputeDType           string `json:"compute_dtype"`
	TeacherForced          bool   `json:"teacher_forced"`
	TeacherReferenceSHA256 string `json:"teacher_reference_sha256,omitempty"`
	Implementation         string `json:"implementation,omitempty"`
	sourceHash             string
	Files                  map[string]string `json:"files"`
	Capacity               int               `json:"capacity"`
	Vocab                  int               `json:"vocab"`
	Cases                  []struct {
		TeacherIDs []int32 `json:"teacher_ids,omitempty"`
		Prompt     string  `json:"prompt"`
		Input      []int32 `json:"input_ids"`
		Generated  []int32 `json:"generated_ids"`
		Decoded    string  `json:"decoded"`
		Logits     string  `json:"logits"`
		SHA256     string  `json:"logits_sha256"`
	} `json:"cases"`
}

func checkPJRTReferenceHash(t *testing.T, path, want string) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.New()
	_, readErr := io.Copy(h, f)
	closeErr := f.Close()
	if readErr != nil {
		t.Fatal(readErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		t.Fatalf("%s SHA256 %s, want %s", path, got, want)
	}
}

func readPJRTReference(t *testing.T, dir string) *pjrtReference {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref pjrtReference
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if len(ref.Cases) == 0 || ref.Capacity <= 0 || ref.Vocab <= 0 || ref.Vocab > math.MaxInt32 || len(ref.Files) == 0 {
		t.Fatal("empty or invalid reference")
	}
	for name := range ref.Files {
		if filepath.Base(name) != name || name == "." || name == ".." {
			t.Fatal("invalid checkpoint filename")
		}
	}
	seen := make(map[string]bool)
	for _, c := range ref.Cases {
		if seen[c.Logits] || c.Logits == "reference.json" {
			t.Fatal("duplicate or reserved logit filename")
		}
		seen[c.Logits] = true
		if filepath.Base(c.Logits) != c.Logits || c.Logits == "." || c.Logits == ".." || c.Logits == "" {
			t.Fatal("invalid logit filename")
		}
		if len(c.Input) == 0 || len(c.Generated) == 0 || len(c.Input) > ref.Capacity || len(c.Generated)-1 > ref.Capacity-len(c.Input) {
			t.Fatal("reference case exceeds capacity or is empty")
		}
		if len(c.Generated) > int(^uint(0)>>1)/4/ref.Vocab {
			t.Fatal("reference logit size overflows")
		}
		if ref.TeacherForced && len(c.TeacherIDs) != len(c.Generated) {
			t.Fatal("missing teacher history")
		}
		if !ref.TeacherForced && len(c.TeacherIDs) != 0 {
			t.Fatal("unexpected teacher history")
		}
		for _, ids := range [][]int32{c.Input, c.Generated, c.TeacherIDs} {
			for _, id := range ids {
				if id < 0 || int(id) >= ref.Vocab {
					t.Fatal("reference token outside vocabulary")
				}
			}
		}
	}
	sum := sha256.Sum256(data)
	ref.sourceHash = hex.EncodeToString(sum[:])
	return &ref
}
