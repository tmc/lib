//go:build darwin && arm64

package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

type metalObserveTop struct {
	ID      int     `json:"id"`
	Logit   float32 `json:"logit"`
	Logprob float32 `json:"logprob"`
}
type metalObserveState struct {
	Accepted int               `json:"accepted_decode_count"`
	Position int               `json:"session_position"`
	Vocab    int               `json:"vocab"`
	Argmax   int               `json:"argmax"`
	Top      []metalObserveTop `json:"top20"`
	Margins  []float64         `json:"adjacent_top20_margins"`
	Logits   []float32         `json:"logits"`
}
type metalObserveSelected struct {
	Index    int   `json:"selected_index"`
	Accepted int   `json:"accepted_before"`
	ID       int   `json:"id"`
	Stop     bool  `json:"stop"`
	Bytes    []int `json:"bytes"`
}

// TestMetalObserveModel is an explicit, expensive model observation. Its output
// schema and stop-before-eval behavior match evidence/oracle-provenance/observe.c.
// Derived from cuda_observe_linux_test.go; CUDA is intentionally left unchanged.
// GPU event timing is unavailable here and is reported as null, never as zero.
func TestMetalObserveModel(t *testing.T) {
	path, dir, idsPath, limitText := os.Getenv("DS4_MODEL"), os.Getenv("DS4_OBSERVE_DIR"), os.Getenv("DS4_OBSERVE_PROMPT_IDS"), os.Getenv("DS4_OBSERVE_MAX_ACCEPTED")
	if dir == "" && idsPath == "" && limitText == "" {
		t.Skip("DS4_OBSERVE_DIR, DS4_OBSERVE_PROMPT_IDS and DS4_OBSERVE_MAX_ACCEPTED are not set")
	}
	if path == "" || dir == "" || idsPath == "" || limitText == "" {
		t.Fatal("DS4_MODEL and all three DS4_OBSERVE settings are required")
	}
	limit, err := strconv.Atoi(limitText)
	if err != nil || limit < 16 || limit > 512 {
		t.Fatal("DS4_OBSERVE_MAX_ACCEPTED must be 16..512")
	}
	var expected []metalObserveSelected
	if oracle := os.Getenv("DS4_OBSERVE_ORACLE_DIR"); oracle != "" {
		raw, err := os.ReadFile(filepath.Join(oracle, "selected.jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte{'\n'}) {
			var selected metalObserveSelected
			if err := json.Unmarshal(line, &selected); err != nil {
				t.Fatal(err)
			}
			for _, v := range selected.Bytes {
				if v < 0 || v > 255 {
					t.Fatal("invalid oracle byte")
				}
			}
			expected = append(expected, selected)
		}
	}
	input, err := os.ReadFile(idsPath)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{}
	for _, field := range strings.Fields(string(input)) {
		for _, c := range field {
			if c < '0' || c > '9' {
				t.Fatalf("non-decimal prompt ID %q", field)
			}
		}
		id, err := strconv.ParseUint(field, 10, 31)
		if err != nil {
			t.Fatal(err)
		}
		prompt = append(prompt, int(id))
	}
	if len(prompt) == 0 || len(prompt)+limit+1 > 1024 {
		t.Fatal("invalid prompt length for context1024")
	}
	metalObserveDirectory(t, dir)
	metalObserveJSON(t, dir, "input.json", struct {
		Backend string `json:"backend"`
		Ctx     int    `json:"ctx"`
		Chunk   int    `json:"prefill_chunk"`
		Max     int    `json:"max_accepted"`
		TP      bool   `json:"tp"`
		Stop    string `json:"stop_rule"`
		IDs     []int  `json:"prompt_ids"`
	}{"metal", 1024, 1, limit, false, "ds4_token_is_stop", prompt})
	begin := time.Now()
	r, err := newMetalRuntime(path, 1024)
	if err != nil {
		metalObserveFailure(t, dir, nil, 0, 0, err)
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			if err := r.Close(); err != nil {
				metalObserveJSON(t, dir, "close-error.json", map[string]string{"error": err.Error()})
				t.Errorf("close runtime: %v", err)
			}
		}
	}()
	opened := time.Now()
	tok, err := newTokenizer(r.model)
	if err != nil {
		metalObserveFailure(t, dir, r.Logits(), 0, r.pos, err)
		t.Fatal(err)
	}
	if err := metalObserveStopRule(tok); err != nil {
		metalObserveFailure(t, dir, r.Logits(), 0, r.pos, err)
		t.Fatal(err)
	}
	for _, id := range prompt {
		if id >= len(tok.tokens) {
			err := fmt.Errorf("prompt token %d outside vocabulary", id)
			metalObserveFailure(t, dir, r.Logits(), 0, r.pos, err)
			t.Fatal(err)
		}
	}
	prefillStart := time.Now()
	for position, id := range prompt {
		// Chunk1 includes each token's output head, matching the pinned oracle.
		if err := r.Eval(id, position); err != nil {
			metalObserveFailure(t, dir, r.Logits(), 0, r.pos, fmt.Errorf("prefill token %d: %w", position, err))
			t.Fatal(err)
		}
	}
	prefillWall := time.Since(prefillStart).Seconds()
	pref := time.Now()

	snapshot := func(accepted int) {
		t.Helper()
		if err := metalObserveLogitsError(r.Logits()); err != nil {
			metalObserveFailure(t, dir, r.Logits(), accepted, r.pos, err)
			t.Fatal(err)
		}
		metalObserveJSON(t, dir, fmt.Sprintf("state-%02d.json", accepted), metalObserveSnapshot(t, r.Logits(), accepted, r.pos))
	}
	snapshot(0)
	var events, decoded bytes.Buffer
	accepted, selected, stopping := 0, 0, -1
	reason := "length"
	wallDecode := []float64{}
	for accepted < limit {
		logits := r.Logits()
		if err := metalObserveLogitsError(logits); err != nil {
			metalObserveFailure(t, dir, logits, accepted, r.pos, err)
			t.Fatal(err)
		}
		id := greedyToken(logits)
		if id < 0 || id >= len(tok.tokens) {
			err := fmt.Errorf("invalid selected token %d", id)
			metalObserveFailure(t, dir, logits, accepted, r.pos, err)
			t.Fatal(err)
		}
		stop := tok.IsStop(id)
		piece, err := tok.Decode([]int{id})
		if err != nil {
			metalObserveFailure(t, dir, logits, accepted, r.pos, err)
			t.Fatal(err)
		}
		values := make([]int, len(piece))
		for i, b := range piece {
			values[i] = int(b)
		}
		if err := json.NewEncoder(&events).Encode(metalObserveSelected{selected, accepted, id, stop, values}); err != nil {
			t.Fatal(err)
		}
		metalObserveWrite(t, dir, "selected.jsonl", events.Bytes())
		if expected != nil {
			if selected >= len(expected) {
				metalObserveFailure(t, dir, logits, accepted, r.pos, fmt.Errorf("oracle selection mismatch at selected %d", selected))
				t.Fatal("candidate selected beyond oracle stop")
			}
			want := expected[selected]
			if want.Index != selected || want.Accepted != accepted || id != want.ID || stop != want.Stop || !bytes.Equal(piece, metalObserveSelectedBytes(want.Bytes)) {
				metalObserveWrite(t, dir, "selected.jsonl", events.Bytes())
				metalObserveFailure(t, dir, logits, accepted, r.pos, fmt.Errorf("oracle selection mismatch at selected %d", selected))
				t.Fatalf("selection %d: token=%d stop=%v bytes=%v, oracle token=%d stop=%v bytes=%v", selected, id, stop, values, want.ID, want.Stop, want.Bytes)
			}
		}
		selected++
		// Persist each selected token, including stop selections that are not evaluated.
		metalObserveWrite(t, dir, "selected.jsonl", events.Bytes())
		if stop {
			reason, stopping = "model_stop", id
			break
		}
		decoded.Write(piece)
		metalObserveWrite(t, dir, "decoded.bin", decoded.Bytes())
		start := time.Now()
		if err := r.Eval(id, r.pos); err != nil {
			metalObserveFailure(t, dir, r.Logits(), accepted, r.pos, fmt.Errorf("decode accepted %d: %w", accepted, err))
			t.Fatal(err)
		}
		wallDecode = append(wallDecode, time.Since(start).Seconds())
		accepted++
		if err := metalObserveLogitsError(r.Logits()); err != nil {
			metalObserveFailure(t, dir, r.Logits(), accepted, r.pos, err)
			t.Fatal(err)
		}
		t.Logf("accepted=%d eval_wall_seconds=%.9g", accepted, wallDecode[len(wallDecode)-1])
		if accepted == 1 || accepted == 2 || accepted == 4 || accepted == 8 || accepted == 16 {
			snapshot(accepted)
		}
	}
	metalObserveWrite(t, dir, "selected.jsonl", events.Bytes())
	metalObserveWrite(t, dir, "decoded.bin", decoded.Bytes())
	if expected != nil && selected != len(expected) {
		metalObserveFailure(t, dir, r.Logits(), accepted, r.pos, fmt.Errorf("selected count %d, oracle %d", selected, len(expected)))
		t.Fatalf("selected %d oracle events, want all %d", selected, len(expected))
	}
	captured := time.Since(pref).Seconds()
	missing := []int{}
	for _, n := range []int{1, 2, 4, 8, 16} {
		if n > accepted {
			missing = append(missing, n)
		}
	}
	metalObserveJSON(t, dir, "timing.json", struct {
		PrefillGPU  *float64  `json:"prefill_gpu_event_seconds"`
		DecodeGPU   []float64 `json:"decode_gpu_event_seconds"`
		PrefillWall float64   `json:"prefill_wall_seconds"`
		Wall        []float64 `json:"decode_wall_seconds"`
		Scope       string    `json:"scope"`
	}{nil, nil, prefillWall, wallDecode, "UNMEASURED GPU event time; wall times include host submission and completed Metal eval; exclude model loading"})
	finalLogits := append([]float32(nil), r.Logits()...)
	if err := r.Close(); err != nil {
		closed = true
		metalObserveFailure(t, dir, finalLogits, accepted, r.pos, fmt.Errorf("close runtime: %w", err))
		t.Fatalf("close runtime: %v", err)
	}
	closed = true
	metalObserveJSON(t, dir, "result.json", struct {
		Reason    string   `json:"stop_reason"`
		Stop      int      `json:"stop_token"`
		Selected  int      `json:"selected_count"`
		Accepted  int      `json:"accepted_count"`
		Engine    float64  `json:"engine_wall_seconds"`
		Prefill   float64  `json:"prefill_wall_seconds"`
		Capture   float64  `json:"capture_wall_seconds"`
		GPU       *float64 `json:"gpu_event_time"`
		GPUReason string   `json:"gpu_event_reason"`
		Missing   []int    `json:"missing_required_states"`
	}{reason, stopping, selected, accepted, opened.Sub(begin).Seconds(), pref.Sub(opened).Seconds(), captured, nil, "UNMEASURED: Metal observer has no equivalent same-stream GPU event instrumentation; see wall times in timing.json", missing})
}

func metalObserveFinite(t *testing.T, logits []float32) {
	t.Helper()
	if err := metalObserveLogitsError(logits); err != nil {
		t.Fatal(err)
	}
}
func metalObserveLogitsError(logits []float32) error {
	if len(logits) != 129280 {
		return fmt.Errorf("logit count %d, want full vocabulary 129280", len(logits))
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("non-finite logit%d: %g", i, v)
		}
	}
	return nil
}

// Pinned ds4.c:39580 resolves EOS by literal token, and :39970 stops only
// on EOS for the DeepSeek family. Extra optional metadata must not silently
// change the observer's claimed ds4_token_is_stop semantics.
func metalObserveStopRule(tok *tokenizer) error {
	if tok.eos < 0 || tok.eos >= len(tok.tokens) || len(tok.stops) != 1 || !tok.stops.Contains(tok.eos) || tok.tokens[tok.eos] != endTokenText {
		return fmt.Errorf("stop set %v does not match pinned DeepSeek EOS-only rule", tok.stops)
	}
	return nil
}

func metalObserveSnapshot(t *testing.T, logits []float32, accepted, position int) metalObserveState {
	t.Helper()
	metalObserveFinite(t, logits)
	ids := make([]int, len(logits))
	for i := range ids {
		ids[i] = i
	}
	sort.SliceStable(ids, func(i, j int) bool { return logits[ids[i]] > logits[ids[j]] })
	maximum := float64(logits[ids[0]])
	sum := 0.0
	for _, v := range logits {
		sum += math.Exp(float64(v) - maximum)
	}
	logsum := maximum + math.Log(sum)
	top := make([]metalObserveTop, 20)
	margins := make([]float64, 19)
	for i := range top {
		id := ids[i]
		top[i] = metalObserveTop{id, logits[id], float32(float64(logits[id]) - logsum)}
	}
	for i := range margins {
		margins[i] = float64(top[i].Logit) - float64(top[i+1].Logit)
	}
	return metalObserveState{accepted, position, len(logits), ids[0], top, margins, append([]float32(nil), logits...)}
}
func metalObserveDirectory(t *testing.T, dir string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatal("observation output directory must be empty")
	}
}
func metalObserveWrite(t *testing.T, dir, name string, b []byte) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), b, 0644); err != nil {
		t.Fatal(err)
	}
}
func metalObserveJSON(t *testing.T, dir, name string, v any) {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	metalObserveWrite(t, dir, name, append(b, '\n'))
}

func TestMetalObserveSnapshotSemantics(t *testing.T) {
	logits := make([]float32, 129280)
	logits[21], logits[23], logits[20] = 5, 5, -2
	got := metalObserveSnapshot(t, logits, 4, 37)
	if got.Accepted != 4 || got.Position != 37 || got.Vocab != 129280 || got.Argmax != 21 {
		t.Fatalf("incorrect snapshot metadata: %+v", got)
	}
	logsum := math.Log(2*math.Exp(5) + 129277 + math.Exp(-2))
	for i, entry := range got.Top {
		want := i - 2
		if i == 0 {
			want = 21
		}
		if i == 1 {
			want = 23
		}
		if entry.ID != want || entry.Logit != logits[want] || entry.Logprob != float32(float64(logits[want])-logsum) {
			t.Fatalf("top[%d]=%+v, want id%d logit%g logprob%g", i, entry, want, logits[want], float32(float64(logits[want])-logsum))
		}
	}
	for i, v := range got.Margins {
		want := 0.0
		if i == 1 {
			want = 5
		}
		if v != want {
			t.Fatalf("margin%d=%g want%g", i, v, want)
		}
	}
	logits[21] = 99
	if got.Logits[21] != 5 {
		t.Fatal("snapshot aliases runtime logits")
	}
	for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		bad := make([]float32, 129280)
		bad[17] = v
		if metalObserveLogitsError(bad) == nil {
			t.Fatalf("accepted nonfinite %g", v)
		}
	}
	if metalObserveLogitsError(make([]float32, 19)) == nil {
		t.Fatal("accepted incomplete logits")
	}
	event, err := json.Marshal(metalObserveSelected{0, 0, 1, true, []int{0, 255}})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(event, []byte(`"bytes":[0,255]`)) {
		t.Fatalf("token bytes are not a numeric array: %s", event)
	}
}

func metalObserveSelectedBytes(values []int) []byte {
	b := make([]byte, len(values))
	for i, v := range values {
		b[i] = byte(v)
	}
	return b
}

// Invalid floats cannot be encoded in JSON. Preserve their exact IEEE bits in a
// sidecar while leaving a readable failure-state receipt before the test fails.
func metalObserveFailure(t *testing.T, dir string, logits []float32, accepted, position int, cause error) {
	t.Helper()
	metalObserveJSON(t, dir, "failure-error.json", struct {
		Accepted int    `json:"accepted_decode_count"`
		Position int    `json:"session_position"`
		Error    string `json:"error"`
	}{accepted, position, cause.Error()})
	if err := metalObserveLogitsError(logits); err == nil {
		metalObserveJSON(t, dir, "failure-state.json", metalObserveSnapshot(t, logits, accepted, position))
		return
	}
	raw := make([]byte, 4*len(logits))
	for i, v := range logits {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	metalObserveWrite(t, dir, "failure-logits.f32", raw)
	metalObserveJSON(t, dir, "failure-state.json", struct {
		Accepted int    `json:"accepted_decode_count"`
		Position int    `json:"session_position"`
		Vocab    int    `json:"vocab"`
		Valid    bool   `json:"logits_valid"`
		Error    string `json:"error"`
		Raw      string `json:"raw_logits_le_f32"`
	}{accepted, position, len(logits), false, cause.Error(), "failure-logits.f32"})
}

func TestMetalObserveFailureReceipt(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "metal-observe-failure-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Error(err)
		}
	})
	logits := make([]float32, 129280)
	metalObserveFailure(t, dir, logits, 4, 37, fmt.Errorf("injected selection mismatch"))
	raw, err := os.ReadFile(filepath.Join(dir, "failure-state.json"))
	if err != nil {
		t.Fatal(err)
	}
	var state metalObserveState
	if err := json.Unmarshal(raw, &state); err != nil {
		t.Fatal(err)
	}
	if state.Accepted != 4 || state.Position != 37 || len(state.Logits) != 129280 || len(state.Top) != 20 {
		t.Fatal("incomplete finite failure receipt")
	}
	logits[9] = math.Float32frombits(0x7fc00042)
	metalObserveFailure(t, dir, logits, 8, 41, fmt.Errorf("injected NaN"))
	raw, err = os.ReadFile(filepath.Join(dir, "failure-state.json"))
	if err != nil {
		t.Fatal(err)
	}
	var invalid struct {
		Valid    bool   `json:"logits_valid"`
		Raw      string `json:"raw_logits_le_f32"`
		Accepted int    `json:"accepted_decode_count"`
	}
	if err := json.Unmarshal(raw, &invalid); err != nil {
		t.Fatal(err)
	}
	if invalid.Valid || invalid.Raw != "failure-logits.f32" || invalid.Accepted != 8 {
		t.Fatal("missing invalid failure receipt")
	}
	raw, err = os.ReadFile(filepath.Join(dir, invalid.Raw))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 129280*4 || binary.LittleEndian.Uint32(raw[36:]) != 0x7fc00042 {
		t.Fatal("failure receipt lost NaN bits")
	}
}

// Keep the isolated backend adapter's persisted records compatible with CUDA.
// This reads source only; it does not build or execute a CUDA runtime.
func TestMetalObserveSchema(t *testing.T) {
	source, err := parser.ParseFile(token.NewFileSet(), "cuda_observe_linux_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	types := map[string]reflect.Type{
		"cudaObserveTop":      reflect.TypeOf(metalObserveTop{}),
		"cudaObserveState":    reflect.TypeOf(metalObserveState{}),
		"cudaObserveSelected": reflect.TypeOf(metalObserveSelected{}),
	}
	for _, decl := range source.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, spec := range gen.Specs {
			named, ok := spec.(*ast.TypeSpec)
			if !ok {
				continue
			}
			want, ok := types[named.Name.Name]
			if !ok {
				continue
			}
			record, ok := named.Type.(*ast.StructType)
			if !ok {
				t.Fatalf("%s is not a struct", named.Name.Name)
			}
			if len(record.Fields.List) != want.NumField() {
				t.Fatalf("%s field count changed", named.Name.Name)
			}
			for i, field := range record.Fields.List {
				if len(field.Names) != 1 || field.Tag == nil {
					t.Fatal("unsupported observer schema declaration")
				}
				tag, err := strconv.Unquote(field.Tag.Value)
				if err != nil {
					t.Fatal(err)
				}
				if field.Names[0].Name != want.Field(i).Name || tag != string(want.Field(i).Tag) {
					t.Fatalf("%s field %d diverged from CUDA schema", named.Name.Name, i)
				}
			}
			delete(types, named.Name.Name)
		}
	}
	if len(types) != 0 {
		t.Fatalf("missing CUDA schema declarations: %v", types)
	}
}
