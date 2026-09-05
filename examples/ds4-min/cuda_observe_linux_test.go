//go:build linux

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/gguf"
)

type cudaObserveTop struct {
	ID      int     `json:"id"`
	Logit   float32 `json:"logit"`
	Logprob float32 `json:"logprob"`
}
type cudaObserveState struct {
	Accepted int              `json:"accepted_decode_count"`
	Position int              `json:"session_position"`
	Vocab    int              `json:"vocab"`
	Argmax   int              `json:"argmax"`
	Top      []cudaObserveTop `json:"top20"`
	Margins  []float64        `json:"adjacent_top20_margins"`
	Logits   []float32        `json:"logits"`
}
type cudaObserveSelected struct {
	Index    int   `json:"selected_index"`
	Accepted int   `json:"accepted_before"`
	ID       int   `json:"id"`
	Stop     bool  `json:"stop"`
	Bytes    []int `json:"bytes"`
}

// TestCUDAObserveModel is an explicit, expensive model observation. Its output
// schema and stop-before-eval behavior match evidence/oracle-provenance/observe.c.
func TestCUDAObserveModel(t *testing.T) {
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
	var expected []cudaObserveSelected
	if oracle := os.Getenv("DS4_OBSERVE_ORACLE_DIR"); oracle != "" {
		raw, err := os.ReadFile(filepath.Join(oracle, "selected.jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte{'\n'}) {
			var selected cudaObserveSelected
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
	cudaObserveDirectory(t, dir)
	begin := time.Now()
	r, err := newCUDARuntime(path, 1024)
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			if err := r.Close(); err != nil {
				t.Errorf("close runtime: %v", err)
			}
		}
	}()
	opened := time.Now()
	tok, err := newTokenizer(r.gguf)
	if err != nil {
		t.Fatal(err)
	}
	cudaObserveStopRule(t, tok)
	for _, id := range prompt {
		if id >= len(tok.tokens) {
			t.Fatalf("prompt token %d outside vocabulary", id)
		}
	}
	gpuPrefill := cudaObserveMeasure(t, r, func() error {
		// Match the oracle's prefill_chunk=1, including each token's output head.
		for position, id := range prompt {
			if err := r.Eval(id, position); err != nil {
				return err
			}
		}
		return nil
	})
	pref := time.Now()
	cudaObserveJSON(t, dir, "input.json", struct {
		Backend string `json:"backend"`
		Ctx     int    `json:"ctx"`
		Chunk   int    `json:"prefill_chunk"`
		Max     int    `json:"max_accepted"`
		TP      bool   `json:"tp"`
		Stop    string `json:"stop_rule"`
		IDs     []int  `json:"prompt_ids"`
	}{"cuda", 1024, 1, limit, false, "ds4_token_is_stop", prompt})
	snapshot := func(accepted int) {
		t.Helper()
		cudaObserveJSON(t, dir, fmt.Sprintf("state-%02d.json", accepted), cudaObserveSnapshot(t, r.Logits(), accepted, r.pos))
	}
	snapshot(0)
	var events, decoded bytes.Buffer
	accepted, selected, stopping := 0, 0, -1
	reason := "length"
	gpuDecode := []float64{}
	wallDecode := []float64{}
	for accepted < limit {
		logits := r.Logits()
		cudaObserveFinite(t, logits)
		id := greedyToken(logits)
		if id < 0 || id >= len(tok.tokens) {
			t.Fatalf("invalid selected token%d", id)
		}
		stop := tok.IsStop(id)
		piece, err := tok.Decode([]int{id})
		if err != nil {
			t.Fatal(err)
		}
		values := make([]int, len(piece))
		for i, b := range piece {
			values[i] = int(b)
		}
		if err := json.NewEncoder(&events).Encode(cudaObserveSelected{selected, accepted, id, stop, values}); err != nil {
			t.Fatal(err)
		}
		if expected != nil {
			if selected >= len(expected) {
				cudaObserveJSON(t, dir, "failure-state.json", cudaObserveSnapshot(t, logits, accepted, r.pos))
				t.Fatal("candidate selected beyond oracle stop")
			}
			want := expected[selected]
			if id != want.ID || stop != want.Stop || !bytes.Equal(piece, observeSelectedBytes(want.Bytes)) {
				cudaObserveWrite(t, dir, "selected.jsonl", events.Bytes())
				cudaObserveJSON(t, dir, "failure-state.json", cudaObserveSnapshot(t, logits, accepted, r.pos))
				t.Fatalf("selection %d: token=%d stop=%v bytes=%v, oracle token=%d stop=%v bytes=%v", selected, id, stop, values, want.ID, want.Stop, want.Bytes)
			}
		}
		selected++
		// Persist each selected token, including stop selections that are not evaluated.
		cudaObserveWrite(t, dir, "selected.jsonl", events.Bytes())
		if stop {
			reason, stopping = "model_stop", id
			break
		}
		decoded.Write(piece)
		cudaObserveWrite(t, dir, "decoded.bin", decoded.Bytes())
		start := time.Now()
		gpuDecode = append(gpuDecode, cudaObserveMeasure(t, r, func() error { return r.Eval(id, r.pos) }))
		wallDecode = append(wallDecode, time.Since(start).Seconds())
		accepted++
		t.Logf("accepted=%d eval_wall_seconds=%.9g", accepted, wallDecode[len(wallDecode)-1])
		if accepted == 1 || accepted == 2 || accepted == 4 || accepted == 8 || accepted == 16 {
			snapshot(accepted)
		}
	}
	cudaObserveWrite(t, dir, "selected.jsonl", events.Bytes())
	cudaObserveWrite(t, dir, "decoded.bin", decoded.Bytes())
	if expected != nil && selected != len(expected) {
		t.Fatalf("selected %d oracle events, want all %d", selected, len(expected))
	}
	captured := time.Since(pref).Seconds()
	missing := []int{}
	for _, n := range []int{1, 2, 4, 8, 16} {
		if n > accepted {
			missing = append(missing, n)
		}
	}
	gpuTotal := gpuPrefill
	for _, v := range gpuDecode {
		gpuTotal += v
	}
	cudaObserveJSON(t, dir, "timing.json", struct {
		Prefill float64   `json:"prefill_gpu_event_seconds"`
		Decode  []float64 `json:"decode_gpu_event_seconds"`
		Wall    []float64 `json:"decode_wall_seconds"`
		Scope   string    `json:"scope"`
	}{gpuPrefill, gpuDecode, wallDecode, "same-stream event spans around each eval; includes host submission gaps and excludes model loading; not a sum of isolated kernel times"})
	if err := r.Close(); err != nil {
		closed = true
		t.Fatalf("close runtime: %v", err)
	}
	closed = true
	cudaObserveJSON(t, dir, "result.json", struct {
		Reason    string  `json:"stop_reason"`
		Stop      int     `json:"stop_token"`
		Selected  int     `json:"selected_count"`
		Accepted  int     `json:"accepted_count"`
		Engine    float64 `json:"engine_wall_seconds"`
		Prefill   float64 `json:"prefill_wall_seconds"`
		Capture   float64 `json:"capture_wall_seconds"`
		GPU       float64 `json:"gpu_event_time"`
		GPUReason string  `json:"gpu_event_reason"`
		Missing   []int   `json:"missing_required_states"`
	}{reason, stopping, selected, accepted, opened.Sub(begin).Seconds(), pref.Sub(opened).Seconds(), captured, gpuTotal, "seconds; sum of prefill and decode default-stream CUDA event spans; see timing.json", missing})
}

func cudaObserveMeasure(t *testing.T, r *cudaRuntime, run func() error) float64 {
	t.Helper()
	start, err := r.device.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := start.Destroy(); err != nil {
			t.Errorf("destroy start event: %v", err)
		}
	}()
	end, err := r.device.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := end.Destroy(); err != nil {
			t.Errorf("destroy end event: %v", err)
		}
	}()
	stream := r.device.DefaultStream()
	if err := start.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := run(); err != nil {
		t.Fatal(err)
	}
	if err := end.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := end.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	elapsed, err := end.ElapsedSince(start)
	if err != nil {
		t.Fatal(err)
	}
	if elapsed <= 0 {
		t.Fatal("nonpositive CUDA event duration")
	}
	return elapsed.Seconds()
}
func cudaObserveFinite(t *testing.T, logits []float32) {
	t.Helper()
	if err := cudaObserveLogitsError(logits); err != nil {
		t.Fatal(err)
	}
}
func cudaObserveLogitsError(logits []float32) error {
	if len(logits) < 20 {
		return fmt.Errorf("incomplete logits")
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
func cudaObserveStopRule(t *testing.T, tok *tokenizer) {
	t.Helper()
	if len(tok.stops) != 1 || !tok.stops.Contains(tok.eos) || tok.tokens[tok.eos] != endTokenText {
		t.Fatalf("stop set %v does not match pinned DeepSeek EOS-only rule", tok.stops)
	}
}
func cudaObserveSnapshot(t *testing.T, logits []float32, accepted, position int) cudaObserveState {
	t.Helper()
	cudaObserveFinite(t, logits)
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
	top := make([]cudaObserveTop, 20)
	margins := make([]float64, 19)
	for i := range top {
		id := ids[i]
		top[i] = cudaObserveTop{id, logits[id], float32(float64(logits[id]) - logsum)}
	}
	for i := range margins {
		margins[i] = float64(top[i].Logit) - float64(top[i+1].Logit)
	}
	return cudaObserveState{accepted, position, len(logits), ids[0], top, margins, append([]float32(nil), logits...)}
}
func cudaObserveDirectory(t *testing.T, dir string) {
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
func cudaObserveWrite(t *testing.T, dir, name string, b []byte) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), b, 0644); err != nil {
		t.Fatal(err)
	}
}
func cudaObserveJSON(t *testing.T, dir, name string, v any) {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	cudaObserveWrite(t, dir, name, append(b, '\n'))
}

// This opt-in metadata-only test emits fixed prompts without creating a GPU
// runtime. DS4_OBSERVE_RENDER_DIR must be an empty artifact directory.
func TestCUDAObserveRenderPrompts(t *testing.T) {
	dir := os.Getenv("DS4_OBSERVE_RENDER_DIR")
	if dir == "" {
		t.Skip("DS4_OBSERVE_RENDER_DIR is not set")
	}
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Fatal("DS4_MODEL is required")
	}
	cudaObserveDirectory(t, dir)
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			t.Error(err)
		}
	}()
	info, err := file.Stat()
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Read(file, info.Size())
	if err != nil {
		t.Fatal(err)
	}
	if err := validatePinnedProfile(g); err != nil {
		t.Fatal(err)
	}
	tok, err := newTokenizer(g)
	if err != nil {
		t.Fatal(err)
	}
	digests, err := profileHashes(g)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err = filepath.EvalSymlinks(canonical)
	if err != nil {
		t.Fatal(err)
	}
	cudaObserveStopRule(t, tok)
	t.Logf("model canonical path: %s", canonical)
	t.Logf("metadata=%s tensors=%s quantization=%s tokenizer=%s", digests.metadata, digests.tensors, digests.quant, digests.tokenizer)
	cudaObserveJSON(t, dir, "identity.json", struct {
		Model     string `json:"model"`
		Size      int64  `json:"size"`
		Metadata  string `json:"metadata"`
		Tensors   string `json:"tensors"`
		Quant     string `json:"quantization"`
		Tokenizer string `json:"tokenizer"`
		Stops     []int  `json:"stop_ids"`
	}{canonical, info.Size(), digests.metadata, digests.tensors, digests.quant, digests.tokenizer, []int{tok.eos}})
	for _, p := range []struct{ name, text string }{{"prompt-a", "a"}, {"count-30", "Count from 1 to 30, writing each number on its own line."}} {
		ids, err := tok.RenderNonThinking("You are a helpful assistant", p.text)
		if err != nil {
			t.Fatal(err)
		}
		var text strings.Builder
		for _, id := range ids {
			fmt.Fprintln(&text, id)
		}
		cudaObserveWrite(t, dir, p.name+".ids", []byte(text.String()))
		cudaObserveJSON(t, dir, p.name+".json", struct {
			System string `json:"system"`
			Prompt string `json:"prompt"`
			IDs    []int  `json:"prompt_ids"`
		}{"You are a helpful assistant", p.text, ids})
	}
}

func TestCUDAObserveSnapshotSemantics(t *testing.T) {
	logits := make([]float32, 24)
	logits[21], logits[23], logits[20] = 5, 5, -2
	got := cudaObserveSnapshot(t, logits, 4, 37)
	if got.Accepted != 4 || got.Position != 37 || got.Vocab != 24 || got.Argmax != 21 {
		t.Fatalf("incorrect snapshot metadata: %+v", got)
	}
	logsum := math.Log(2*math.Exp(5) + 21 + math.Exp(-2))
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
		bad := make([]float32, 24)
		bad[17] = v
		if cudaObserveLogitsError(bad) == nil {
			t.Fatalf("accepted nonfinite %g", v)
		}
	}
	if cudaObserveLogitsError(make([]float32, 19)) == nil {
		t.Fatal("accepted incomplete logits")
	}
	event, err := json.Marshal(cudaObserveSelected{0, 0, 1, true, []int{0, 255}})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(event, []byte(`"bytes":[0,255]`)) {
		t.Fatalf("token bytes are not a numeric array: %s", event)
	}
}

func observeSelectedBytes(values []int) []byte {
	b := make([]byte, len(values))
	for i, v := range values {
		b[i] = byte(v)
	}
	return b
}
