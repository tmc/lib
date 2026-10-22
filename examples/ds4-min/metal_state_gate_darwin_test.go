//go:build darwin && arm64

package main

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"math"
	"os"
	"os/exec"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// TestMetalIndependentStateGate uses the registered Q2 model, not synthetic
// weights or a loader bypass. Memory admission and the hardware lock must
// precede execution. It qualifies bounded request-state behavior, not speed.
func TestMetalIndependentStateGate(t *testing.T) {
	if os.Getenv("DS4_METAL_STATE_GATE") != "1" {
		t.Skip("opt-in admitted-model gate required")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	path := os.Getenv("DS4_GGUF_PROFILE_MODEL")
	if path == "" {
		t.Fatal("registered model required")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		t.Fatal(err)
	}
	_, err = validatePinnedModel(f, info.Size())
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	memory, err := exec.Command("/usr/bin/vm_stat").Output()
	if err != nil {
		t.Fatalf("memory admission: %v", err)
	}
	available, err := metalStateGateAvailableMemory(string(memory))
	if err != nil {
		t.Fatal(err)
	}
	const reserve = uint64(16 << 30)
	if uint64(info.Size()) > available || available-uint64(info.Size()) < reserve {
		t.Fatalf("memory admission: available=%d model=%d reserve=%d", available, info.Size(), reserve)
	}
	t.Logf("memory_admission_available_bytes=%d model_bytes=%d reserve_bytes=%d", available, info.Size(), reserve)
	r, err := newMetalRuntime(path, 128)
	if err != nil {
		t.Fatal(err)
	}
	a := &metalStateAdapter{runtime: r}
	t.Logf("metal_device=%q recommended_working_set_bytes=%d", r.dev.Name(), r.dev.Raw().RecommendedMaxWorkingSetSize())
	t.Cleanup(func() {
		if err := a.Close(); err != nil {
			t.Errorf("backend close: %v", err)
		}
	})
	newState := func() *metalRequestState {
		s, err := a.NewState(128)
		if err != nil {
			t.Fatal(err)
		}
		v := s.(*metalRequestState)
		t.Cleanup(func() {
			if err := v.Close(); err != nil {
				t.Errorf("request close: %v", err)
			}
		})
		return v
	}
	first, second, third := newState(), newState(), newState()
	if first.runtime.ops == second.runtime.ops || first.runtime.ops.hc == second.runtime.ops.hc || first.runtime.arena == second.runtime.arena {
		t.Fatal("request scratch owner alias")
	}
	left, right := metalStateFields(&first.runtime.state), metalStateFields(&second.runtime.state)
	if len(left) != len(right) {
		t.Fatal("independent field counts differ")
	}
	for i := range left {
		x, y := reflect.ValueOf(left[i]), reflect.ValueOf(right[i])
		if x.Len() != y.Len() || (x.Len() != 0 && x.Pointer() == y.Pointer()) {
			t.Fatalf("mutable field %d aliases or differs in shape", i)
		}
	}
	save := func(s *metalRequestState) []byte {
		var b bytes.Buffer
		if err := s.Save(&b); err != nil {
			t.Fatal(err)
		}
		return b.Bytes()
	}
	finite := func(s *metalRequestState) {
		logits := a.Logits(s)
		if len(logits) != Flash.Vocab {
			t.Fatalf("logits shape %d", len(logits))
		}
		if extent := nonFiniteExtent(logits); extent.bad() {
			t.Fatalf("nonfinite logits: %s", extent)
		}
	}
	for pos, token := range []int{1, 2, 3} {
		if err := a.Eval(first, token, pos); err != nil {
			t.Fatal(err)
		}
	}
	if err := a.EvalBatch(second, []int{1, 2, 3}, 0); err != nil {
		t.Fatal(err)
	}
	finite(first)
	finite(second)
	logits := func(s *metalRequestState) []float32 { return append([]float32(nil), a.Logits(s)...) }
	equalLogits := func(want, got []float32) {
		if len(want) != Flash.Vocab || len(got) != Flash.Vocab {
			t.Fatal("logits shape mismatch")
		}
		for i := range want {
			if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
				t.Fatalf("logit bits differ at %d", i)
			}
		}
	}
	equalLogits(logits(first), logits(second))
	prefix := save(first)
	if !bytes.Equal(prefix, save(second)) {
		t.Fatal("batch final state differs from scalar order")
	}
	snapshot, err := first.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := snapshot.Close(); err != nil {
			t.Errorf("snapshot close: %v", err)
		}
	}()
	if err := a.Eval(first, 4, 3); err != nil {
		t.Fatal(err)
	}
	want := save(first)
	finite(first)
	wantLogits := logits(first)
	if err := third.Load(bytes.NewReader(prefix)); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(prefix, save(third)) {
		t.Fatal("native restored prefix bytes differ")
	}
	if err := a.Eval(third, 4, 3); err != nil {
		t.Fatal(err)
	}
	finite(third)
	equalLogits(wantLogits, logits(third))
	if !bytes.Equal(want, save(third)) {
		t.Fatal("native restored continuation differs")
	}
	if err := first.Restore(snapshot); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(prefix, save(first)) {
		t.Fatal("snapshot mutated after branch evaluation")
	}
	if err := a.Eval(first, 4, 3); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(want, save(first)) {
		t.Fatal("snapshot restored continuation differs")
	}
	equalLogits(wantLogits, logits(first))
	bad := append(append([]byte(nil), want...), 1)
	if first.Load(bytes.NewReader(bad)) == nil || !bytes.Equal(want, save(first)) {
		t.Fatal("failed load mutated live state")
	}
	foreign := &metalSnapshot{owner: &metalStateAdapter{}, data: prefix}
	if !errors.Is(first.Restore(foreign), dsv4.ErrUnimplemented) || !bytes.Equal(want, save(first)) {
		t.Fatal("foreign snapshot admitted or mutated state")
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(a.Eval(first, 5, 4), dsv4.ErrClosed) {
		t.Fatal("closed request reused")
	}
	if err := a.Eval(second, 4, 3); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(want, save(second)) {
		t.Fatal("closing one request disturbed another")
	}
	equalLogits(wantLogits, logits(second))
	t.Logf("serialized_prefix_bytes=%d serialized_continuation_bytes=%d equal_prefix_checks=3 equal_continuation_checks=3 logits_equal_checks=4 logits_per_check=%d", len(prefix), len(want), Flash.Vocab)
	t.Logf("model_sha256=%s context=128 prompt=1,2,3 continuation=4 states=3 prefix_state_sha256=%x continuation_state_sha256=%x", pinnedModelSHA256, sha256.Sum256(prefix), sha256.Sum256(want))
}

func metalStateGateAvailableMemory(text string) (uint64, error) {
	lines := strings.Split(text, "\n")
	if len(lines) == 0 {
		return 0, errors.New("memory admission: missing vm_stat")
	}
	fields := strings.Fields(lines[0])
	var pageSize uint64
	for i, word := range fields {
		if word == "size" && i+2 < len(fields) && fields[i+1] == "of" {
			pageSize, _ = strconv.ParseUint(fields[i+2], 10, 64)
		}
	}
	if pageSize == 0 || pageSize > 1<<20 {
		return 0, errors.New("memory admission: invalid page size")
	}
	var pages uint64
	seen := make(map[string]bool)
	for _, line := range lines[1:] {
		name, count, ok := strings.Cut(line, ":")
		if !ok || (name != "Pages free" && name != "Pages inactive" && name != "Pages speculative") {
			continue
		}
		if seen[name] {
			return 0, errors.New("memory admission: duplicate page count")
		}
		seen[name] = true
		n, err := strconv.ParseUint(strings.TrimSuffix(strings.TrimSpace(count), "."), 10, 64)
		if err != nil || n > (^uint64(0))/pageSize || pages > (^uint64(0))/pageSize-n {
			return 0, errors.New("memory admission: invalid page count")
		}
		pages += n
	}
	if len(seen) != 3 {
		return 0, errors.New("memory admission: missing page count")
	}
	return pages * pageSize, nil
}

func TestMetalStateGateMemoryAdmission(t *testing.T) {
	text := "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 1.\nPages inactive: 2.\nPages speculative: 3.\n"
	got, err := metalStateGateAvailableMemory(text)
	if err != nil || got != 6*16384 {
		t.Fatalf("available=%d err=%v", got, err)
	}
	for _, bad := range []string{"", strings.Replace(text, "16384", "0", 1), strings.Replace(text, "Pages inactive: 2.", "", 1), text + "Pages free: 4.\n", strings.Replace(text, "Pages free: 1.", "Pages free: -1.", 1)} {
		if _, err := metalStateGateAvailableMemory(bad); err == nil {
			t.Fatal("malformed memory receipt admitted")
		}
	}
}
