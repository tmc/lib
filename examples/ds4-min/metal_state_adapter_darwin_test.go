//go:build darwin && arm64

package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/gguf"
)

func metalStateFixtureIdentity(t *testing.T, r *metalRuntime) {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "identity")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	if _, err := f.WriteString("synthetic state model"); err != nil {
		t.Fatal(err)
	}
	r.file, r.model, r.stateIdentity = f, &gguf.File{}, new(modelStateIdentityCache)
}

func TestMetalStateAdapterFailsClosed(t *testing.T) {
	a := &metalStateAdapter{}
	state, err := a.NewState(4096)
	if !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("NewState error = %v, want dsv4.ErrUnimplemented", err)
	}
	if state != nil {
		t.Fatal("NewState returned state with ErrUnimplemented")
	}
	if err := a.Eval(nil, 1, 0); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("Eval error = %v, want dsv4.ErrUnimplemented", err)
	}
	if got := a.Logits(nil); got != nil {
		t.Fatalf("Logits = %v, want nil for unsupported state", got)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("Close on nil runtime = %v, want nil", err)
	}
}

func TestMetalStateHeaderSizeMatchesLayerMetadata(t *testing.T) {
	const fixedHeader = 8 + 4 + 4 + 8 + 4
	want := fixedHeader + 8*ds4Layers + 64 // counts and model/quantization hashes
	if metalStateHeaderSize != want {
		t.Fatalf("metalStateHeaderSize = %d, want %d", metalStateHeaderSize, want)
	}
}

func TestMetalStateAdapterCloseIsTerminal(t *testing.T) {
	a := &metalStateAdapter{}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if state, err := a.NewState(16); !errors.Is(err, dsv4.ErrClosed) || state != nil {
		t.Fatalf("NewState after Close = (%v, %v)", state, err)
	}
	if err := a.Eval(nil, 1, 0); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("Eval after Close = %v", err)
	}
}

func TestMetalRequestStatePersistenceFailsClosed(t *testing.T) {
	s := &metalRequestState{}
	if _, err := s.Snapshot(); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("Snapshot error = %v, want dsv4.ErrUnimplemented", err)
	}
	if err := s.Restore(nil); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("Restore error = %v, want dsv4.ErrUnimplemented", err)
	}
	var saved bytes.Buffer
	if err := s.Save(&saved); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("Save error = %v, want dsv4.ErrUnimplemented", err)
	}
	if err := s.Load(bytes.NewReader(nil)); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("Load error = %v, want dsv4.ErrUnimplemented", err)
	}
}

// metalStateDeviceRequest returns a request whose saved state includes real
// shared-buffer contents, so byte comparisons cover data as well as metadata.
func metalStateDeviceRequest(t *testing.T, dev *libmetal.Device, owner *metalStateAdapter, seed float32) *metalRequestState {
	t.Helper()
	r := &metalRuntime{ctx: 16, pos: 7, arena: metalmem.NewArena(dev), ops: &metalPinnedOps{hc: &metalHC{}}}
	t.Cleanup(r.arena.Free)
	metalStateFixtureIdentity(t, r)
	r.state.hc[0] = metalmem.UploadIn(r.arena, "hc-a", []float32{seed, seed + 1, seed + 2})
	r.state.hc[1] = metalmem.UploadIn(r.arena, "hc-b", []float32{seed + 3, seed + 4})
	r.state.logits = metalmem.UploadIn(r.arena, "logits", []float32{seed + 5, seed + 6})
	if err := r.arena.Err(); err != nil {
		t.Fatal(err)
	}
	r.state.layers[0].attn.cap = 4
	r.state.layers[0].attn.n = 2
	return &metalRequestState{runtime: r, owner: owner}
}

func metalStateBytes(t *testing.T, r *metalRuntime) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := saveMetalState(r, &b); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// metalStateDiverge changes buffer contents and metadata so that a restore
// which drops any of them is visible in the saved bytes.
func metalStateDiverge(t *testing.T, r *metalRuntime, want []byte) {
	t.Helper()
	r.state.hc[0].HostSlice()[0] = -1
	r.state.logits.HostSlice()[1] = -2
	r.pos = 1
	r.state.layers[0].attn.n = 0
	if bytes.Equal(metalStateBytes(t, r), want) {
		t.Fatal("divergence did not change saved state")
	}
}

func newMetalStateTestDevice(t *testing.T) *libmetal.Device {
	t.Helper()
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.Close() })
	return dev
}

// Snapshot and Restore are request-local (spec 05 §4.1): a request restores
// its own snapshot bit-exactly.
func TestMetalRequestStateSnapshotRoundTrip(t *testing.T) {
	dev := newMetalStateTestDevice(t)
	s := metalStateDeviceRequest(t, dev, &metalStateAdapter{}, 1)
	want := metalStateBytes(t, s.runtime)
	snapshot, err := s.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	defer snapshot.Close()
	metalStateDiverge(t, s.runtime, want)
	epoch := s.runtime.speculativeEpoch
	if err := s.Restore(snapshot); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if got := metalStateBytes(t, s.runtime); !bytes.Equal(got, want) {
		t.Fatalf("same-request restore is not bit-exact: pos %d attn n %d hc[0] %v logits %v", s.runtime.pos, s.runtime.state.layers[0].attn.n, s.runtime.state.hc[0].HostSlice(), s.runtime.state.logits.HostSlice())
	}
	if s.runtime.speculativeEpoch != epoch+1 {
		t.Fatalf("restore epoch = %d, want %d", s.runtime.speculativeEpoch, epoch+1)
	}
}

// A snapshot is bound to the request that took it, even within one backend
// and model. Cross-request transfer uses Save and Load.
func TestMetalRequestStateRestoreRejectsForeignRequest(t *testing.T) {
	dev := newMetalStateTestDevice(t)
	owner := &metalStateAdapter{}
	s := metalStateDeviceRequest(t, dev, owner, 1)
	snapshot, err := s.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	defer snapshot.Close()
	other := metalStateDeviceRequest(t, dev, owner, 20)
	other.runtime.pos = 3
	before := metalStateBytes(t, other.runtime)
	epoch := other.runtime.speculativeEpoch
	if err := other.Restore(snapshot); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("foreign-request restore: %v, want ErrUnimplemented", err)
	}
	if !bytes.Equal(metalStateBytes(t, other.runtime), before) || other.runtime.speculativeEpoch != epoch {
		t.Fatal("foreign-request restore changed the destination")
	}
	if err := s.Restore(snapshot); err != nil {
		t.Fatalf("source request rejected its own snapshot: %v", err)
	}
}

// Save and Load are the portable path: another request, under another
// adapter with the same model identity, loads the bytes exactly.
func TestMetalRequestStatePortableSaveLoad(t *testing.T) {
	dev := newMetalStateTestDevice(t)
	s := metalStateDeviceRequest(t, dev, &metalStateAdapter{}, 1)
	var saved bytes.Buffer
	if err := s.Save(&saved); err != nil {
		t.Fatalf("Save: %v", err)
	}
	other := metalStateDeviceRequest(t, dev, &metalStateAdapter{}, 20)
	metalStateDiverge(t, other.runtime, saved.Bytes())
	if err := other.Load(bytes.NewReader(saved.Bytes())); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := metalStateBytes(t, other.runtime); !bytes.Equal(got, saved.Bytes()) {
		t.Fatalf("portable Load is not bit-exact: pos %d attn n %d hc[0] %v", other.runtime.pos, other.runtime.state.layers[0].attn.n, other.runtime.state.hc[0].HostSlice())
	}
}

func TestMetalRequestStateSnapshotRejectsForeignOwner(t *testing.T) {
	owner := &metalStateAdapter{}
	s := &metalRequestState{runtime: &metalRuntime{ctx: 16, arena: &metalmem.Arena{}}, owner: owner}
	metalStateFixtureIdentity(t, s.runtime)
	snapshot, err := s.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	defer snapshot.Close()
	other := &metalRequestState{runtime: &metalRuntime{ctx: 16, arena: &metalmem.Arena{}}, owner: &metalStateAdapter{}}
	if err := other.Restore(snapshot); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("Restore foreign snapshot: %v, want ErrUnimplemented", err)
	}
}

func TestMetalSnapshotCloseInvalidatesRestore(t *testing.T) {
	owner := &metalStateAdapter{}
	s := &metalRequestState{runtime: &metalRuntime{ctx: 16, arena: &metalmem.Arena{}}, owner: owner}
	metalStateFixtureIdentity(t, s.runtime)
	snapshot, err := s.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	if err := snapshot.Close(); err != nil {
		t.Fatalf("Close snapshot: %v", err)
	}
	// Restore into the source request, so only the closed check can refuse.
	if err := s.Restore(snapshot); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("Restore closed snapshot: %v, want ErrUnimplemented", err)
	}
}

func TestMetalStateAdapterNeverAliasesLegacyRuntime(t *testing.T) {
	var legacy *metalRuntime
	b := legacy.stateBackend()
	for _, context := range []int{2, 128, 4096} {
		state, err := b.NewState(context)
		if !errors.Is(err, dsv4.ErrUnimplemented) || state != nil {
			t.Fatalf("context %d: NewState = (%v, %v), want (nil, ErrUnimplemented)", context, state, err)
		}
	}
	if got := b.Logits(nil); got != nil {
		t.Fatalf("legacy adapter Logits = %v, want nil", got)
	}
}

func TestMetalStateAdapterRejectsIncompleteRuntime(t *testing.T) {
	a := (&metalRuntime{}).stateBackend()
	state, err := a.NewState(4096)
	if !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("NewState error = %v, want dsv4.ErrUnimplemented", err)
	}
	if state != nil {
		t.Fatal("NewState returned a state for an incomplete runtime")
	}
}

func TestMetalRequestRuntimeCloseKeepsBorrowedModel(t *testing.T) {
	f, err := os.CreateTemp("", "ds4-metal-state-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())

	r := &metalRuntime{file: f, borrowed: true}
	if err := r.Close(); err != nil {
		t.Fatalf("Close borrowed runtime: %v", err)
	}
	if r.file != f {
		t.Fatal("borrowed runtime closed the backend-owned model file")
	}
	if _, err := f.Stat(); err != nil {
		t.Fatalf("backend-owned model file is unusable after request close: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestMetalRequestStateClosesOnceAndRejectsUseAfterClose(t *testing.T) {
	a := &metalStateAdapter{}
	s := &metalRequestState{owner: a}
	if err := s.Close(); err != nil {
		t.Fatalf("first Close = %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("second Close = %v", err)
	}
	if err := a.Eval(s, 1, 0); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("Eval after Close = %v, want dsv4.ErrClosed", err)
	}
	if got := a.Logits(s); got != nil {
		t.Fatalf("Logits after Close = %v, want nil", got)
	}
}

func TestMetalStateLoadRejectsInvalidLayerCountWithoutMutation(t *testing.T) {
	r := &metalRuntime{ctx: 16, pos: 3, arena: &metalmem.Arena{}}
	metalStateFixtureIdentity(t, r)
	var saved bytes.Buffer
	if err := saveMetalState(r, &saved); err != nil {
		t.Fatalf("save empty state: %v", err)
	}
	data := append([]byte(nil), saved.Bytes()...)
	const firstLayerCountOffset = 8 + 4 + 4 + 8 + 4
	binary.LittleEndian.PutUint32(data[firstLayerCountOffset:], 1)
	if err := loadMetalState(r, bytes.NewReader(data)); err == nil {
		t.Fatal("load accepted a layer count beyond the zero capacity")
	}
	if r.pos != 3 {
		t.Fatalf("failed load changed position to %d", r.pos)
	}
	for i, layer := range r.state.layers {
		if layer.attn.n != 0 || layer.indexer.n != 0 {
			t.Fatalf("failed load changed layer %d counts: attn=%d indexer=%d", i, layer.attn.n, layer.indexer.n)
		}
	}
}

func TestMetalStateClosedOwnerRejectsAllOperations(t *testing.T) {
	r := &metalRuntime{ctx: 16, arena: &metalmem.Arena{}}
	a := &metalStateAdapter{runtime: r, closed: true}
	s := &metalRequestState{runtime: r, owner: a}
	checks := []struct {
		name string
		run  func() error
	}{
		{"eval", func() error { return a.Eval(s, 1, 0) }},
		{"batch", func() error { return a.EvalBatch(s, nil, 0) }},
		{"reset", s.Reset},
		{"snapshot", func() error { _, err := s.Snapshot(); return err }},
		{"restore", func() error { return s.Restore(&metalSnapshot{owner: a}) }},
		{"save", func() error { return s.Save(&bytes.Buffer{}) }},
		{"load", func() error { return s.Load(bytes.NewReader(nil)) }},
	}
	for _, tt := range checks {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.run(); !errors.Is(err, dsv4.ErrClosed) {
				t.Fatalf("error=%v", err)
			}
		})
	}
	if a.Logits(s) != nil || s.logits() != nil {
		t.Fatal("closed owner exposed logits")
	}
}

func TestMetalStateLoadFailureAndCacheBoundary(t *testing.T) {
	r := &metalRuntime{ctx: 16, pos: 3, arena: &metalmem.Arena{}, ops: &metalPinnedOps{hc: &metalHC{key: metalHCKey{Tokens: 1}}}, lastBatch: metalBatchStats{Encoded: 9}}
	metalStateFixtureIdentity(t, r)
	var saved bytes.Buffer
	if err := saveMetalState(r, &saved); err != nil {
		t.Fatal(err)
	}
	for _, mutation := range []struct {
		name   string
		change func([]byte) []byte
	}{
		{"magic", func(p []byte) []byte { p[0] ^= 1; return p }},
		{"version", func(p []byte) []byte { binary.LittleEndian.PutUint32(p[8:], metalStateVersion+1); return p }},
		{"old-format", func(p []byte) []byte { copy(p, "DS4MST01"); return p }},
		{"model-identity", func(p []byte) []byte { p[metalStateHeaderSize-64] ^= 1; return p }},
		{"quant-identity", func(p []byte) []byte { p[metalStateHeaderSize-32] ^= 1; return p }},
		{"context", func(p []byte) []byte { binary.LittleEndian.PutUint32(p[12:], 17); return p }},
		{"position", func(p []byte) []byte { binary.LittleEndian.PutUint64(p[16:], 17); return p }},
		{"field-count", func(p []byte) []byte { binary.LittleEndian.PutUint32(p[24:], 0); return p }},
		{"field-size", func(p []byte) []byte {
			binary.LittleEndian.PutUint64(p[metalStateHeaderSize+steeringIdentityBytes:], 4)
			return p
		}},
		{"truncated", func(p []byte) []byte { return p[:len(p)-1] }},
		{"trailer", func(p []byte) []byte { return append(p, 1) }},
	} {
		t.Run(mutation.name, func(t *testing.T) {
			p := mutation.change(append([]byte(nil), saved.Bytes()...))
			if loadMetalState(r, bytes.NewReader(p)) == nil {
				t.Fatal("mutation admitted")
			}
			var after bytes.Buffer
			if err := saveMetalState(r, &after); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(saved.Bytes(), after.Bytes()) || r.ops.hc.key.Tokens != 1 || r.lastBatch.Encoded != 9 {
				t.Fatal("rejected load changed state/cache")
			}
		})
	}
	if err := loadMetalState(r, bytes.NewReader(saved.Bytes())); err != nil {
		t.Fatal(err)
	}
	if r.ops.hc.pending != nil || r.ops.hc.key != (metalHCKey{}) || r.lastBatch != (metalBatchStats{}) {
		t.Fatal("restore retained cache")
	}
	r.ops.hc.key.Tokens = 2
	if err := r.Reset(); err != nil {
		t.Fatal(err)
	}
	if r.ops.hc.pending != nil || r.ops.hc.key != (metalHCKey{}) {
		t.Fatal("reset retained cache")
	}
}

func TestMetalStateSynchronizationFailure(t *testing.T) {
	r := &metalRuntime{pos: 7, ops: &metalPinnedOps{hc: &metalHC{pending: &metalBatch{}, key: metalHCKey{Tokens: 2}}}}
	called := false
	if synchronizeMetalState(r, func() error { called = true; return nil }) == nil || called || r.pos != 7 || r.ops.hc.pending == nil || r.ops.hc.key.Tokens != 2 {
		t.Fatal("unsubmitted producer discarded or waited")
	}
	r.ops.hc.pending.done = true
	want := errors.New("injected command failure")
	if err := synchronizeMetalState(r, func() error { return want }); !errors.Is(err, want) || r.pos != 7 || r.ops.hc.pending == nil || r.ops.hc.key.Tokens != 2 {
		t.Fatal("failed wait changed frontier/cache")
	}
	if err := synchronizeMetalState(r, func() error { return nil }); err != nil || r.ops.hc.pending != nil {
		t.Fatal("committed producer not synchronized")
	}
}

func TestMetalStateIdentityFailsClosed(t *testing.T) {
	source := &metalRuntime{ctx: 16, pos: 3, arena: &metalmem.Arena{}}
	metalStateFixtureIdentity(t, source)
	var saved bytes.Buffer
	if err := saveMetalState(source, &saved); err != nil {
		t.Fatal(err)
	}
	destination := &metalRuntime{ctx: 16, pos: 1, arena: &metalmem.Arena{}}
	if err := loadMetalState(destination, bytes.NewReader(saved.Bytes())); err == nil {
		t.Fatal("missing identity admitted")
	}
	if err := saveMetalState(destination, &bytes.Buffer{}); err == nil {
		t.Fatal("missing identity saved")
	}
	metalStateFixtureIdentity(t, destination)
	if _, err := destination.file.WriteAt([]byte("different"), 0); err != nil {
		t.Fatal(err)
	}
	if err := loadMetalState(destination, bytes.NewReader(saved.Bytes())); err == nil {
		t.Fatal("same-shape different model admitted")
	}
	if destination.pos != 1 {
		t.Fatal("rejected identity changed frontier")
	}
}
