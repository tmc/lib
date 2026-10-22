//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// This gate exercises native persistence after real device writes. Injected
// failures are driver-return surrogates, not naturally occurring GPU faults.
func TestCUDAStateTransactionGate(t *testing.T) {
	if os.Getenv("DS4_STATE_TRANSACTION_GATE") != "1" {
		t.Skip("DS4_STATE_TRANSACTION_GATE is not set")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" || os.Getenv("DS4_MODEL") == "" || os.Getenv("DS4_DISABLE_CUDA_GRAPH") == "1" {
		t.Fatal("GPU lock, model and graphs required")
	}
	f, err := os.Open(os.Getenv("DS4_MODEL"))
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.New()
	_, err = io.Copy(h, f)
	closeErr := f.Close()
	if err != nil || closeErr != nil {
		t.Fatal(errors.Join(err, closeErr))
	}
	const modelSHA = "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"
	if hex.EncodeToString(h.Sum(nil)) != modelSHA {
		t.Fatal("qualified model hash mismatch")
	}
	t.Logf("model_sha256=%s", modelSHA)
	r, err := newCUDARuntime(os.Getenv("DS4_MODEL"), 128)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	b := r.stateBackend()
	state, err := b.NewState(128)
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	s := state.(*cudaRequestState)
	eval := func(token, pos int) {
		t.Helper()
		if err := b.Eval(s, token, pos); err != nil {
			t.Fatal(err)
		}
	}
	save := func() []byte {
		t.Helper()
		var v bytes.Buffer
		if err := s.Save(&v); err != nil {
			t.Fatal(err)
		}
		return v.Bytes()
	}
	eval(0, 0)
	eval(1, 1)
	graphs := s.runtime.graphs
	qkv, ffn := 0, 0
	for i, g := range graphs {
		if g.qkv == nil || i >= 3 && g.ffn == nil || i < 3 && g.ffn != nil {
			t.Fatalf("UNREAD graph census layer=%d qkv=%p ffn=%p", i, g.qkv, g.ffn)
		}
		qkv++
		if g.ffn != nil {
			ffn++
		}
		t.Logf("graph layer=%d qkv_exec=%p ffn_exec=%p", i, g.qkv, g.ffn)
	}
	if qkv != cudaLayers || ffn != cudaLayers-3 {
		t.Fatalf("UNREAD graph census=%d/%d", qkv, ffn)
	}
	graphIdentity := func() {
		t.Helper()
		if graphs != s.runtime.graphs {
			t.Fatal("graph executables changed across restore/replay")
		}
	}
	prefix := save()
	snapshot, err := s.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	eval(2, 2)
	graphIdentity()
	current := save()
	want := float32Bytes(b.Logits(s))
	if bytes.Equal(prefix, current) {
		t.Fatal("fixture did not change state")
	}
	for _, position := range []int{-1, 129} {
		v := snapshot.(*cudaStateSnapshot)
		original := v.pos
		v.pos = position
		err := s.Restore(v)
		v.pos = original
		if err == nil || !bytes.Equal(current, save()) {
			t.Fatalf("bad frontier %d changed destination: %v", position, err)
		}
	}
	for _, off := range []int{40, 72} {
		bad := bytes.Clone(prefix)
		bad[off] ^= 1
		writes := 0
		err := loadCUDAStateWithHooks(s, bytes.NewReader(bad), &cudaStateLoadHooks{afterUpload: func(int) error { writes++; return nil }})
		if err == nil || writes != 0 || !bytes.Equal(current, save()) {
			t.Fatalf("identity mutation offset=%d writes=%d err=%v", off, writes, err)
		}
	}
	copyErr := errors.New("injected after actual upload")
	beforeFields := cudaStateFields(&s.runtime.state, nil)
	beforeFirst, err := cudaStateFieldBytes(beforeFields[0])
	if err != nil {
		t.Fatal(err)
	}
	writes := 0
	changed := false
	err = loadCUDAStateWithHooks(s, bytes.NewReader(prefix), &cudaStateLoadHooks{afterUpload: func(n int) error {
		writes = n
		if n == 2 {
			after, err := cudaStateFieldBytes(cudaStateFields(&s.runtime.state, nil)[0])
			if err != nil {
				return err
			}
			changed = !bytes.Equal(beforeFirst, after)
			return copyErr
		}
		return nil
	}})
	if !errors.Is(err, copyErr) || writes != 2 || !changed || !bytes.Equal(current, save()) {
		t.Fatalf("partial-write rollback writes=%d err=%v", writes, err)
	}
	if err := s.Load(bytes.NewReader(prefix)); err != nil {
		t.Fatal(err)
	}
	graphIdentity()
	if !bytes.Equal(prefix, save()) {
		t.Fatal("native Save Load Save differs")
	}
	eval(2, 2)
	graphIdentity()
	if !bytes.Equal(want, float32Bytes(b.Logits(s))) {
		t.Fatal("Load graph replay continuation differs")
	}
	if err := s.Restore(snapshot); err != nil {
		t.Fatal(err)
	}
	graphIdentity()
	eval(2, 2)
	graphIdentity()
	if !bytes.Equal(want, float32Bytes(b.Logits(s))) {
		t.Fatal("Snapshot Restore graph replay differs")
	}
	rollbackErr := errors.New("injected rollback failure")
	err = loadCUDAStateWithHooks(s, bytes.NewReader(prefix), &cudaStateLoadHooks{afterUpload: func(n int) error {
		if n == 2 {
			return copyErr
		}
		return nil
	}, beforeRollback: func() error { return rollbackErr }})
	if !errors.Is(err, copyErr) || !errors.Is(err, rollbackErr) || !errors.Is(err, dsv4.ErrDesynchronized) {
		t.Fatal("failed rollback did not join and poison", err)
	}
	for _, check := range []func() error{func() error { return b.Eval(s, 3, 3) }, func() error { return s.Save(new(bytes.Buffer)) }, s.Reset, func() error { return s.Restore(snapshot) }} {
		if err := check(); !errors.Is(err, dsv4.ErrDesynchronized) {
			t.Fatalf("poisoned state operation err=%v", err)
		}
	}
	t.Logf("payload_bytes=%d partial_uploads=2 continuation_logits=%d graphs=on", len(prefix), len(want)/4)
}
