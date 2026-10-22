package dsv41

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"io"
	"testing"
)

func TestNativeContextIdentityRollback(t *testing.T) {
	s := payloadFixture(t, 1)
	b, err := NewBackend(s.model)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	})
	raw, err := b.NewState(132)
	if err != nil {
		t.Fatal(err)
	}
	r := raw.(*request)
	r.core = s
	var original bytes.Buffer
	if err := r.Save(&original); err != nil {
		t.Fatal(err)
	}
	bad := append([]byte(nil), original.Bytes()...)
	// Change only the embedded source context and its dependent row capacity.
	// Re-seal so a checksum mismatch cannot mask the identity control.
	binary.LittleEndian.PutUint32(bad[84+8:], 131)
	binary.LittleEndian.PutUint32(bad[84+24:], 132)
	sum := sha256.Sum256(bad[84:])
	copy(bad[40:72], sum[:])
	before := r.core
	if err := r.Load(bytes.NewReader(bad)); err == nil {
		t.Fatal("native context disagreement accepted")
	}
	if r.core != before {
		t.Fatal("failed load replaced request owner")
	}
	var after bytes.Buffer
	if err := r.Save(&after); err != nil || !bytes.Equal(after.Bytes(), original.Bytes()) {
		t.Fatal("failed context load changed state", err)
	}
	// The upstream raw format intentionally permits a larger target context.
	next, err := b.NewState(133)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := next.Close(); err != nil {
			t.Error(err)
		}
	})
	if err := next.(*request).core.RestorePayload(bytes.NewReader(original.Bytes()[84:]), int64(len(original.Bytes())-84), s.model.AssetSHA256()); err != nil {
		t.Fatal("raw upstream context resize rejected", err)
	}
}

type continuationReadFault struct{ remaining int }

var errContinuationRead = errors.New("continuation read fault")

func (r *continuationReadFault) ReadAt(dst []byte, _ int64) (int, error) {
	if r.remaining == 0 {
		return 0, errContinuationRead
	}
	if r.remaining > 0 {
		r.remaining--
	}
	clear(dst)
	return len(dst), nil
}

func TestNativeLoadPartialOwnerRollback(t *testing.T) {
	reader := &continuationReadFault{remaining: -1}
	g := modelFixture(t)
	m, err := Bind(reader, int64(g.Size), g)
	if err != nil {
		t.Fatal(err)
	}
	fixture := payloadFixture(t, 1)
	m.assetSHA = fixture.model.assetSHA
	fixture.model = m
	b, err := NewBackend(m)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	})
	xi, err := b.NewState(132)
	if err != nil {
		t.Fatal(err)
	}
	yi, err := b.NewState(132)
	if err != nil {
		t.Fatal(err)
	}
	x, y := xi.(*request), yi.(*request)
	x.core = fixture
	var wire bytes.Buffer
	if err := x.Save(&wire); err != nil {
		t.Fatal(err)
	}
	if err := y.Load(bytes.NewReader(wire.Bytes())); err != nil {
		t.Fatal(err)
	}
	before := y.core
	// NewEngram reads one gate-weight row, then fails on the next one.
	reader.remaining = 1
	if err := y.Load(bytes.NewReader(wire.Bytes())); !errors.Is(err, errContinuationRead) {
		t.Fatalf("load did not preserve acquisition error: %v", err)
	}
	reader.remaining = -1
	if y.core != before || !b.states[x] || !b.states[y] || len(b.states) != 2 {
		t.Fatal("failed load changed request ownership")
	}
	var after bytes.Buffer
	if err := y.Save(&after); err != nil || !bytes.Equal(after.Bytes(), wire.Bytes()) {
		t.Fatal("failed acquisition changed destination state", err)
	}
	// A short private envelope also leaves the exact same owned core live.
	if err := y.Load(io.LimitReader(bytes.NewReader(wire.Bytes()), 83)); err == nil || y.core != before {
		t.Fatal("short envelope changed destination")
	}
}
