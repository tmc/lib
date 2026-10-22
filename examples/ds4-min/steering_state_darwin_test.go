//go:build darwin && arm64

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"testing"

	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestMetalSteeringCloseRetryRetention(t *testing.T) {
	failure := errors.New("stream sync failed")
	r := &metalRuntime{steering: &metalSteering{}, arena: &metalmem.Arena{}}
	s := &metalRequestState{runtime: r}
	if err := s.finishClose(failure); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	if s.closed || s.runtime != r || s.runtime.steering == nil {
		t.Fatal("early close failure lost retryable resources")
	}
	if err := s.Reset(); !errors.Is(err, dsv4.ErrDesynchronized) {
		t.Fatalf("terminal request reset: %v", err)
	}
	if err := s.setSteeringFFN(1); !errors.Is(err, failure) {
		t.Fatalf("terminal request changed control: %v", err)
	}
	if err := s.loadSteering(nil, 0, 0); !errors.Is(err, failure) {
		t.Fatalf("terminal request admitted steering: %v", err)
	}
	if s.logits() != nil {
		t.Fatal("terminal request exposed logits")
	}
	r.steering = nil
	if err := s.finishClose(nil); err != nil {
		t.Fatal(err)
	}
	if !s.closed || s.runtime != nil {
		t.Fatal("successful close retained resources")
	}
}

func TestMetalSteeringCodecRejectsBeforeWrite(t *testing.T) {
	r := &metalRuntime{ctx: 16, pos: 7, arena: &metalmem.Arena{}}
	metalStateFixtureIdentity(t, r)
	identity := identityForSteering(&steeringVectors{layers: 1, width: 1, digest: sha256.Sum256([]byte("vector"))}, 0, 0)
	r.steering = &metalSteering{identity: identity}
	var saved bytes.Buffer
	if err := saveMetalState(r, &saved); err != nil {
		t.Fatal(err)
	}
	r.pos = 3
	for i := 0; i < steeringIdentityBytes; i++ {
		data := append([]byte(nil), saved.Bytes()...)
		data[metalStateHeaderSize+i] ^= 1
		if err := loadMetalState(r, bytes.NewReader(data)); err == nil {
			t.Fatalf("accepted metadata mutation %d", i)
		}
		if r.pos != 3 {
			t.Fatal("rejected metadata changed destination")
		}
	}
	if err := loadMetalState(r, bytes.NewReader(saved.Bytes())); err != nil {
		t.Fatal(err)
	}
	if r.pos != 7 {
		t.Fatalf("matching v3 position %d", r.pos)
	}
	legacy := append([]byte(nil), saved.Bytes()[:metalStateHeaderSize]...)
	legacy = append(legacy, saved.Bytes()[metalStateHeaderSize+steeringIdentityBytes:]...)
	binary.LittleEndian.PutUint32(legacy[8:12], 2)
	r.pos = 3
	if err := loadMetalState(r, bytes.NewReader(legacy)); err == nil {
		t.Fatal("v2 restored loaded zero-scale request")
	}
	if r.pos != 3 {
		t.Fatal("rejected v2 changed destination")
	}
	r.steering = nil
	if err := loadMetalState(r, bytes.NewReader(legacy)); err != nil {
		t.Fatal(err)
	}
	if r.pos != 7 {
		t.Fatalf("unsteered legacy position %d", r.pos)
	}
}

func TestMetalSteeringSnapshotRejectsChangedControls(t *testing.T) {
	r := &metalRuntime{ctx: 16, pos: 7, arena: &metalmem.Arena{}}
	metalStateFixtureIdentity(t, r)
	identity := identityForSteering(&steeringVectors{layers: 1, width: 1, digest: sha256.Sum256([]byte("vector"))}, 0, 1)
	r.steering = &metalSteering{identity: identity}
	s := &metalRequestState{runtime: r, owner: &metalStateAdapter{runtime: r}}
	snapshot, err := s.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	r.pos = 3
	r.steering.identity.ffn = 0
	if err := s.Restore(snapshot); err == nil {
		t.Fatal("restored snapshot with changed controls")
	}
	if r.pos != 3 {
		t.Fatal("rejected snapshot changed position")
	}
	r.steering.identity = identity
	if err := s.Restore(snapshot); err != nil {
		t.Fatal(err)
	}
	if r.pos != 7 {
		t.Fatal("matching snapshot did not restore position")
	}
}
