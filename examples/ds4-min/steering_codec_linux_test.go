//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"strings"
	"testing"
)

func TestCUDAStateSteeringLoadRejectsBeforeUpload(t *testing.T) {
	identity := identityForSteering(&steeringVectors{layers: 1, width: 1, digest: sha256.Sum256([]byte("directions"))}, 0, 1)
	r := &cudaRuntime{ctx: 16, pos: 3, steering: &cudaSteering{identity: identity}, logits: []float32{7}}
	s := &cudaRequestState{runtime: r, lifecycleEpoch: 23}
	data := make([]byte, cudaStateHeaderBytes+steeringIdentityBytes)
	copy(data[:8], cudaStateMagic[:])
	binary.LittleEndian.PutUint32(data[8:12], cudaStateFormatVersion)
	meta, err := identity.encode()
	if err != nil {
		t.Fatal(err)
	}
	copy(data[cudaStateHeaderBytes:], meta[:])
	for i := 0; i < steeringIdentityBytes; i++ {
		bad := append([]byte(nil), data...)
		bad[cudaStateHeaderBytes+i] ^= 1
		err := loadCUDAStateWithHooks(s, bytes.NewReader(bad), &cudaStateLoadHooks{afterUpload: func(int) error { t.Fatal("uploaded mismatched controls"); return nil }})
		if err == nil || !strings.Contains(err.Error(), "steering") {
			t.Fatalf("metadata %d: %v", i, err)
		}
		if r.pos != 3 || r.logits[0] != 7 || s.lifecycleEpoch != 23 {
			t.Fatal("rejected metadata mutated request")
		}
	}
}

func TestCUDAStateSteeringSnapshotRejectsBeforeWrite(t *testing.T) {
	identity := identityForSteering(&steeringVectors{layers: 1, width: 1, digest: sha256.Sum256([]byte("directions"))}, 0, 0)
	for _, change := range []string{"control", "directions", "unloaded"} {
		t.Run(change, func(t *testing.T) {
			r := &cudaRuntime{ctx: 16, pos: 3, steering: &cudaSteering{identity: identity}, logits: []float32{7}}
			s := &cudaRequestState{runtime: r, lifecycleEpoch: 23}
			other := identity
			switch change {
			case "control":
				other.ffn = 1
			case "directions":
				other.digest[0] ^= 1
			case "unloaded":
				other = steeringIdentity{}
			}
			snapshot := &cudaStateSnapshot{owner: r, steering: other, pos: 9}
			if err := s.Restore(snapshot); err == nil || !strings.Contains(err.Error(), "steering identity mismatch") {
				t.Fatalf("restore rejected after steering prewrite boundary: %v", err)
			}
			if r.pos != 3 || r.logits[0] != 7 || s.lifecycleEpoch != 23 {
				t.Fatal("rejected snapshot mutated request")
			}
		})
	}
}

func TestCUDAStateSteeringVersionAdmission(t *testing.T) {
	loaded := identityForSteering(&steeringVectors{layers: 1, width: 1, digest: sha256.Sum256([]byte("vector"))}, 0, 0)
	data := make([]byte, cudaStateHeaderBytes+steeringIdentityBytes)
	binary.LittleEndian.PutUint32(data[8:12], 2)
	if n, err := cudaStateSteeringHeader(data, steeringIdentity{}); err != nil || n != cudaStateHeaderBytes {
		t.Fatalf("v2 offset %d error %v", n, err)
	}
	if _, err := cudaStateSteeringHeader(data, loaded); err == nil {
		t.Fatal("v2 admitted loaded zero-scale request")
	}
	binary.LittleEndian.PutUint32(data[8:12], cudaStateFormatVersion)
	meta, err := loaded.encode()
	if err != nil {
		t.Fatal(err)
	}
	copy(data[cudaStateHeaderBytes:], meta[:])
	if n, err := cudaStateSteeringHeader(data, loaded); err != nil || n != len(data) {
		t.Fatalf("v3 offset %d error %v", n, err)
	}
	for i := 0; i < steeringIdentityBytes; i++ {
		mutant := append([]byte(nil), data...)
		mutant[cudaStateHeaderBytes+i] ^= 1
		if _, err := cudaStateSteeringHeader(mutant, loaded); err == nil {
			t.Fatalf("admitted field mutation %d", i)
		}
	}
	if _, err := cudaStateSteeringHeader(data[:len(data)-1], loaded); err == nil {
		t.Fatal("admitted truncated v3 header")
	}
}
