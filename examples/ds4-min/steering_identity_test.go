package main

import (
	"bytes"
	"crypto/sha256"
	"testing"
)

func TestSteeringIdentity(t *testing.T) {
	direction := &steeringVectors{layers: 2, width: 2, digest: sha256.Sum256([]byte("directions"))}
	loaded := identityForSteering(direction, 0, 1)
	data, err := loaded.encode()
	if err != nil {
		t.Fatal(err)
	}
	if err := checkSteeringIdentity(bytes.NewReader(data[:]), loaded); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name   string
		change func(*steeringIdentity)
	}{
		{"unloaded", func(v *steeringIdentity) { *v = steeringIdentity{} }},
		{"direction", func(v *steeringIdentity) { v.digest[0] ^= 1 }},
		{"attention", func(v *steeringIdentity) { v.attention = 1 }},
		{"ffn", func(v *steeringIdentity) { v.ffn = 0 }},
		{"shape", func(v *steeringIdentity) { v.width++ }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			other := loaded
			tt.change(&other)
			if err := checkSteeringIdentity(bytes.NewReader(data[:]), other); err == nil {
				t.Fatal("accepted mismatched steering identity")
			}
		})
	}
	if err := checkSteeringIdentity(bytes.NewReader(data[:59]), loaded); err == nil {
		t.Fatal("accepted truncated metadata")
	}
	for _, index := range []int{0, 4, 8, 12, 16, 20, 52, 56} {
		bad := data
		bad[index] ^= 1
		if err := checkSteeringIdentity(bytes.NewReader(bad[:]), loaded); err == nil {
			t.Fatalf("accepted mutation at byte %d", index)
		}
	}
}

func TestSteeringIdentityLoadedZeroIsNotUnloaded(t *testing.T) {
	loaded := identityForSteering(&steeringVectors{layers: 1, width: 1, digest: sha256.Sum256([]byte("zero-scale"))}, 0, 0)
	data, err := loaded.encode()
	if err != nil {
		t.Fatal(err)
	}
	if err := checkSteeringIdentity(bytes.NewReader(data[:]), steeringIdentity{}); err == nil {
		t.Fatal("zero scale erased loaded direction identity")
	}
}
