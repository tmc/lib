package main

import (
	"encoding/binary"
	"errors"
	"io"
	"math"
)

// steeringIdentity binds mutable state to its loaded directions and current
// controls. The directions themselves remain request-owned, not serialized
// device addresses. Restore requires the same directions already loaded.
type steeringIdentity struct {
	loaded         bool
	layers, width  uint32
	digest         [32]byte
	attention, ffn float32
}

const steeringIdentityBytes = 60

func identityForSteering(v *steeringVectors, attention, ffn float32) steeringIdentity {
	if v == nil {
		return steeringIdentity{}
	}
	return steeringIdentity{loaded: true, layers: uint32(v.layers), width: uint32(v.width), digest: v.digest, attention: attention, ffn: ffn}
}

func (v steeringIdentity) encode() ([steeringIdentityBytes]byte, error) {
	var data [steeringIdentityBytes]byte
	if err := v.validate(); err != nil {
		return data, err
	}
	copy(data[:4], "DS4D")
	binary.LittleEndian.PutUint32(data[4:8], 1)
	if v.loaded {
		binary.LittleEndian.PutUint32(data[8:12], 1)
	}
	binary.LittleEndian.PutUint32(data[12:16], v.layers)
	binary.LittleEndian.PutUint32(data[16:20], v.width)
	copy(data[20:52], v.digest[:])
	binary.LittleEndian.PutUint32(data[52:56], math.Float32bits(v.attention))
	binary.LittleEndian.PutUint32(data[56:60], math.Float32bits(v.ffn))
	return data, nil
}

func (v steeringIdentity) validate() error {
	if !v.loaded {
		if v != (steeringIdentity{}) {
			return errors.New("unloaded steering has nonzero metadata")
		}
		return nil
	}
	if v.layers == 0 || v.width == 0 || v.digest == ([32]byte{}) {
		return errors.New("invalid loaded steering identity")
	}
	return errors.Join(steeringScale(v.attention), steeringScale(v.ffn))
}

// checkSteeringIdentity consumes only the fixed metadata, before any mutable
// buffer is restored. Mismatched loaded state, direction or scales is rejected.
func checkSteeringIdentity(r io.Reader, want steeringIdentity) error {
	expected, err := want.encode()
	if err != nil {
		return err
	}
	var actual [steeringIdentityBytes]byte
	if _, err := io.ReadFull(r, actual[:]); err != nil {
		return err
	}
	if actual != expected {
		return errors.New("steering state identity does not match request")
	}
	return nil
}
