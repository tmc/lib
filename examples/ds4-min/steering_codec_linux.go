//go:build linux

package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
)

// cudaStateSteeringHeader validates the persisted control binding before
// parsing mutable fields or allocating a rollback snapshot.
func cudaStateSteeringHeader(data []byte, want steeringIdentity) (int, error) {
	if len(data) < cudaStateHeaderBytes {
		return 0, errors.New("cuda state load: truncated header")
	}
	switch binary.LittleEndian.Uint32(data[8:12]) {
	case 2:
		if want != (steeringIdentity{}) {
			return 0, errors.New("cuda state load: unsteered version 2 cannot restore steered request")
		}
		return cudaStateHeaderBytes, nil
	case cudaStateFormatVersion:
		if err := checkSteeringIdentity(bytes.NewReader(data[cudaStateHeaderBytes:]), want); err != nil {
			return 0, fmt.Errorf("cuda state load steering: %w", err)
		}
		return cudaStateHeaderBytes + steeringIdentityBytes, nil
	default:
		return 0, errors.New("cuda state load: unsupported version")
	}
}
