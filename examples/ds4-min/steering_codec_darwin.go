//go:build darwin && arm64

package main

import (
	"errors"
	"fmt"
	"io"
)

func checkMetalSteeringVersion(r io.Reader, version uint32, want steeringIdentity) error {
	if version == 2 {
		if want != (steeringIdentity{}) {
			return errors.New("Metal state load: unsteered version 2 cannot restore steered request")
		}
		return nil
	}
	if version != metalStateVersion {
		return errors.New("Metal state load: unsupported version")
	}
	if err := checkSteeringIdentity(r, want); err != nil {
		return fmt.Errorf("Metal state load steering: %w", err)
	}
	return nil
}
