//go:build !darwin && !linux

package model

import "os"

func mapFile(name string) ([]byte, func() error, error) {
	data, err := os.ReadFile(name)
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return nil }, nil
}
