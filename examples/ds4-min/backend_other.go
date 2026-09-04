//go:build !linux && (!darwin || !arm64)

package main

import (
	"errors"
	"fmt"
)

func selectBackend(name string) (string, error) {
	switch name {
	case "cpu":
		return "cpu", nil
	case "auto":
		return "", errors.New("no automatic GPU backend available")
	case "metal", "cuda":
		return "", fmt.Errorf("%s backend unsupported on this platform", name)
	default:
		return "", fmt.Errorf("unknown backend %q", name)
	}
}
