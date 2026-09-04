//go:build darwin && arm64

package main

import (
	"errors"
	"fmt"
)

func selectBackend(name string) (string, error) {
	switch name {
	case "auto", "metal":
		return "metal", nil
	case "cpu":
		return "cpu", nil
	case "cuda":
		return "", errors.New("cuda backend unsupported on darwin/arm64")
	default:
		return "", fmt.Errorf("unknown backend %q", name)
	}
}
