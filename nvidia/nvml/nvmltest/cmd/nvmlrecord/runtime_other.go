//go:build !linux

package main

import (
	"encoding/json"
	"errors"
)

func runtimeProofSupported() error             { return errors.New("native runtime proof requires Linux") }
func captureRuntime() (json.RawMessage, error) { return nil, runtimeProofSupported() }
