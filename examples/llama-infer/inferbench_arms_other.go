//go:build !darwin

package main

import "fmt"

func configureBenchmarkArm(eng *ModelEngine, arm string) error {
	serial := false
	switch arm {
	case "llama-infer", "llama-infer-observed-output":
	case "llama-infer-serial", "llama-infer-serial-observed-output":
		serial = true
	default:
		return fmt.Errorf("inferbench: unsupported arm %q", arm)
	}
	if eng != nil {
		eng.benchmarkSerial = serial
	}
	return nil
}
