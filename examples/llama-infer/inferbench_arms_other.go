//go:build !darwin

package main

import "fmt"

func configureBenchmarkArm(_ *ModelEngine, arm string) error {
	if arm != "llama-infer" && arm != "llama-infer-observed-output" {
		return fmt.Errorf("inferbench: unsupported arm %q", arm)
	}
	return nil
}
