//go:build !darwin

package main

import "fmt"

func configureBenchmarkArm(_ *ModelEngine, arm string) error {
	if arm != "llama-infer" {
		return fmt.Errorf("inferbench: unsupported arm %q", arm)
	}
	return nil
}
