//go:build darwin

package main

import "fmt"

func configureBenchmarkArm(eng *ModelEngine, arm string) error {
	var disable bool
	switch arm {
	case "llama-infer", "llama-infer-observed-output":
	case "llama-infer-unfused-projections", "llama-infer-unfused-projections-observed-output":
		disable = true
	default:
		return fmt.Errorf("inferbench: unsupported arm %q", arm)
	}
	if eng != nil {
		eng.disableProjectionFusion = disable
	}
	return nil
}
