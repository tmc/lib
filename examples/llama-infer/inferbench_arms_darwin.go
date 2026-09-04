//go:build darwin

package main

import "fmt"

func configureBenchmarkArm(eng *ModelEngine, arm string) error {
	var disable bool
	switch arm {
	case "llama-infer":
	case "llama-infer-unfused-projections":
		disable = true
	default:
		return fmt.Errorf("inferbench: unsupported arm %q", arm)
	}
	if eng != nil {
		eng.disableProjectionFusion = disable
	}
	return nil
}
