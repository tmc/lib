package main

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalineinfo"
)

// control recompiles the first kernel with the #line directives withheld. If
// the Go attribution survives that, it was never coming from the directives
// and the tables above prove nothing.
func control(ctx context.Context, opts cudalineinfo.Options) error {
	k := kernels[0]
	with, _, err := attribute(ctx, k, opts)
	if err != nil {
		return err
	}
	opts.NoDirectives = true
	without, _, err := attribute(ctx, k, opts)
	if err != nil {
		return err
	}

	fmt.Printf("\ncontrol (%s): SASS instructions naming a Go file\n", k.name)
	fmt.Printf("  with #line directives:    %d of %d\n", with.Attributed, with.Instructions)
	fmt.Printf("  without #line directives: %d of %d\n", without.Attributed, without.Instructions)
	if without.Attributed != 0 {
		return fmt.Errorf("control failed: %d instructions named a Go file with the directives withheld", without.Attributed)
	}
	if with.Attributed == 0 {
		return fmt.Errorf("control failed: no instruction named a Go file with the directives present")
	}
	return nil
}
